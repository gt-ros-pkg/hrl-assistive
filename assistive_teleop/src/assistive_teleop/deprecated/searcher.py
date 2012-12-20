#!/usr/bin/python

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
import math
import decimal
import numpy as np
import tf
from tf import transformations as tft
from geometry_msgs.msg import (PoseWithCovarianceStamped, PointStamped, 
                                Twist, Point32, PoseStamped)
from sensor_msgs.msg import LaserScan, PointCloud
from std_msgs.msg import String

class Follower:
    def __init__(self):
        rospy.init_node('relative_servoing')
        rospy.Subscriber('robot_pose_ekf/odom_combined', 
                         PoseWithCovarianceStamped, 
                         self.update_position)
        rospy.Subscriber('/base_scan', LaserScan, self.update_base_laser)
        rospy.Subscriber('/servo_goal', PoseStamped, self.update_goal)
        self.vel_out = rospy.Publisher('base_controller/command', Twist)
        self.rate_test = rospy.Publisher('rate_test', String)
        self.tfl = tf.TransformListener()
        self.goal_out = rospy.Publisher('/servo_dest', PoseStamped, latch=True)
        self.left_out = rospy.Publisher('/left_points', PointCloud)
        self.right_out = rospy.Publisher('/right_points', PointCloud)
        self.front_out = rospy.Publisher('/front_points', PointCloud)
        #Initialize variables, so as not to spew errors before seeing a goal
        self.command = Twist()
        self.goal_received = False
        self.goal_present = False
        self.rot_safe = True
        self.bfp_goal = PoseStamped()
        self.odom_goal = PoseStamped()
        self.x_max = 0.5
        self.x_min = 0.05
        self.y_man = 0.3
        self.y_min = 0.05
        self.z_max = 0.5
        self.z_min = 0.05
        self.ang_goal = 0.0
        self.ang_thresh_small = 0.01
        self.ang_thresh_big = 0.04
        self.ang_thresh = self.ang_thresh_big
        self.retreat_thresh = 0.3
        self.curr_pos = PoseWithCovarianceStamped()
        self.dist_thresh = 0.15
        self.left = [[],[]]
        self.right = [[],[]]
        self.front = [[],[]]

    def update_goal(self, msg):
        msg.header.stamp = rospy.Time.now()
                
        self.tfl.waitForTransform(msg.header.frame_id, '/base_footprint',
                                  msg.header.stamp, rospy.Duration(0.5))
        self.bfp_goal = self.tfl.transformPose('/base_footprint', msg)
        self.tfl.waitForTransform(msg.header.frame_id, 'odom_combined',
                                  msg.header.stamp, rospy.Duration(0.5))
        self.odom_goal = self.tfl.transformPose('/odom_combined', msg)
        self.goal_out.publish(self.odom_goal)
        
        ang_to_goal = math.atan2(self.bfp_goal.pose.position.y,
                                 self.bfp_goal.pose.position.x)
        #The desired angular position 
        #(current angle in odom, plus the robot-relative change to face goal)
        self.ang_goal = self.curr_ang[2] + ang_to_goal
        self.goal_received = True
        print "New Goal: \r\n %s" %self.bfp_goal

    def update_position(self, msg):
        self.curr_ang=tft.euler_from_quaternion([msg.pose.pose.orientation.x,
                                                 msg.pose.pose.orientation.y,
                                                 msg.pose.pose.orientation.z,
                                                 msg.pose.pose.orientation.w])
        # Normalized via unwrap relative to 0; (keeps between -pi/pi)
        self.ang_diff = np.unwrap([0, self.ang_goal - self.curr_ang[2]])[1]
        #print "Ang Diff: %s" %self.ang_diff

        self.dist_to_goal = ((self.odom_goal.pose.position.x-
                              msg.pose.pose.position.x)**2 + 
                              (self.odom_goal.pose.position.y-
                              msg.pose.pose.position.y)**2)**(1./2.)
        #print "Dist to goal: %s" %self.dist_to_goal

        if self.goal_received:
            if ((self.dist_to_goal < self.dist_thresh) and 
                (abs(self.ang_diff) < self.ang_thresh)):
                self.goal_present = False
                self.goal_received = False
                print "Achieved Goal!"
            else:
                self.goal_present = True

    def update_base_laser(self, msg):
        max_angle = msg.angle_max
        ranges = np.array(msg.ranges)
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        #Filter out noise(<0.003), points >1m, leaves obstacles
        near_angles = np.extract(np.logical_and(ranges<1, ranges>0.003),
                                 angles)
        near_ranges = np.extract(np.logical_and(ranges<1, ranges>0.003),
                                 ranges)
        self.bad_side = np.sign(near_angles[np.argmax(abs(near_angles))])
        #print "bad side: %s" %bad_side # (1 (pos) = left, -1 = right)
        #print "Min in Ranges: %s" %min(ranges)
       
        #if len(near_ranges) > 0:
        xs = near_ranges * np.cos(near_angles)
        ys = near_ranges * np.sin(near_angles)
        #print "xs: %s" %xs
        self.points = np.vstack((xs,ys))
        #print "Points: %s" %points
        self.bfp_points = np.vstack((np.add(0.275, xs),ys))
        #print "bfp Points: %s" %bfp_points
        self.bfp_dists = np.sqrt(np.add(np.square(self.bfp_points[0][:]),
                                        np.square(self.bfp_points[1][:])))
        #print min(self.bfp_dists)
        if len(self.bfp_dists) >0:
            if min(self.bfp_dists) > 0.5:
                self.rot_safe = True
            else:
                self.rot_safe = False
        else:
            self.rot_safe = True
        
        self.left = np.vstack((xs[np.nonzero(ys>0.35)[0]],
                               ys[np.nonzero(ys>0.35)[0]]))
        self.right= np.vstack((xs[np.nonzero(ys<-0.35)[0]],
                                ys[np.nonzero(ys<-0.35)[0]]))
        self.front = np.vstack(
                          (np.extract(np.logical_and(ys<0.35,ys>-0.35),xs),
                           np.extract(np.logical_and(ys<0.35, ys>-0.35),ys)))

        front_dist = (self.front[:][0]**2+self.front[:][1]**2)**(1/2)

        ##Testing and Visualization:###
        if len(self.left[:][0]) > 0:
            leftScan  = PointCloud()
            leftScan.header.frame_id = '/base_laser_link'
            leftScan.header.stamp = rospy.Time.now()
        
            for i in range(len(self.left[0][:])):
                pt = Point32()
                pt.x = self.left[0][i]
                pt.y = self.left[1][i]
                pt.z = 0
                leftScan.points.append(pt)
            
            self.left_out.publish(leftScan)

        if len(self.right[:][0]) > 0:
            rightScan = PointCloud()
            rightScan.header.frame_id = '/base_laser_link'
            rightScan.header.stamp = rospy.Time.now()

            for i in range(len(self.right[:][0])):
                pt = Point32()
                pt.x = self.right[0][i]
                pt.y = self.right[1][i]
                pt.z = 0
                rightScan.points.append(pt)
            
            self.right_out.publish(rightScan)

        if len(self.front[:][0]) > 0:
            frontScan = PointCloud()
            frontScan.header.frame_id = 'base_laser_link'
            frontScan.header.stamp = rospy.Time.now()
            
            for i in range(len(self.front[:][0])):
                pt = Point32()
                pt.x = self.front[0][i]
                pt.y = self.front[1][i]
                pt.z = 0
                frontScan.points.append(pt)
            
            self.front_out.publish(frontScan)

    def set_rot(self):
        if abs(self.ang_diff) < self.ang_thresh: 
            #Fully oriented, relax constraint to avoid osscilation
            self.ang_thresh = self.ang_thresh_big
            self.command.angular.z = 0.0
        else: 
            #Not oriented, continue until pointed within small constraint
            if self.rot_safe: # Check for obstacles in the rotation radius
                self.command.angular.z = (np.sign(self.ang_diff)*
                                          np.clip(abs(0.35*self.ang_diff),
                                                  0.1, 0.5))
                self.ang_thresh = self.ang_thresh_small
            else:
                print "Cannot Rotate, obstacles nearby"

    def set_trans(self):
        if (abs(self.ang_diff) < math.pi/6 and
            self.dist_to_goal > self.dist_thresh):
            #Facing the right direction and not there yet, so start moving.
            self.command.linear.x=np.clip(self.dist_to_goal*0.125, 0.05, 0.3)
        else:
            self.command.linear.x = 0.0

    def slide(self):
       # Determine left/right movement speed for strafing obstacle avoidance 
        push_from_left = push_from_right = 0.0
        #print "LEFT: \r\n %s" %self.left
        if len(self.left[:][0]) > 0:
            lefts = np.extract(self.left[:][1]<0.45, self.left[:][1])
           # print "lefts:\r\n %s" %lefts
            if len(lefts) > 0:
                #left_dists = np.sqrt(np.add(np.square(lefts[:][0]),
                #                            np.square(lefts[:][1])))
                #push_from_left = np.sum(np.reciprocal(np.exp(lefts)))
                push_from_left = -0.45 + min(lefts)
                #print "lpush: \r\n %s" %push_from_left
        if len(self.right[:][0]) > 0:
            rights = np.extract(self.right[:][1]>-0.45, self.right[:][1])
           # print "rights:\r\n %s" %rights
            if len(rights) > 0:
                #right_dists = np.sqrt(np.add(np.square(rights[:][0]),
                #                             np.square(rights[:][1])))
                #push_from_right = np.sum(np.reciprocal(np.exp(rights)))
                push_from_right = 0.45 + max(rights)
                #print "rpush: \r\n %s" %push_from_right
        #print "Push: %s" %(push_from_right + push_from_left)
        slide = push_from_right + push_from_left
        self.command.linear.y = np.sign(slide)*np.clip(abs(slide), 0.04, 0.07)
        print "Slide speed (m/s): %s" %slide

        ##Determine rotation to avoid obstacles in front of robot#
    def avoid_obstacles(self):
        if len(self.front[0][:]) > 0:
            if min(self.front[0][:]) < self.retreat_thresh: 
                #(round-up on corner-to-corner radius of robot) -
                # 0.275 (x diff from base laser link to base footprint)
                #print "front[0][:] %s" %self.front[0][:]
                front_dists = np.sqrt(np.add(np.square(self.front[0][:]),
                                             np.square(self.front[1][:])))
                min_x = self.front[0][np.argmin(front_dists)]
                min_y = self.front[1][np.argmin(front_dists)]
                #print "min x/y: %s,%s" %(min_x, min_y)
                self.command.linear.x = (-np.sign(min_x)*
                                          np.clip(abs(min_x),0.05,0.1))
                self.command.linear.y = (-np.sign(min_y)*
                                          np.clip(abs(min_y),0.05,0.1)) 
                self.command.angular.z = 0.0
                # This should probably be avoided...
                print "TOO CLOSE: Back up slowly..." 
                self.retreat_thresh = 0.4
            elif min(self.front[0][:]) < 0.45: 
                self.retreat_thresh = 0.3
                print "Turning Away from obstacles in front"
                #self.command.linear.x = 0.0
               # front_dists = np.sqrt(np.add(np.square(self.front[0][:]),
               #                              np.square(self.front[1][:])))
               # min_x = self.front[0][np.argmin(front_dists)]
               # min_y = self.front[1][np.argmin(front_dists)]
               # min_ang = math.atan2(min_y, min_x)
               # print "min_ang: %s" %min_ang
               # self.command.angular.z 

                lfobs = self.front[0][np.logical_and(self.front[1]>0,
                                                     self.front[0]<0.45)]
                rfobs = self.front[0][np.logical_and(self.front[1]<0,
                                                     self.front[0]<0.45)]
               # print "lfobs: %s" %lfobs
                if len(lfobs) == 0:
                    self.command.linear.y = 0.07
                #print "rfobs: %s" %rfobs
                elif len(rfobs) == 0:
                    self.command.linear.y += -0.07
                weight = np.reciprocal(np.sum(np.reciprocal(rfobs)) -
                                       np.sum(np.reciprocal(lfobs)))
                if weight > 0:
                    self.command.angular.z += 0.05 
                else:
                    self.command.angular.z += -0.05
            else:
                self.retreat_thresh = 0.3

    def left_clear(self): # Base Laser cannot see obstacles beside the back edge of the robot, which could cause problems, especially just after passing through doorways...
        if len(self.left[0][:])>0:
            #Find points directly to the right or left of the robot (not in front or behind)
            # x<0.1 (not in front), x>-0.8 (not behind)
            left_obs = self.left[:, self.left[1,:]<0.4] #points too close.
            if len(left_obs[0][:])>0:
                left_obs = left_obs[:, np.logical_and(left_obs[0,:]<0.1,
                                                      left_obs[0,:]>-0.8)]
                if len(left_obs[:][0])>0:
                    print "Obstacle immediately to the left, cannot move."
                    self.command.linear.y = 0

    def right_clear(self):
        if len(self.right[0][:])>0:
            #Find points directly to the right or left of the robot (not in front or behind)
            # x<0.1 (not in front), x>-0.8 (not behind)
           right_obs = self.right[:, self.right[1,:]>-0.4] #points too close.
           if len(right_obs[0][:])>0:
                right_obs = right_obs[:, np.logical_and(right_obs[0,:]<0.1,
                                                        right_obs[0,:]>-0.8)]
                if len(right_obs[:][0])>0:
                    print "Obstacle immediately to the right, cannot move."
                    self.command.linear.y = 0

    def run(self):
        if self.goal_present:
            self.set_rot()
            self.set_trans()
            self.slide()
            self.avoid_obstacles()
            if self.command.linear.y > 0:
                self.left_clear()
            elif self.command.linear.y < 0:
                self.right_clear()
            print "Sending vel_command: \r\n %s" %self.command
            self.vel_out.publish(self.command)
        else:
            pass

if __name__ == '__main__':
    Follower = Follower()
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        Follower.run()
        r.sleep()
