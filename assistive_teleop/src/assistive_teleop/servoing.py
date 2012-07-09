#!/usr/bin/env python

import math
import decimal
import numpy as np

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
import actionlib
from std_msgs.msg import String, Bool
from geometry_msgs.msg import (PoseWithCovarianceStamped, PointStamped, 
                                Twist, Point32, PoseStamped)
from tf import TransformListener, transformations as tft
from sensor_msgs.msg import LaserScan, PointCloud

from assistive_teleop.msg import ServoAction, ServoResult, ServoFeedback

class ServoingServer(object):
    def __init__(self):
        rospy.init_node('relative_servoing')
        rospy.Subscriber('robot_pose_ekf/odom_combined', 
                         PoseWithCovarianceStamped, 
                         self.update_position)
        rospy.Subscriber('/base_scan', LaserScan, self.base_laser_cb)
        self.servoing_as = actionlib.SimpleActionServer('servoing_action', 
                                                        ServoAction,
                                                        self.goal_cb, False)
        self.vel_out = rospy.Publisher('base_controller/command', Twist)
        self.tfl = TransformListener()
        self.goal_out = rospy.Publisher('/servo_dest', PoseStamped, latch=True)
        self.left_out = rospy.Publisher('/left_points', PointCloud)
        self.right_out = rospy.Publisher('/right_points', PointCloud)
        self.front_out = rospy.Publisher('/front_points', PointCloud)
        #Initialize variables, so as not to spew errors before seeing a goal
        self.at_goal = False
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
        self.servoing_as.start()

    def goal_cb(self, goal):
        self.update_goal(goal.goal)
        update_rate = rospy.Rate(40)
        command = Twist()
        while not (rospy.is_shutdown() or self.at_goal):
            command.linear.x = self.get_trans_x()
            command.linear.y = self.get_trans_y()
            command.angular.z = self.get_rot()
            (x,y,z) = self.avoid_obstacles()
            if x is not None:
                command.linear.x = x
            if y is not None:
                command.linear.y = y
            command.angular.z += z
            if command.linear.y > 0:
                if not self.left_clear():
                    command.linear.y = 0.0
            elif command.linear.y < 0:
                if not self.right_clear():
                    command.linear.y = 0.0
            #print "Sending vel_command: \r\n %s" %self.command
            self.vel_out.publish(command)
            rospy.sleep(0.01) #Min sleep
            update_rate.sleep() #keep pace
        if self.at_goal:
            print "Arrived at goal"
            result = ServoResult()
            result.arrived = Bool(True)
            self.servoing_as.set_succeeded(result)
    
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
        #(current angle in odom, plus the robot-relative change to face goal)
        self.ang_goal = self.curr_ang[2] + ang_to_goal
        print "New Goal: \r\n %s" %self.bfp_goal

    def update_position(self, msg):
        if not self.servoing_as.is_active():
            return
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
                              msg.pose.pose.position.y)**2)**(1./2)

        if ((self.dist_to_goal < self.dist_thresh) and 
            (abs(self.ang_diff) < self.ang_thresh)):
            self.at_goal = True
        else:
            self.at_goal = False

    def base_laser_cb(self, msg):
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

    def get_rot(self):
        if abs(self.ang_diff) < self.ang_thresh:
            self.ang_thresh = self.ang_thresh_big
            return 0.0
        else: 
            self.ang_thresh = self.ang_thresh_small
            if self.rot_safe:
                return np.sign(self.ang_diff)*np.clip(abs(0.35*self.ang_diff),
                                                       0.1, 0.5)
            else:
                fdbk = ServoFeedback()
                fdbk.current_action = String("Cannot Rotate, obstacles nearby")
                self.servoing_as.publish_feedback(fdbk)
                return 0.0

    def get_trans_x(self):
        if (abs(self.ang_diff) < math.pi/6 and
            self.dist_to_goal > self.dist_thresh):
            return np.clip(self.dist_to_goal*0.125, 0.05, 0.3)
        else:
            return 0.0

    def get_trans_y(self):
       # Determine left/right movement speed for strafing obstacle avoidance 
        push_from_left = push_from_right = 0.0
        if len(self.left[:][0]) > 0:
            lefts = np.extract(self.left[:][1]<0.45, self.left[:][1])
            if len(lefts) > 0:
                push_from_left = -0.45 + min(lefts)
        if len(self.right[:][0]) > 0:
            rights = np.extract(self.right[:][1]>-0.45, self.right[:][1])
            if len(rights) > 0:
                push_from_right = 0.45 + max(rights)
        slide = push_from_right + push_from_left
        #print "Slide speed (m/s): %s" %slide
        return  np.sign(slide)*np.clip(abs(slide), 0.04, 0.07)

    def avoid_obstacles(self):
        ##Determine rotation to avoid obstacles in front of robot#
        x = y = None
        z = 0.
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
                x = -np.sign(min_x)*np.clip(abs(min_x),0.05,0.1)
                y = -np.sign(min_y)*np.clip(abs(min_y),0.05,0.1) 
                z = 0.
                # This should probably be avoided...
                fdbk = ServoFeedback()
                fdbk.current_action = String("TOO CLOSE: Back up slowly...")
                self.servoing_as.publish_feedback(fdbk)
                self.retreat_thresh = 0.4
            elif min(self.front[0][:]) < 0.45: 
                self.retreat_thresh = 0.3
                fdbk = ServoFeedback()
                fdbk.current_action=String("Turning Away from obstacles in front")
                self.servoing_as.publish_feedback(fdbk)
                lfobs = self.front[0][np.logical_and(self.front[1]>0,
                                                     self.front[0]<0.45)]
                rfobs = self.front[0][np.logical_and(self.front[1]<0,
                                                     self.front[0]<0.45)]
                if len(lfobs) == 0:
                    y = 0.07
                elif len(rfobs) == 0:
                    y = -0.07
                weight = np.reciprocal(np.sum(np.reciprocal(rfobs)) -
                                       np.sum(np.reciprocal(lfobs)))
                if weight > 0:
                    z = 0.05 
                else:
                    z = -0.05
            else:
                self.retreat_thresh = 0.3
        return (x,y,z)

    def left_clear(self): # Base Laser cannot see obstacles beside the back edge of the robot, which could cause problems, especially just after passing through doorways...
        if len(self.left[0][:])>0:
            #Find points directly to the right or left of the robot (not in front or behind)
            # x<0.1 (not in front), x>-0.8 (not behind)
            left_obs = self.left[:, self.left[1,:]<0.4] #points too close.
            if len(left_obs[0][:])>0:
                left_obs = left_obs[:, np.logical_and(left_obs[0,:]<0.1,
                                                      left_obs[0,:]>-0.8)]
                if len(left_obs[:][0])>0:
                    fdbk = ServoFeedback()
                    fdbk.current_action = String("Obstacle to the left, cannot move.")
                    self.servoing_as.publish_feedback(fdbk)
                    return False
        return True

    def right_clear(self):
        if len(self.right[0][:])>0:
            #Find points directly to the right or left of the robot (not in front or behind)
            # x<0.1 (not in front), x>-0.8 (not behind)
           right_obs = self.right[:, self.right[1,:]>-0.4] #points too close.
           if len(right_obs[0][:])>0:
                right_obs = right_obs[:, np.logical_and(right_obs[0,:]<0.1,
                                                        right_obs[0,:]>-0.8)]
                if len(right_obs[:][0])>0:
                    fdbk = ServoFeedback()
                    fdbk.current_action = String("Obstacle immediately to the right, cannot move.")
                    self.servoing_as.publish_feedback(fdbk)
                    return False
        return True

if __name__ == '__main__':
    servoer = ServoingServer()
    while not rospy.is_shutdown():
        rospy.spin()
    servoer.servoing_as.set_aborted()
