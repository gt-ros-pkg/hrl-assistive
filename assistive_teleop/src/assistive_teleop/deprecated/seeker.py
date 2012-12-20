#!/usr/bin/python

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
import math
import decimal
import numpy as np
import tf
from tf import transformations as tft
from geometry_msgs.msg import PoseWithCovarianceStamped, PointStamped, Twist, Point32, PoseStamped
from sensor_msgs.msg import LaserScan, PointCloud
from std_msgs.msg import String
import costmap_services.python_client as costmap

class Follower:

    def __init__(self):
        rospy.init_node('relative_servoing')
        rospy.Subscriber('robot_pose_ekf/odom_combined', PoseWithCovarianceStamped, self.update_position) 
        rospy.Subscriber('/base_scan', LaserScan, self.update_base_laser)
        rospy.Subscriber('/goal', PoseStamped, self.update_goal)
        self.vel_out = rospy.Publisher('base_controller/command', Twist)
        self.rate_test = rospy.Publisher('rate_test', String)
        #self.cs = costmap.CostmapServices( accum = 3 )
        self.tfl = tf.TransformListener()
        #self.driver = rospy.Publisher('/base_controller/command', Twist)
        self.left_out = rospy.Publisher('/left_points', PointCloud)
        self.right_out = rospy.Publisher('/right_points', PointCloud)
        self.front_out = rospy.Publisher('/front_points', PointCloud)
        #Initialize variables, so as not to spew errors before seeing a goal
        self.command = Twist()
        self.goal_received = False
        self.goal_present = False
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
        self.dist_thresh = 0.4
        self.left = [[],[]]
        self.right = [[],[]]
        self.front = [[],[]]

    def update_goal(self, msg):
        self.goal_received = True
        msg.header.stamp = rospy.Time.now()
                
        #if not msg.header.frame_id == '/base_footprint' or msg.header.frame_id == 'base_footprint'
        #    self.tfl.waitForTransform(msg.header.frame_id, '/base_footprint', msg.header.stamp, rospy.Duration(0.5))
        #    msg = self.tfl.transformPose('/base_footprint', msg)
        #self.ang_to_goal = 

        self.tfl.waitForTransform(msg.header.frame_id, '/base_footprint', msg.header.stamp, rospy.Duration(0.5))
        self.bfp_goal = self.tfl.transformPose('/base_footprint', msg)
        self.tfl.waitForTransform(msg.header.frame_id, 'odom_combined', msg.header.stamp, rospy.Duration(0.5))
        self.odom_goal = self.tfl.transformPose('/odom_combined', msg)
        
        ang_to_goal = math.atan2(self.bfp_goal.pose.position.y, self.bfp_goal.pose.position.x)
        self.ang_goal = self.curr_ang[2] + ang_to_goal #The desired angular position (the current angle in odom, plus the robot-relative change to face goal)
        #print "New Goal: \r\n %s" %self.bfp_goal

    def update_position(self, msg):
        #self.curr_pose = msg.pose.pose
        self.curr_ang = tft.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.ang_diff = np.unwrap([0, self.ang_goal - self.curr_ang[2]])[1] # Normalized via unwrap relative to 0; (keeps between -pi/pi)
        #print "Ang Diff: %s" %self.ang_diff

        self.dist_to_goal = math.sqrt((self.odom_goal.pose.position.x-msg.pose.pose.position.x)**2 + (self.odom_goal.pose.position.y-msg.pose.pose.position.y)**2)
        #print "Dist to goal: %s" %self.dist_to_goal

        if self.goal_received:
            if self.dist_to_goal < self.dist_thresh and abs(self.ang_diff) < self.ang_thresh:
                self.goal_present = False
                #self.goal_received = False
                #print "Achieved Goal!"
            else:
                self.goal_present = True

    def update_base_laser(self, msg):
        max_angle = msg.angle_max
        ranges = np.array(msg.ranges)
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        near_angles = np.extract(np.logical_and(ranges<1, ranges>0.003), angles)#Filter out noise (<0.003), distant points(>1m), leaves nearby, relevant obstacles
        near_ranges = np.extract(np.logical_and(ranges<1, ranges>0.003), ranges)
        #print "Min in Ranges: %s" %min(ranges)
       
        #if len(near_ranges) > 0:
        xs = near_ranges * np.cos(near_angles)
        ys = near_ranges * np.sin(near_angles)
        #print "xs: %s" %xs
        self.points = np.vstack((xs,ys))
        #print "Points: %s" %points
        self.bfp_points = np.vstack((np.add(0.275, xs),ys))
        #print "bfp Points: %s" %bfp_points
        self.bfp_dists = np.sqrt(np.add(np.square(self.bfp_points[0][:]),np.square(self.bfp_points[1][:])))
        #print min(self.bfp_dists)
        if len(self.bfp_dists) >0:
            if min(self.bfp_dists) > 0.5:
                self.rot_safe = True
            else:
                self.rot_safe = False
        else:
            self.rot_safe = True
        
        self.left = np.vstack((xs[np.nonzero(ys>0.35)[0]], ys[np.nonzero(ys>0.35)[0]]))
        self.right = np.vstack((xs[np.nonzero(ys<-0.35)[0]], ys[np.nonzero(ys<-0.35)[0]]))
        self.front = np.vstack((np.extract(np.logical_and(ys<0.35,ys>-0.35),xs), np.extract(np.logical_and(ys<0.35, ys>-0.35),ys)))

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
            frontScan.header.frame_id = '/base_laser_link'
            frontScan.header.stamp = rospy.Time.now()
            
            for i in range(len(self.front[:][0])):
                pt = Point32()
                pt.x = self.front[0][i]
                pt.y = self.front[1][i]
                pt.z = 0
                frontScan.points.append(pt)
            
            self.front_out.publish(frontScan)

    def set_rot(self):
        if abs(self.ang_diff) < self.ang_thresh: #Fully oriented, relax constraint to avoid osscilation
            self.ang_thresh = self.ang_thresh_big
            self.command.angular.z = 0.0

        else: # Not fully oriented, continue until pointed within small constraint
            if self.rot_safe: # Check for obstacles in the narrow rotation radius
                self.command.angular.z = np.sign(self.ang_diff)*np.clip(abs(0.35*self.ang_diff), 0.1, 0.5)
                self.ang_thresh = self.ang_thresh_small
            else:
                print "Cannot Rotate, obstacles nearby"

    def set_trans(self):
        if abs(self.ang_diff) < math.pi/20 and self.dist_to_goal > self.dist_thresh: #Facing the right direction and not there yet, so start moving.
            self.command.linear.x = np.clip(self.dist_to_goal*0.125, 0.15, 0.3)
        else:
            self.command.linear.x = 0.0

        ##Determine obstacle avoidance rotation to avoid obstacles in front of robot#
    def avoid_obstacles(self):
        if len(self.front[0][:]) > 0:
            if min(self.front[0][:]) < self.retreat_thresh: #0.225:  #0.5 (round-up on corner-to-corner radius of robot) - 0.275 (x diff from base laser link to base footprint)
                self.command.linear.x = -0.05
                self.command.angular.z = 0.0
                print "TOO CLOSE: Back up slowly..." # This should probably be avoided...
                self.retreat_thresh = 0.4
            elif min(self.front[0][:]) < 0.6: 
                self.retreat_thresh = 0.3
                print "Turning Away from obstacles in front"
                self.command.linear.x = 0.0
                lfobs = self.front[0][np.nonzero(self.front[1]>0)]
                #print "lfobs: %s" %lfobs
                rfobs = self.front[0][np.nonzero(self.front[1]<0)]
                #print "rfobs: %s" %rfobs
                weight = np.reciprocal(np.sum(np.reciprocal(rfobs)) - np.sum(np.reciprocal(lfobs)))
                if weight > 0:
                    self.command.angular.z = 0.1 
                else:
                    self.command.angular.z = -0.1
            else:
                self.retreat_thresh = 0.3

    def left_clear(self): # Base Laser cannot see obstacles beside the back edge of the robot, which could cause problems, especially just after passing through doorways...
        if len(self.left[0][:])>0:
            #Find points directly to the right or left of the robot (not in front or behind)
            # x<0.1 (not in front), x>-0.8 (not behind)
            left_obs = self.left[:, self.left[1,:]<0.4] #points too close.
            if len(left_obs[0][:])>0:
                left_obs = left_obs[:, np.logical_and(left_obs[0,:]<0.15, left_obs[0,:]>-0.25)]
                if len(left_obs[:][0])>0:
                    print "Obstacle immediately to the left, don't turn that direction"
                    if self.command.angular.z > 0:
                        self.command.angular.z = 0

    def right_clear(self):
        if len(self.right[0][:])>0:
            #Find points directly to the right or left of the robot (not in front or behind)
            # x<0.1 (not in front), x>-0.8 (not behind)
            right_obs = self.right[:, self.right[1,:]<0.4] #points too close.
            if len(right_obs[0][:])>0:
                right_obs = right_obs[:, np.logical_and(right_obs[0,:]<0.15, right_obs[0,:]>-0.25)]
                if len(right_obs[:][0])>0:
                    print "Obstacle immediately to the right, don't turn that direction"
                    if self.command.angular.z < 0:
                        self.command.angular.z = 0

    def check_costmap(self):
        if self.cs.scoreTraj_PosHyst( self.command.linear.x, self.command.linear.y, self.command.linear.z ) != -1.0:
            return True
        else:
            return False

    def run(self):
        if self.goal_present:
            self.set_trans()
            print "linear %s" %self.command.linear.x
            if self.command.linear.x == 0: 
                self.set_rot()
                self.vel_out.publish(self.command)
                return 
            else:
                self.set_rot()
            if self.command.angular.z == 0 and self.command.linear.x == 0:
                print "Can't rot, so move foreward"
                self.commad.linear.x = 0.1
            self.avoid_obstacles()
            
            self.left_clear()
            self.right_clear()
          #  if check_costmap():
            print "Sending vel_command: \r\n %s" %self.command
            self.vel_out.publish(self.command)
          #  else:
            #    print "COSTMAP SENSES COLLISION, NOT MOVING"
        else:
            pass

if __name__ == '__main__':
    Follower = Follower()
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        Follower.run()
        r.sleep()
