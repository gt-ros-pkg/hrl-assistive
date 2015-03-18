#!/usr/bin/env python  

import roslib; roslib.load_manifest('sandbox_dpark_darpa_m3')
roslib.load_manifest('hrl_feeding_task')
import rospy
import numpy as np, math
import time
import tf

import hrl_haptic_mpc.haptic_mpc_util as haptic_mpc_util

from hrl_srvs.srv import None_Bool, None_BoolResponse
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from sandbox_dpark_darpa_m3.lib.hrl_mpc_base import mpcBaseAction
import std_msgs.msg

import hrl_haptic_manipulation_in_clutter_msgs.msg as haptic_msgs

class armReachAction(mpcBaseAction):
    def __init__(self, d_robot, controller, arm = 'l'):

        mpcBaseAction.__init__(self, d_robot, controller, arm)
        
	#Subscribers to publishers of bowl location data
        rospy.Subscriber('bowl_location', PoseStamped, self.bowlPoseCallback)
	rospy.Subscriber('RYDS_CupLocation', PoseStamped, self.bowlPoseKinectCallback)	
	#Be able to read the current state of the robot, used for setting new joint angles for init
	rospy.Subscriber('haptic_mpc/robot_state', haptic_msgs.RobotHapticState, self.robotStateCallback)

	#Be able to publish new data to the robot state, especially new joint angles
	#self.robotState_pub = rospy.Publisher('haptic_mpc/robot_state', haptic_msgs.RobotHapticState)

        #self.tfListener = tf.TransformListener()
        #self.tfBroadcaster = tf.TransformBroadcaster()

        # service request
        self.reach_service = rospy.Service('/arm_reach_enable', None_Bool, self.start_cb)

	self.pose1JointAngles = list(0.02204231193041639, 0.72850849623194, 0.08302486916827911, -2.051374187142846, -3.1557218713638484, -1.2799710435005978, 10.952306846165152)
	#These joint angles correspond to 'Pose 1' - See Keep note

        rate = rospy.Rate(100) # 25Hz, nominally.                
        while not rospy.is_shutdown():
            if self.getJointAngles() != []:
                print "----------------------"
                print "Current joint angles"
                print self.getJointAngles()
                print "Current pose"
                print self.getEndeffectorPose()
                print "----------------------"
                break

        rospy.spin()
        
    def start_cb(self, req):

        # Run manipulation tasks
        if self.run():
            return None_BoolResponse(True)
        else:
            return None_BoolResponse(False)

    def bowlPoseCallback(self, data):
	self.bowl_frame = data.header.frame_id
	self.bowl_pos = np.matrix([ [data.pose.position.x], [data.pose.position.y], [data.pose.position.z] ])
        self.bowl_quat = np.matrix([ [data.pose.orientation.x], [data.pose.orientation.y], [data.pose.orientation.z], [data.pose.orientation.w] ])

	#rospy.loginfo(self.bowl_pos)
        #rospy.loginfo(self.bowl_quat)


    def bowlPoseKinectCallback(self, data):
	#Takes in a PointStamped() type message, contains Header() and Pose(), from Kinect bowl location publisher
	self.bowl_frame = data.header.frame_id
	self.bowl_pos = np.matrix([ [data.pose.position.x], [data.pose.position.y], [data.pose.position.z] ])
        self.bowl_quat = np.matrix([ [data.pose.orientation.x], [data.pose.orientation.y], [data.pose.orientation.z], [data.pose.orientation.w] ])


	#rospy.loginfo(self.bowl_pos)
	#rospy.loginfo(self.bowl_quat)	

    def robotStateCallback(self, data): #Read from RobotState topic and saves the current data, including joint angles, used to create new message containing initialized joint angles to publish to RobotState topic
	self.robotData = data
	self.joint_angles_raw = data.joint_angles
	#rospy.loginfo("Raw Robot State Data: ")
	#rospy.loginfo(self.robotData)
	#rospy.loginfo("Raw Robot Joint Angles: ")
	#rospy.loginfo(self.joint_angles_raw)
	#rospy.sleep(1)
	
    def run(self):

        pos  = Point()
        quat = Quaternion()

        confirm = False
        while not confirm:
            print "Current pose"
            print self.getEndeffectorPose()
            ans = raw_input("Enter y to confirm to start: ")
            if ans == 'y':
                confirm = True
        print self.getJointAngles()        
        
	print "Setting initial joint angles... "
	
	#BAD DOESN'T WORK BELOW!!!
	#self.robotData.joint_angles = self.pose1JointAngles
	#self.robotState_pub.publish(self.robotData)
	#Publishes initial joint angles to robot_state topic, in theory supposed to make robot left arm go to specified joint angles, but doesn't work well. New method needed...

	jointAnglesList = self.pose1JointAngles

	self.setPostureGoal(jointAnglesList, 1)

        #!!---- BASIC SCOOPING MOTION WITH BOWL POSITION OFFSET
	#Flat Gripper Orientation Values:
	#(0.642, 0.150, 0.154, 0.736)

	#raw_input("Enter anything to go to initial position: ")
	#print "MOVE0, Initial Position"
        #(pos.x, pos.y, pos.z) = (0.803, 0.182, 0.067)
        #(quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)
        #timeout = 1.0
        #self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")


        print "MOVES1"
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + -0.2920, self.bowl_pos[1] + -0.7260, self.bowl_pos[2] + 0.2600)
        (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) # (0.573, -0.451, -0.534, 0.428)
        timeout = 1.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        print "MOVES2"
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + -0.1080, self.bowl_pos[1] + -0.1410, self.bowl_pos[2] + 0.2550)
        (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.690, -0.092, -0.112, 0.709)
        timeout = 1.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        print "MOVES3"
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + -0.0920, self.bowl_pos[1] + 0.0310, self.bowl_pos[2] + 0.0950)
        (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  #  (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.713, 0.064, -0.229, 0.659)
        timeout = 1.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        print "MOVES4"
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + -0.0140, self.bowl_pos[1] +  0, self.bowl_pos[2] + 0.00900)
        (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.700, 0.108, -0.321, 0.629)
        timeout = 1.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        print "MOVES5"
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + 0, self.bowl_pos[1] + 0, self.bowl_pos[2] +  0) #REACHED THE ACTUAL BOWL
        (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.706, 0.068, -0.235, 0.664)
        timeout = 1.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        print "MOVES6"
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + 0.0120, self.bowl_pos[1] + -0.00100, self.bowl_pos[2] +  0.0130)
        (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.672, 0.037, -0.137, 0.727)
        timeout = 1.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        print "MOVES7"
        print "MOVES7"
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + -0.122, self.bowl_pos[1] + -0.0319, self.bowl_pos[2] + 0.300)
        (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.699, -0.044, -0.085, 0.709)
        timeout = 1.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        print "MOVES8"
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + -0.0210, self.bowl_pos[1] +  -0.0519, self.bowl_pos[2] +  0.410)
        (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.677, -0.058, -0.004, 0.733)
        timeout = 1.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")
        
        return True

    
if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    haptic_mpc_util.initialiseOptParser(p)
    opt = haptic_mpc_util.getValidInput(p)

    # Initial variables
    d_robot    = 'pr2'
    controller = 'static'
    #controller = 'actionlib'
    #arm        = 'l'

    rospy.init_node('arm_reacher')    
    ara = armReachAction(d_robot, controller, opt.arm)
    rospy.spin()


    #print "MOVES1"
    # (pos.x, pos.y, pos.z) = (0.471, -0.134, -0.041)
    # (quat.x, quat.y, quat.z, quat.w) = (0.573, -0.451, -0.534, 0.428)
    # timeout = 1.0
    # self.setOrientGoal(pos, quat, timeout)
    # raw_input("Enter anything to start: ")

    # print "MOVES2"
    # (pos.x, pos.y, pos.z) = (0.655, 0.451, -0.046)
    # (quat.x, quat.y, quat.z, quat.w) = (0.690, -0.092, -0.112, 0.709)
    # timeout = 1.0
    # self.setOrientGoal(pos, quat, timeout)
    # raw_input("Enter anything to start: ")

    # print "MOVES3"
    # (pos.x, pos.y, pos.z) = (0.671, 0.623, -0.206)
    # (quat.x, quat.y, quat.z, quat.w) = (0.713, 0.064, -0.229, 0.659)
    # timeout = 1.0
    # self.setOrientGoal(pos, quat, timeout)
    # raw_input("Enter anything to start: ")

    # print "MOVES4"
    # (pos.x, pos.y, pos.z) = (0.749, 0.592, -0.292)
    # (quat.x, quat.y, quat.z, quat.w) = (0.700, 0.108, -0.321, 0.629)
    # timeout = 1.0
    # self.setOrientGoal(pos, quat, timeout)
    # raw_input("Enter anything to start: ")

    # print "MOVES5"
    # (pos.x, pos.y, pos.z) = (0.763, 0.592, -0.301)
    # (quat.x, quat.y, quat.z, quat.w) = (0.706, 0.068, -0.235, 0.664)
    # timeout = 1.0
    # self.setOrientGoal(pos, quat, timeout)
    # raw_input("Enter anything to start: ")

    # print "MOVES6"
    # (pos.x, pos.y, pos.z) = (0.775, 0.591, -0.288)
    # (quat.x, quat.y, quat.z, quat.w) = (0.672, 0.037, -0.137, 0.727)
    # timeout = 1.0
    # self.setOrientGoal(pos, quat, timeout)
    # raw_input("Enter anything to start: ")

    # print "MOVES7"
    # (pos.x, pos.y, pos.z) = (0.641, 0.560, -0.001)
    # (quat.x, quat.y, quat.z, quat.w) = (0.699, -0.044, -0.085, 0.709)
    # timeout = 1.0
    # self.setOrientGoal(pos, quat, timeout)
    # raw_input("Enter anything to start: ")

    # print "MOVES8"
    # (pos.x, pos.y, pos.z) = (0.742, 0.540, 0.109)
    # (quat.x, quat.y, quat.z, quat.w) = (0.677, -0.058, -0.004, 0.733)
    # timeout = 1.0
    # self.setOrientGoal(pos, quat, timeout)
