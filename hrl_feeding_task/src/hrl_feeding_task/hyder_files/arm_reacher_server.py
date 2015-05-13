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
from std_msgs.msg import String
from pr2_controllers_msgs.msg import JointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import hrl_haptic_manipulation_in_clutter_msgs.msg as haptic_msgs

class armReachAction(mpcBaseAction):
    def __init__(self, d_robot, controller, arm): #removed arm= 'l' so I can use right arm as well as an option

        mpcBaseAction.__init__(self, d_robot, controller, arm)

        #Subscribers to publishers of bowl location data
        rospy.Subscriber('hrl_feeding_task/manual_bowl_location', PoseStamped, self.bowlPoseCallback)  # hrl_feeding/bowl_location
        rospy.Subscriber('hrl_feeding_task/RYDS_CupLocation', PoseStamped, self.bowlPoseKinectCallback) # launch you can remap the topic name (ros wiki)

        rospy.Subscriber('hrl_feeding_task/emergency_arm_stop', String, self.stopCallback)

        # service request
        self.reach_service = rospy.Service('/arm_reach_enable', None_Bool, self.start_cb)

        #VARIABLES! # better variable name
        self.initialJointAnglesFrontOfBody = [0, 0.786, 0, -2, -3.141, 0, 0]

        self.initialJointAnglesSideOfBody = [1.570, 0, 0, -1.570, 3.141, 0, -1.570]

        self.initialJointAnglesSideFacingFoward = [1.570, 0, 0, -1.570, 1.570, -1.570, -1.481043325223495]

        #!self.previousGoals = JointTrajectory()
        #NEED TO APPEND A JointTrajectoryPoint to the end!!! FIX THIS!!! NOT FIXED AS OF FRIDAY MARCH 27, 2015
        #!self.point = JointTrajectoryPoint() #ATTEMPT AT FIRST FIX!

        #Variables...! #
        armReachAction.iteration = -1 # armReachAction.iteration

        # offset variable
        self.bowlPosOffsets = np.array([[-0.09743569,    -0.11179373,    0.18600000], #Row 1 = Move 1
                                   [-0.03143569,    -0.09879372,    0.02800000], #Row 2 = Move 2
                                   [-0.03543569,    -0.09179373,    -0.02600000], # ...
                                   [0.00156431,     -0.07279373,    0.04900000],
                                   [0.00756431,     -0.13179373,    0.59400000]])

        self.bowlQuatOffsets = np.array([[0.580, 0.333, 0.050, 0.742],
                                   [0.484, 0.487, -0.164, 0.708],
                                   [0.505, 0.516, -0.160, 0.673],
                                   [0.617, 0.300, -0.035, 0.726],
                                   [0.702, 0.168, 0.132, 0.679]])

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

	print 'Bowl Pos: '
	print self.bowl_pos
	print 'Bowl Quaternions: '
	print self.bowl_quat

    def run(self):

        pos  = Point()
        quat = Quaternion()

        # duplication
        confirm = False
        while not confirm:
            print "Current pose"
            print self.getEndeffectorPose()
            ans = raw_input("Enter y to confirm to start: ")
            if ans == 'y':
                confirm = True
            print self.getJointAngles()

        #Variables...! # local
        armReachAction.iteration = -1

        #self.getJointPlan()
        calibrateJoints = raw_input("Enter 'front' or 'side' to calibrate joint angles to front or side of robot: ")
        if calibrateJoints == 'front':
            print "Setting initial joint angles... "
            self.setPostureGoal(self.jointAnglesFront, 5)

        elif calibrateJoints == 'side':
            print "Setting initial joint angles..."
            self.setPostureGoal(self.jointAnglesSideForward, 5)

            #!!---- BASIC SCOOPING MOTION WITH BOWL POSITION OFFSET
    	#Flat Gripper Orientation Values:
    	#(0.642, 0.150, 0.154, 0.736)

        raw_input("Iteration # %d. Enter anything to start: " % armReachAction.iteration)

        #---------------------------------------------------------------------------------------#

    	kinectPose = raw_input('Press k in order to position spoon to Kinect-provided bowl position, used for testing: ')
    	if kinectPose == 'k':
    		#THIS IS SOME TEST CODE, BASICALLY PUTS THE SPOON WHERE THE KINECT THINKS THE BOWL IS, USED TO COMPARE ACTUAL BOWL POSITION WITH KINECT-PROVIDED BOWL POSITION!! UNCOMMENT ALL THIS OUT IF NOT USED MUCH!!#
    		print "MOVES_KINECT_BOWL_POSITION"
            	(pos.x, pos.y, pos.z) = (self.bowl_pos[0], self.bowl_pos[1], self.bowl_pos[2])
            	(quat.x, quat.y, quat.z, quat.w) = (0, 0, 0, 1)
            	timeout = 7
            	#self.setPositionGoal(pos, quat, timeout)
            	self.setOrientGoal(pos, quat, timeout)
    		raw_input('Press Enter to continue: ' )

        print "MOVES1 - Moving over bowl... "
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + self.bowlPosOffsets[0][0], self.bowl_pos[1] + self.bowlPosOffsets[0][1], self.bowl_pos[2] + self.bowlPosOffsets[0][2])
        (quat.x, quat.y, quat.z, quat.w) = (self.bowlQuatOffsets[0][0], self.bowlQuatOffsets[0][1], self.bowlQuatOffsets[0][2], self.bowlQuatOffsets[0][3])
        timeout = 4
        #self.setPositionGoal(pos, quat, timeout)
        self.setOrientGoal(pos, quat, timeout)

        # #Code for storing current joint angles in case of playback...
        # self.currentAngles = self.getJointAngles()
        # print "Current Angles:"
        # print self.currentAngles
        # self.point.positions = self.currentAngles
        # self.previousGoals.points.append(self.point)
        # print "EVERYTHING:"
        # print self.previousGoals
        # print "resized Points:"
        # print self.previousGoals.points[armReachAction.iteration]
        armReachAction.iteration += 1
        # print "Stored joint angles: "
        # print self.previousGoals.points[armReachAction.iteration].positions[2]


        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)


        #---------------------------------------------------------------------------------------#

        print "MOVES2 - Pointing down into bottom of bowl..."
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + self.bowlPosOffsets[1][0], self.bowl_pos[1] + self.bowlPosOffsets[1][1], self.bowl_pos[2] + self.bowlPosOffsets[1][2])
        (quat.x, quat.y, quat.z, quat.w) = (self.bowlQuatOffsets[1][0], self.bowlQuatOffsets[1][1], self.bowlQuatOffsets[1][2], self.bowlQuatOffsets[1][3])
        timeout = 4
        #self.setPositionGoal(pos, quat, timeout)
        self.setOrientGoal(pos, quat, timeout)

        # #Code for storing current joint angles in case of playback...
        # currentAngles = self.getJointAngles()
        # np.resize(self.previousGoals.points, armReachAction.iteration+1)
        # np.resize(self.previousGoals.points[armReachAction.iteration].positions, 7)
        # self.previousGoals.points[armReachAction.iteration].positions = currentAngles
        armReachAction.iteration += 1
        #
        # print "Stored joint angles: "
        # print self.previousGoals.points[iteration].positions
        raw_input("Iteration # = %d. Enter anything to continue: " % armReachAction.iteration)

        #---------------------------------------------------------------------------------------#

        print "MOVES3 - Scooping/pushing down into bottom of bowl..."
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + self.bowlPosOffsets[2][0], self.bowl_pos[1] + self.bowlPosOffsets[2][1], self.bowl_pos[2] + self.bowlPosOffsets[2][2])
        (quat.x, quat.y, quat.z, quat.w) = (self.bowlQuatOffsets[2][0], self.bowlQuatOffsets[2][1], self.bowlQuatOffsets[2][2], bowlQuatOffsets[2][3])
        timeout = 2
        #self.setPositionGoal(pos, quat, timeout)
        self.setOrientGoal(pos, quat, timeout)

        # #Code for storing current joint angles in case of playback...
        # currentAngles = self.getJointAngles()
        # np.resize(self.previousGoals.points, armReachAction.iteration+1)
        # np.resize(self.previousGoals.points[armReachAction.iteration].positions, 7)
        # self.previousGoals.points[armReachAction.iteration].positions = currentAngles
        armReachAction.iteration += 1
        #
        # print "Stored joint angles: "
        # print self.previousGoals.points[iteration].positions
        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

        #---------------------------------------------------------------------------------------#

        print "MOVES4 - Lifting a little out of bottom of bowl..."
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + self.bowlPosOffsets[3][0], self.bowl_pos[1] +  self.bowlPosOffsets[3][1], self.bowl_pos[2] + self.bowlPosOffsets[3][2])
        (quat.x, quat.y, quat.z, quat.w) = (self.bowlQuatOffsets[3][0], self.bowlQuatOffsets[3][1], self.bowlQuatOffsets[3][2], self.bowlQuatOffsets[3][3])
        timeout = 2
        #self.setPositionGoal(pos, quat, timeout)
        self.setOrientGoal(pos, quat, timeout)

        # #Code for storing current joint angles in case of playback...
        # currentAngles = self.getJointAngles()
        # np.resize(self.previousGoals.points, armReachAction.iteration+1)
        # np.resize(self.previousGoals.points[armReachAction.iteration].positions, 7)
        # self.previousGoals.points[armReachAction.iteration].positions = currentAngles
        armReachAction.iteration += 1
        #
        # print "Stored joint angles: "
        # print self.previousGoals.points[iteration].positions
        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

        #---------------------------------------------------------------------------------------#

        print "MOVES5 - Lifting above bowl..."
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + self.bowlPosOffsets[4][0], self.bowl_pos[1] + self.bowlPosOffsets[4][1], self.bowl_pos[2] + self.bowlPosOffsets[4][2])
        (quat.x, quat.y, quat.z, quat.w) = (self.bowlQuatOffsets[4][0], self.bowlQuatOffsets[4][1], self.bowlQuatOffsets[4][2], self.bowlQuatOffsets[4][3])
        timeout = 2
        #self.setPositionGoal(pos, quat, timeout)
        self.setOrientGoal(pos, quat, timeout)

        # #Code for storing current joint angles in case of playback...
        # currentAngles = self.getJointAngles()
        # np.resize(self.previousGoals.points, armReachAction.iteration+1)
        # np.resize(self.previousGoals.points[armReachAction.iteration].positions, 7)
        # self.previousGoals.points[armReachAction.iteration].positions = currentAngles
        armReachAction.iteration += 1
        #
        # print "Stored joint angles: "
        # print self.previousGoals.points[iteration].positions
        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

        return True

    def stopCallback(self, msg):
        print "Stopping Motion..."
        self.setStop() #Stops Current Motion
        posStop = Point()
        quatStop = Quaternion()
        #Sets goal positions and quaternions to match previously reached end effector position, go to last step
        (posStop.x, posStop.y, posStop.z) = (self.bowl_pos[0] + self.bowlPosOffsets[armReachAction.iteration][0], self.bowl_pos[1] + self.bowlPosOffsets[armReachAction.iteration][1], self.bowl_pos[2] + self.bowlPosOffsets[armReachAction.iteration][2])

        (quatStop.x, quatStop.y, quatStop.z, quatStop.w) = (self.bowlQuatOffsets[armReachAction.iteration][0], self.bowlQuatOffsets[armReachAction.iteration][1], self.bowlQuatOffsets[armReachAction.iteration][2], self.bowlQuatOffsets[armReachAction.iteration][3])

        timeout = 4
        print "Moving to previous position..."
        self.setOrientGoal(posStop, quatStop, timeout) #go to previously reached position, last step

        #Safe reversed position 1
        (posStop.x, posStop.y, posStop.z) = (0.967, 0.124, 0.525)
        (quatStop.x, quatStop.y, quatStop.z, quatStop.w) = (-0.748, -0.023, -0.128, 0.651)
        print "Moving to safe position 1"
        self.setOrientGoal(posStop, quatStop, timeout)

        print "Moving to safe position 2"
        (posStop.x, posStop.y, posStop.z) = (0.420, 0.814, 0.682)
        (quatStop.x, quatStop.y, quatStop.z, quatStop.w) = (-0.515, -0.524, 0.144, 0.663)
        self.setOrientGoal(posStop, quatStop, timeout)



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
