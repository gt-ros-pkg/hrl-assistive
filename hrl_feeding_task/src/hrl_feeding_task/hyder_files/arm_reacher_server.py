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
import hrl_lib.quaternion as quatMath #Used for quaternion math :)
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

	#self.tfListener = tf.TransformListener()
	try:
		raw_input("Register head position! Then press Enter")
		self.tf_lstnr.waitForTransform('/torso_lift_link', 'head_frame', rospy.Time.now(), rospy.Duration(10))
		(self.headPos, self.headQuat) = self.tf_lstnr.lookupTransform('/torso_lift_link', 'head_frame', rospy.Time(0))
	except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
		print "Oops, can't get head_frame tf info!, trying again :)"
		(self.headPos, self.headQuat) = self.tf_lstnr.lookupTransform('/torso_lift_link', 'head_frame', rospy.Time(0))

        #Stored initialization joint angles
        self.initialJointAnglesFrontOfBody = [0, 0.786, 0, -2, -3.141, 0, 0]

        if arm == 'r':
		self.initialJointAnglesSideOfBody = [-1.570, 0, 0, -1.570, 3.141, 0, -1.570]
		self.initialJointAnglesSideFacingFoward = [-1.570, 0, 0, -1.570, 1.570, -1.570, -1.570]
		self.timeout = 2

	else:
		self.initialJointAnglesSideOfBody = [1.570, 0, 0, -1.570, 3.141, 0, -4.712]
		self.initialJointAnglesSideFacingFoward = [1.570, 0, 0, -1.570, 1.570, -1.570, -4.712]
		self.timeout = 2

        #Variables...! #
        armReachAction.iteration = -1 # armReachAction.iteration

        self.bowlPosOffsets = np.array([[-.01,	0,	.4],
                                        [-.01,	0,    .008],
					[.05,	0,    .008],
                                        [.05,   0,    .008],
                                        [.02,   0,	.6]])

        self.bowlEulers = np.array([	[90, -60, 0], #Euler angles, XYZ rotations
					[90, -60, 0],
					[90, -60, 0],
					[90, -30, 0],
					[90,  0,  0]])

	self.bowlQuatOffsets = np.zeros((5, 4))
	self.timeouts = [15, 7, 4, 4, 4]

	for i in xrange(0, 5): #convert all the Euler angles into quaternions!
		self.eulerRads = np.radians([self.bowlEulers[i][0], self.bowlEulers[i][2], self.bowlEulers[i][1]])
		self.quatConv = quatMath.euler2quat(self.eulerRads[2], self.eulerRads[1], self.eulerRads[0])
		self.bowlQuatOffsets[i][0], self.bowlQuatOffsets[i][1], self.bowlQuatOffsets[i][2], self.bowlQuatOffsets[i][3] = self.quatConv[0], self.quatConv[1], self.quatConv[2], self.quatConv[3]

	print "Calculated quaternions:"
	print self.bowlQuatOffsets

        #Choose which set of offsets to use

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



    def bowlPoseKinectCallback(self, data):
        #Takes in a PointStamped() type message, contains Header() and Pose(), from Kinect bowl location publisher
        self.bowl_frame = data.header.frame_id
        self.bowl_pos = np.matrix([ [data.pose.position.x - .08], [data.pose.position.y - .04], [data.pose.position.z] ])
        #self.bowl_quat = np.matrix([ [data.pose.orientation.x], [data.pose.orientation.y], [data.pose.orientation.z], [data.pose.orientation.w] ])
        #^Proper code!
        self.bowl_quat = np.matrix([0,0,0,0]) #JUST FOR TESTING, in order to manually set all quaternions!
        print '-----------------------------------------------------'
        print 'Bowl Pos: '
        print self.bowl_pos
        print 'Bowl Quaternions: '
        print self.bowl_quat
        print '-----------------------------------------------------'

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
        armReachAction.iteration = 0

        #self.getJointPlan()
        calibrateJoints = raw_input("Enter 'front' or 'side' to calibrate joint angles to front or side of robot: ")
        if calibrateJoints == 'front':
            print "Setting initial joint angles... "
            self.setPostureGoal(self.initialJointAnglesFrontOfBody, 7)

        elif calibrateJoints == 'side':
            print "Setting initial joint angles..."
            self.setPostureGoal(self.initialJointAnglesSideOfBody, 7)

            #!!---- BASIC SCOOPING MOTION WITH BOWL POSITION OFFSET
    	#Flat Gripper Orientation Values:
    	#(0.642, 0.150, 0.154, 0.736)


        #---------------------------------------------------------------------------------------#

    	kinectPose = raw_input('Press k in order to position spoon to Kinect-provided bowl position, used for testing: ')
    	if kinectPose == 'k':
    		#THIS IS SOME TEST CODE, BASICALLY PUTS THE SPOON WHERE THE KINECT THINKS THE BOWL IS, USED TO COMPARE ACTUAL BOWL POSITION WITH KINECT-PROVIDED BOWL POSITION!! UNCOMMENT ALL THIS OUT IF NOT USED MUCH!!#
    		print "MOVES_KINECT_BOWL_POSITION"
            	(pos.x, pos.y, pos.z) = (self.bowl_pos[0], self.bowl_pos[1], self.bowl_pos[2])
            	(quat.x, quat.y, quat.z, quat.w) = (0.632, 0.395, -0.205, 0.635)
                self.kinectTimeout = 7
            	#self.setPositionGoal(pos, quat, self.timeout)
            	self.setOrientGoal(pos, quat, self.kinectTimeout)
    		raw_input('Press Enter to continue: ' )

	calibrateJoints = raw_input("Enter 'front' or 'side' to calibrate joint angles to front or side of robot: ")
        if calibrateJoints == 'front':
            print "Setting initial joint angles... "
            self.setPostureGoal(self.initialJointAnglesFrontOfBody, 7)

        elif calibrateJoints == 'side':
            print "Setting initial joint angles..."
            self.setPostureGoal(self.initialJointAnglesSideOfBody, 7)

        print "MOVES1 - Moving over bowl... "
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + self.bowlPosOffsets[0][0], self.bowl_pos[1] + self.bowlPosOffsets[0][1], self.bowl_pos[2] + self.bowlPosOffsets[0][2])
        (quat.x, quat.y, quat.z, quat.w) = (self.bowlQuatOffsets[0][0], self.bowlQuatOffsets[0][1], self.bowlQuatOffsets[0][2], self.bowlQuatOffsets[0][3])
        #self.setPositionGoal(pos, quat, self.timeout)
        self.setOrientGoal(pos, quat, self.timeouts[0])

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
        #self.setPositionGoal(pos, quat, self.timeout)
        self.setOrientGoal(pos, quat, self.timeouts[1])

        armReachAction.iteration += 1

        raw_input("Iteration # = %d. Enter anything to continue: " % armReachAction.iteration)

        #---------------------------------------------------------------------------------------#

        print "MOVES3 - Scooping/pushing down into bottom of bowl..."
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + self.bowlPosOffsets[2][0], self.bowl_pos[1] + self.bowlPosOffsets[2][1], self.bowl_pos[2] + self.bowlPosOffsets[2][2])
        (quat.x, quat.y, quat.z, quat.w) = (self.bowlQuatOffsets[2][0], self.bowlQuatOffsets[2][1], self.bowlQuatOffsets[2][2], self.bowlQuatOffsets[2][3])
        #self.setPositionGoal(pos, quat, self.timeout)
        self.setOrientGoal(pos, quat, self.timeouts[2])

        armReachAction.iteration += 1

        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

        #---------------------------------------------------------------------------------------#

        print "MOVES4 - Lifting a little out of bottom of bowl..."
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + self.bowlPosOffsets[3][0], self.bowl_pos[1] +  self.bowlPosOffsets[3][1], self.bowl_pos[2] + self.bowlPosOffsets[3][2])
        (quat.x, quat.y, quat.z, quat.w) = (self.bowlQuatOffsets[3][0], self.bowlQuatOffsets[3][1], self.bowlQuatOffsets[3][2], self.bowlQuatOffsets[3][3])
        #self.setPositionGoal(pos, quat, self.timeout)
        self.setOrientGoal(pos, quat, self.timeouts[3])

        armReachAction.iteration += 1

        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

        #---------------------------------------------------------------------------------------#

        print "MOVES5 - Lifting above bowl..."
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + self.bowlPosOffsets[4][0], self.bowl_pos[1] + self.bowlPosOffsets[4][1], self.bowl_pos[2] + self.bowlPosOffsets[4][2])
        (quat.x, quat.y, quat.z, quat.w) = (self.bowlQuatOffsets[4][0], self.bowlQuatOffsets[4][1], self.bowlQuatOffsets[4][2], self.bowlQuatOffsets[4][3])
        #self.setPositionGoal(pos, quat, self.timeout)
        self.setOrientGoal(pos, quat, self.timeouts[4])

        armReachAction.iteration += 1

        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

	print "MOVES6 - Reaching to mouth..."
	try:
                (self.headPos, self.headQuat) = self.tfListener.lookupTransform('/torso_lift_link', 'head_frame', rospy.Time(0))
		(pos.x, pos.y, pos.z) = (self.headPos[0] + self.headPosOffsets[0][0], self.headPos[1] + self.headPosOffsets[0][1], self.headPos[2] + self.headPosOffsets[0][2]);
        	(quat.x, quat.y, quat.z, quat.w) = (self.headQuat[0] + self.headQuatOffsets[0][0], self.headQuat[1] + self.headQuatOffsets[0][1], self.headQuat[2] + self.headQuatOffsets[0][2], self.headQuat[3] + self.headQuatOffsets[0][3])
        	self.setOrientGoal(pos, quat, self.headTimeouts[0])

        	armReachAction.iteration += 1

        	raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)
        except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print "Oops, can't get head_frame tf info!"

	#print "MOVES7 - Moving away from face..."
	#self.stopCallback('!')
	#armReachAction.iteration += 1
	#raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

        return True

    def stopCallback(self, msg):

	print "Stopping Motion..."
       	self.setStop() #Stops Current Motion
        posStop = Point()
        quatStop = Quaternion()
        #Sets goal positions and quaternions to match previously reached end effector position, go to last step
        #(posStop.x, posStop.y, posStop.z) = (self.bowl_pos[0] + self.bowlPosOffsets[armReachAction.iteration][0], self.bowl_pos[1] + self.bowlPosOffsets[armReachAction.iteration][1], self.bowl_pos[2] + self.bowlPosOffsets[armReachAction.iteration][2])

        #(quatStop.x, quatStop.y, quatStop.z, quatStop.w) = (self.bowlQuatOffsets[armReachAction.iteration][0], self.bowlQuatOffsets[armReachAction.iteration][1], self.bowlQuatOffsets[armReachAction.iteration][2], self.bowlQuatOffsets[armReachAction.iteration][3])

        print "Moving to previous position..."
        #self.setOrientGoal(posStop, quatStop, self.timeout) #go to previously reached position, last step

        #Safe reversed position 1
        (posStop.x, posStop.y, posStop.z) = (0.967, 0.124, 0.525)
        (quatStop.x, quatStop.y, quatStop.z, quatStop.w) = (-0.748, -0.023, -0.128, 0.651)
        print "Moving to safe position 1"
        self.setOrientGoal(posStop, quatStop, 5)

        print "Moving to safe position 2"
        (posStop.x, posStop.y, posStop.z) = (0.420, 0.814, 0.682)
        (quatStop.x, quatStop.y, quatStop.z, quatStop.w) = (-0.515, -0.524, 0.144, 0.663)
        self.setOrientGoal(posStop, quatStop, 5)

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
