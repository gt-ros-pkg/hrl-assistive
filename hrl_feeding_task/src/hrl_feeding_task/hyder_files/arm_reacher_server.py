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
from hrl_feeding_task.srv import PosQuatTimeoutSrv, AnglesTimeoutSrv
#Used to communicate with right arm control server

#from arm_reacher_helper_right import rightArmControl
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
        rospy.Subscriber('hrl_feeding_task/RYDS_CupLocation', PoseStamped, self.bowlPoseKinectCallback)

        rospy.Subscriber('hrl_feeding_task/emergency_arm_stop', String, self.stopCallback)

        # service request
        self.reach_service = rospy.Service('/arm_reach_enable', None_Bool, self.start_cb)

        try:
            self.setOrientGoalRight = rospy.ServiceProxy('/setOrientGoalRightService', PosQuatTimeoutSrv)
            self.setStopRight = rospy.ServiceProxy('/setStopRightService', None_Bool)
            self.setPostureGoalRight = rospy.ServiceProxy('/setPostureGoalRightService', AnglesTimeoutSrv)
            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            print "CONNECTED TO RIGHT ARM SERVER YAY!"
        except:
            print "Oops, can't connect to right arm server!"

        #Stored initialization joint angles
        self.initialJointAnglesFrontOfBody = [0, 0.786, 0, -2, -3.141, 0, 0]

        self.initialJointAnglesSideOfBodyLEFT = [1.570, 0, 0, -1.570, 3.141, 0, -1.570]
        #self.initialJointAnglesSideOfBodyLEFT = [-1.570, 0, 0, -1.570, 3.141, 0, -1.570]
        self.initialJointAnglesSideFacingFowardLEFT = [-1.570, 0, 0, -1.570, 1.570, -1.570, -1.570]

        self.initialJointAnglesSideOfBodyRIGHT = [-1.570, 0, 0, -1.570, 3.141, 0, -4.712]
        self.initialJointAnglesSideFacingFowardRIGHT = [1.570, 0, 0, -1.570, 1.570, -1.570, -4.712]

        #Variables...! #
        armReachAction.iteration = 0

        self.bowlPosOffsets = np.array([[-.01,	0,	   .4], #offsets from bowl_pos to left arm spoon
                                        [-.01,	0,   .008], #scooping motion is set of offsets
                                        [.05,	0,   .008],
                                        [.05,   0,   .008],
                                        [.02,   0,	   .6],
                                        [0,     0,      0],
                                        [.02,	0,	  .6]])

        self.bowlEulers = np.array([[90,	-60,	0], #Euler angles, XYZ rotations for left arm spoon
                                    [90,	-60,	0], #controls scooping motion
                                    [90,	-60,	0],
                                    [90,	-30,	0],
                                    [90,	  0,    0],
                                    [0,       0,	0],
                                    [90,	  0,	0]])

        self.headPosOffsets = np.array([[.01,   .075,   -.01], #offsets from head_frame to left arm spoon
                                        [.01,   .2,       .1], #feeding motion set of offsets
                                        [0,      0,       0]])

        self.headEulers = np.array([[90,    0,  -90], #Euler angles, XYZ rotations for left arm spoon
                                    [90,    0,  -90], #controls feeding to mouth motion
                                    [90,    0,   0]])

        self.stopPos = np.array([[.7, .7, .5]])
        self.stopEulers = np.array([[90, 0, 30]])

        self.rightArmPosOffsets = np.array([[.02, 0, .6]]) #Set of pos offests for the right arm end effector
        self.rightArmEulers = np.array([[90, 0, 0]]) #Set of end effector angles for right arm

    	self.kinectBowlFoundPosOffsets = [-.08, -.04, 0]

    	self.timeouts = [15, 7, 4, 4, 4, 12, 12]
        self.timeoutsR = [10, 10, 10]
    	self.kinectReachTimeout = 15

    	self.bowlQuatOffsets = self.euler2quatArray(self.bowlEulers) #converts the array of eulers to an array of quats
    	self.headQuatOffsets = self.euler2quatArray(self.headEulers)
        self.rightArmQuatOffsets = self.euler2quatArray(self.rightArmEulers)
        self.stopQuatOffsets = self.euler2quatArray(self.stopEulers)

    	print "Calculated quaternions:"
    	print self.bowlQuatOffsets

        try:
                print "--------------------------------"
                raw_input("Register bowl &  head position! Then press Enter \m")
                #self.tf_lstnr.waitForTransform('/torso_lift_link', 'head_frame', rospy.Time.now(), rospy.Duration(10))
                (self.headPos, self.headQuat) = self.tf_lstnr.lookupTransform('/torso_lift_link', 'head_frame', rospy.Time(0))
                print "Recived head position: \n"
                print self.headPos
                print self.headQuat
                print "--------------------------------"
                raw_input("Press Enter to confirm.")
                print "--------------------------------"
        except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print "Oops, can't get head_frame tf info!, trying again :)"

        rate = rospy.Rate(100) # 25Hz, nominally.
        while not rospy.is_shutdown():
            if self.getJointAngles() != []:
                print "--------------------------------"
                print "Current joint angles"
                print self.getJointAngles()
                print "Current pose"
                print self.getEndeffectorPose()
                print "--------------------------------"
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
        self.bowl_pos = np.matrix([ [data.pose.position.x + self.kinectBowlFoundPosOffsets[0]], [data.pose.position.y + self.kinectBowlFoundPosOffsets[1]], [data.pose.position.z + self.kinectBowlFoundPosOffsets[2]] ])
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

        posL  = Point()
        quatL = Quaternion()
        posR = Point()
        quatR = Quaternion()

        # duplication
        confirm = False
        while not confirm:
            print "Current pose"
            print self.getEndeffectorPose()
            ans = raw_input("Enter y to confirm to start: ")
            if ans == 'y':
                confirm = True
            print self.getJointAngles()

    	try:
    	    print self.bowl_pos
    	except:
    	    print "Please register bowl position!"

        #---------------------------------------------------------------------------------------#

    	kinectPose = raw_input('Press k in order to position spoon to Kinect-provided bowl position, used for testing: ')
    	if kinectPose == 'k':
    		#THIS IS SOME TEST CODE, BASICALLY PUTS THE SPOON WHERE THE KINECT THINKS THE BOWL IS, USED TO COMPARE ACTUAL BOWL POSITION WITH KINECT-PROVIDED BOWL POSITION!! UNCOMMENT ALL THIS OUT IF NOT USED MUCH!!#
    		print "MOVES_KINECT_BOWL_POSITION"
            	(posL.x, posL.y, posL.z) = (self.bowl_pos[0], self.bowl_pos[1], self.bowl_pos[2])
            	(quatL.x, quatL.y, quatL.z, quatL.w) = (0.632, 0.395, -0.205, 0.635)
            	#self.setPositionGoal(posL, quatL, self.timeout)
            	self.setOrientGoal(posL, quatL, self.kinectReachTimeout)
    		raw_input("Press Enter to continue")

        print "--------------------------------"

        self.initJoints()

        testL = raw_input("Press 'y' to test left arm movements!")
        if testL == 'y':
            posL.x, posL.y, posL.z = .5, .5, 0

            testEulers = np.array([[90, -60, 0],
                                   [90, -30, 0],
                                   [90,   0, 0]])
            testQuats = self.euler2quatArray(testEulers)
	    print "Calculated test quaternions: "
	    print testQuats
	    
            quatL.x, quatL.y, quatL.z, quatL.w = testQuats[0][0], testQuats[0][1], testQuats[0][2], testQuats[0][3]
            self.setOrientGoal(posL, quatL, 15)
	    print "Position should be: "
	    print posL
	    print "Quaternions should be: "
	    print quatL
            raw_input("Press Enter to continue")
            quatL.x, quatL.y, quatL.z, quatL.w = testQuats[1][0], testQuats[1][1], testQuats[1][2], testQuats[1][3]
            self.setOrientGoal(posL, quatL, 15)
	    print "Position should be: "
            print posL
            print "Quaternions should be: "
            print quatL
            raw_input("Press Enter to continue")
            quatL.x, quatL.y, quatL.z, quatL.w = testQuats[2][0], testQuats[2][1], testQuats[2][2], testQuats[2][3]
            self.setOrientGoal(posL, quatL, 15)
	    print "Position should be: "
            print posL
            print "Quaternions should be: "
            print quatL
            raw_input("Press Enter to continue")

    	raw_input("Press Enter to test both arms")
    	posL.x, posL.y, posL.z = 1, 0, 0
    	quatL.x, quatL.y, quatL.z, quatL.w = 0, 0, 0, 1
    	posR.x, posR.y, posR.z = 1, 0, 0
    	quatR.x, quatR.y, quatR.z, quatR.w = 0, 0, 0, 1
    	#self.setOrientGoal(posL, quatL, 10)
    	raw_input("Press Enter to continue")
    	#self.setOrientGoalRight(posR, quatR, 10)
    	raw_input("Press Enter to continue")

        print "MOVES1 - Pointing down over bowl "
        (posL.x, posL.y, posL.z) = (self.bowl_pos[0] + self.bowlPosOffsets[0][0], self.bowl_pos[1] + self.bowlPosOffsets[0][1], self.bowl_pos[2] + self.bowlPosOffsets[0][2])
        (quatL.x, quatL.y, quatL.z, quatL.w) = (self.bowlQuatOffsets[0][0], self.bowlQuatOffsets[0][1], self.bowlQuatOffsets[0][2], self.bowlQuatOffsets[0][3])
        #self.setPositionGoal(posL, quatL, self.timeout)
        self.setOrientGoal(posL, quatL, self.timeouts[0])

        armReachAction.iteration += 1

        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

    	print "--------------------------------"

        print "MOVES2 - Moving down into bowl"
        (posL.x, posL.y, posL.z) = (self.bowl_pos[0] + self.bowlPosOffsets[1][0], self.bowl_pos[1] + self.bowlPosOffsets[1][1], self.bowl_pos[2] + self.bowlPosOffsets[1][2])
        (quatL.x, quatL.y, quatL.z, quatL.w) = (self.bowlQuatOffsets[1][0], self.bowlQuatOffsets[1][1], self.bowlQuatOffsets[1][2], self.bowlQuatOffsets[1][3])
        #self.setPositionGoal(posL, quatL, self.timeout)
        self.setOrientGoal(posL, quatL, self.timeouts[1])

        armReachAction.iteration += 1

        raw_input("Iteration # = %d. Enter anything to continue: " % armReachAction.iteration)


    	print "--------------------------------"

        print "MOVES3 - Pushing forward in bowl, scooping"
        (posL.x, posL.y, posL.z) = (self.bowl_pos[0] + self.bowlPosOffsets[2][0], self.bowl_pos[1] + self.bowlPosOffsets[2][1], self.bowl_pos[2] + self.bowlPosOffsets[2][2])
        (quatL.x, quatL.y, quatL.z, quatL.w) = (self.bowlQuatOffsets[2][0], self.bowlQuatOffsets[2][1], self.bowlQuatOffsets[2][2], self.bowlQuatOffsets[2][3])
        #self.setPositionGoal(posL, quatL, self.timeout)
        self.setOrientGoal(posL, quatL, self.timeouts[2])

        armReachAction.iteration += 1

        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

    	print "--------------------------------"

        print "MOVES4 - Scooping in bowl"
        (posL.x, posL.y, posL.z) = (self.bowl_pos[0] + self.bowlPosOffsets[3][0], self.bowl_pos[1] +  self.bowlPosOffsets[3][1], self.bowl_pos[2] + self.bowlPosOffsets[3][2])
        (quatL.x, quatL.y, quatL.z, quatL.w) = (self.bowlQuatOffsets[3][0], self.bowlQuatOffsets[3][1], self.bowlQuatOffsets[3][2], self.bowlQuatOffsets[3][3])
        #self.setPositionGoal(posL, quatL, self.timeout)
        self.setOrientGoal(posL, quatL, self.timeouts[3])

        armReachAction.iteration += 1

        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

    	print "--------------------------------"

        print "MOVES5 - Lifting above bowl"
        (posL.x, posL.y, posL.z) = (self.bowl_pos[0] + self.bowlPosOffsets[4][0], self.bowl_pos[1] + self.bowlPosOffsets[4][1], self.bowl_pos[2] + self.bowlPosOffsets[4][2])
        (quatL.x, quatL.y, quatL.z, quatL.w) = (self.bowlQuatOffsets[4][0], self.bowlQuatOffsets[4][1], self.bowlQuatOffsets[4][2], self.bowlQuatOffsets[4][3])
        #self.setPositionGoal(posL, quatL, self.timeout)
        self.setOrientGoal(posL, quatL, self.timeouts[4])

        armReachAction.iteration += 1

        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

    	print "--------------------------------"

    	print "MOVES6 - Reaching to mouth"
    	try:
    		(posL.x, posL.y, posL.z) = (self.headPos[0] + self.headPosOffsets[0][0], self.headPos[1] + self.headPosOffsets[0][1], self.headPos[2] + self.headPosOffsets[0][2]);
        	(quatL.x, quatL.y, quatL.z, quatL.w) = (self.headQuatOffsets[0][0], self.headQuatOffsets[0][1], self.headQuatOffsets[0][2], self.headQuatOffsets[0][3])
        	self.setOrientGoal(posL, quatL, self.timeouts[5])

        	armReachAction.iteration += 1

        	raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)
        except:
            raw_input("Oops, can't get head_frame tf info, press Enter to continue: ")

    	print "--------------------------------"

    	print "MOVES7 - Moving away from mouth"

        try:
            (posL.x, posL.y, posL.z) = (self.headPos[0] + self.headPosOffsets[1][0], self.headPos[1] + self.headPosOffsets[1][1], self.headPos[2] + self.headPosOffsets[1][2])
            (quatL.x, quatL.y, quatL.z, quatL.w) = (self.headQuatOffsets[1][0], self.headQuatOffsets[1][1], self.headQuatOffsets[1][2], self.headQuatOffsets[1][3])
            #self.setPositionGoal(posL, quatL, self.timeout)
            self.setOrientGoal(posL, quatL, self.timeouts[6])

            armReachAction.iteration += 1

            raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)
        except:
            raw_input("Oops, can't get head_frame tf info, press Enter to continue")

        print "--------------------------------"

        print "MOVES8 - Moving RIGHT ARM above bowl"

        posR.x, posR.y, posR.z = (self.bowl_pos[0] + self.rightArmPosOffsets[0], self.bowl_pos[1] + self.rightArmPosOffsets[1], self.bowl_pos[2] + self.rightArmPosOffsets[2])
        quatR.x, quatR.y, quatR.z, quatR.w = (self.rightArmQuatOffsets[0], self.rightArmQuatOffsets[1], self.rightArmQuatOffsets[2], self.rightArmQuatOffsets[3])
        self.setOrientGoalRight(posR, quatR, 10) #Sends request to right arm server

        armReachAction.iteration += 1

        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

        return True

    def stopCallback(self, msg):

        print "Stopping Motion..."
        self.setStop() #Stops Current Motion
        self.setStopRight() #Sends message to service node
        posStopL = Point()
        quatStopL = Quaternion()

        print "Moving to safe position "
        (posStopL.x, posStopL.y, posStopL.z) = self.stopPos[0][0], self.stopPos[0][1], self.stopPos[0][2]
        (quatStopL.x, quatStopL.y, quatStopL.z, quatStopL.w) = self.stopQuatOffsets[0][0], self.stopQuatOffsets[0][1], self.stopQuatOffsets[0][2], self.stopQuatOffsets[0][3]
        self.setOrientGoal(posStop, quatStop, 10)

    def euler2quatArray(self, eulersIn): #converts an array of euler angles (in degrees) to array of quaternions
    	(rows, cols) = np.shape(eulersIn)
    	quatArray = np.zeros((rows, cols+1))
    	for r in xrange(0, rows):
    	    rads = np.radians([eulersIn[r][0], eulersIn[r][2], eulersIn[r][1]]) #CHECK THIS ORDER!!!
    	    quats = quatMath.euler2quat(rads[2], rads[1], rads[0])
    	    quatArray[r][0], quatArray[r][1], quatArray[r][2], quatArray[r][3] = quats[0], quats[1], quats[2], quats[3]

    	return quatArray

    def initJoints(self):

        initLeft = raw_input("Initialize left arm joint angles? [y/n]")
        if initLeft == 'y':
            print "Initializing left arm joint angles: "
            self.setPostureGoal(self.initialJointAnglesSideOfBodyLEFT, 7)
            raw_input("Press Enter to continue")
        initRight = raw_input("Initialize right arm joint angles? [y/n]")
        if initRight == 'y':
            print "Initializing right arm joint angles: "
            self.setPostureGoalRight(self.initialJointAnglesSideOfBodyRIGHT, 7)
	    raw_input("Press Enter to continue")

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

    try:
        arm = opt.arm1 #added/changed due to new launch file controlling both arms (arm1, arm2)
    except:
        arm = opt.arm

    rospy.init_node('arm_reacher_server')
    ara = armReachAction(d_robot, controller, arm)
    rospy.spin()
