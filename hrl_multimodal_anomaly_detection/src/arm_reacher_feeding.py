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
        #MAY NEED TO REMAP ROOT TOPIC NAME GROUP!
        rospy.Subscriber('hrl_feeding_task/manual_bowl_location', PoseStamped, self.bowlPoseManualCallback)  # hrl_feeding/bowl_location
        rospy.Subscriber('hrl_feeding_task/manual_head_location', PoseStamped, self.headPoseManualCallback)

        rospy.Subscriber('hrl_feeding_task/RYDS_CupLocation', PoseStamped, self.bowlPoseKinectCallback)

        #rospy.Subscriber('hrl_feeding_task/emergency_arm_stop', String, self.stopCallback)

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
        self.leftArmInitialJointAnglesScooping = []
        self.leftArmInitialJointAnglesFeeding = []
        self.rightArmInitialJointAnglesHoldingBowl = []
        self.rightArmInitialJointAnglesFoldingUp = []

        #Variables...! #
        armReachAction.iteration = 0

        #Array of offsets from bowl/head positions
        #Used to perform motions relative to bowl/head positions
        self.leftArmScoopingPos = np.array([[-.01,	0,	   .4], 
                                            [-.01,	0,   .008], #Moving down into bowl
                                            [.05,	0,   .008], #Moving forward in bowl
                                            [.05,   0,   .008], #While rotating spoon to scoop out
                                        [   [-.01,  0,	   .4]]) #Moving up out of bowl

        self.leftArmFeedingPos = np.array([[.01,    .2,   -.01],
                                           [.01,    .085, -.01],
                                           [.01,    .2,   -.01]])

        self.leftArmScoopingEulers = np.array([[90,   0,    0], 
                                               [90, -60,	0], #Moving down into bowl
                                               [90,	-60,	0], #Moving forward in bowl
                                               [90,	-30,	0], #Rotating spoon to scoop out of bowl
                                               [90,	  0,    0]]) #Moving up out of bowl
        
        self.leftArmFeedingEulers = np.array([[90, 0, -90],
                                              [90, 0, -90],
                                              [90, 0, -90]])

        #converts the array of eulers to an array of quats
        self.leftArmScoopingQuats = self.euler2quatArray(self.leftArmScoopingEulers) 
        self.leftArmFeedingQuats = self.euler2quatArray(self.leftArmFeedingEulers)

        self.stopPos = np.array([[.7, .7, .5]])
        self.stopEulers = np.array([[90, 0, 30]])

        #Declares bowl positions options
        self.bowl_pos_manual = None
        self.bowl_pos_kinect = None
        self.head_pos_manual = None
        self.head_pos_kinect = None

        #How much to offset Kinect provided bowl position
    	self.kinectBowlFoundPosOffsets = [-.08, -.04, 0]
        #^ MAY BE REDUNDANT SINCE WE CAN ADD/SUBTRACT
        # ... THESE FROM ARRAY OF OFFSETS FOR SCOOPING!!!

        #Timeouts used in setOrientGoal() function for each motion
    	self.timeoutsScooping = [20, 7, 4, 4, 4]
        self.timeoutsFeeding = [10, 7, 5]

        #Paused used between each motion
        #... for automatic movement
        self.pausesScooping = self.timeoutsScooping
        self.pausesFeeding = self.timeoutsFeeding

    	print "Calculated quaternions:"
    	print self.bowlQuatOffsets

        # try:
        #         print "--------------------------------"
        #         raw_input("Register bowl &  head position! Then press Enter \m")
        #         #self.tf_lstnr.waitForTransform('/torso_lift_link', 'head_frame', rospy.Time.now(), rospy.Duration(10))
        #         (self.headPos, self.headQuat) = self.tf_lstnr.lookupTransform('/torso_lift_link', 'head_frame', rospy.Time(0))
        #         print "Recived head position: \n"
        #         print self.headPos
        #         print self.headQuat
        #         print "--------------------------------"
        #         raw_input("Press Enter to confirm.")
        #         print "--------------------------------"
        # except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #         print "Oops, can't get head_frame tf info!, trying again :)"

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

    def bowlBowlPoseManualCallback(self, data):
        self.bowl_frame_manual = data.header.frame_id
        self.bowl_pos_manual = np.matrix([ [data.pose.position.x], [data.pose.position.y], [data.pose.position.z] ])
        self.bowl_quat_manual = np.matrix([ [data.pose.orientation.x], [data.pose.orientation.y], [data.pose.orientation.z], [data.pose.orientation.w] ])
        print '-----------------------------------------------------'
        print 'Manually Provided Bowl Pos: '
        print self.bowl_pos_manual
        print 'Manually Provided Bowl Quaternions: '
        print self.bowl_quat_manual
        print '-----------------------------------------------------'

    def bowlPoseKinectCallback(self, data):
        #Takes in a PointStamped() type message, contains Header() and Pose(), from Kinect bowl location publisher
        self.bowl_frame_kinect = data.header.frame_id
        self.bowl_pos_kinect = np.matrix([ [data.pose.position.x + self.kinectBowlFoundPosOffsets[0]], [data.pose.position.y + self.kinectBowlFoundPosOffsets[1]], [data.pose.position.z + self.kinectBowlFoundPosOffsets[2]] ])
        self.bowl_quat_kinect = np.matrix([ [data.pose.orientation.x], [data.pose.orientation.y], [data.pose.orientation.z], [data.pose.orientation.w] ])
        print '-----------------------------------------------------'
        print 'Kinect Provided Bowl Pos: '
        print self.bowl_pos_kinect
        print 'Kinect Provided Bowl Quaternions: '
        print self.bowl_quat_kinect
        print '-----------------------------------------------------'

    def headPoseManualCallback(self, data):
        self.head_frame_manual
        self.head_pos_manual
        self.head_quat_manual

    def headPoseKinectCallback(self, data):

    def chooseBowlPose(self):
        if self.bowl_pos_kinect is None and self.bowl_pos_manual is not None:
            print "No Kinect provided bowl information, using manually provided bowl information"
            self.bowl_frame = self.bowl_frame_manual
            self.bowl_pos = self.bowl_pos_manual
            self.bowl_quat = self.bowl_quat_manual
        elif self.bowl_pos_manual is None and self.bowl_pos_kinect is not None:
            print "No manually provided bowl information, using Kinect provided bowl information"
            self.bowl_frame = self.bowl_frame_kinect
            self.bowl_pos = self.bowl_pos_kinect
            self.bowl_quat = self.bowl_quat_kinect
        elif self.bowl_pos_manual is not None and self.bowl_pos_kinect is not None:
            which_bowl = raw_input("Use Kinect or manually provided bowl position? [k/m] ")
            while which_bowl != 'k' and which_bowl != 'm':
                which_bowl = raw_input("Use Kinect or manually provided bowl position? [k/m] ")
            if which_bowl == 'k':
                self.bowl_frame = self.bowl_frame_kinect
                self.bowl_pos = self.bowl_pos_kinect
                self.bowl_quat = self.bowl_quat_kinect
            elif which_bowl == 'm':
                self.bowl_frame = self.bowl_frame_manual
                self.bowl_pos = self.bowl_pos_manual
                self.bowl_frame = self.bowl_quat_manual
        else:
            print "No bowl information available, publish info before running client/run again!! " 
            sys.exit()       

    def scooping(self):
        posL = Point()
        quatL = Quaternion()

        runScooping = True
        while runScooping:
            print "Initializing left arm for scooping... "
            self.setPostureGoal(self.leftArmInitialJointAnglesScooping)

            print "Current joint angles: "
            print self.getJointAngles()
            print "Current end effector pose: "
            print self.getEndeffectorPose()

            raw_input("Press anything to continue... ")

            print "#1 Moving over bowl... "
            posL.x, posL.y, posL.z = (self.bowl_pos[0] + self.leftArmScoopingPos[0][0], 
                self.bowl_pos[1] + self.leftArmScoopingPos[0][1], 
                self.bowl_pos[2] + self.leftArmScoopingPos[0][2])
            quatL.x, quatL.y, quatL.z, quatL.w = (self.leftArmScoopingQuats[0][0], self.leftArmScoopingQuats[0][1], 
                self.leftArmScoopingQuats[0][2], self.leftArmScoopingQuats[0][3])
            self.setOrientGoal(posL, quatL, timeoutsL[0])
            print "Pausing for {} seconds ".format(pausesL[0])
            time.sleep(pausesL[0])

            print "#2 Moving down into bowl... "
            posL.x, posL.y, posL.z = (self.bowl_pos[0] + self.leftArmScoopingPos[1][0], 
                self.bowl_pos[1] + self.leftArmScoopingPos[1][1], 
                self.bowl_pos[2] + self.leftArmScoopingPos[1][2])
            quatL.x, quatL.y, quatL.z, quatL.w = (self.leftArmScoopingQuats[1][0], self.leftArmScoopingQuats[1][1], 
                self.leftArmScoopingQuats[1][2], self.leftArmScoopingQuats[1][3])
            self.setOrientGoal(posL, quatL, timeoutsL[1])
            print "Pausing for {} seconds ".format(pausesL[1])
            time.sleep(pausesL[1])

            print "#3 Moving forward in bowl... "
            posL.x, posL.y, posL.z = (self.bowl_pos[0] + self.leftArmScoopingPos[2][0], 
                self.bowl_pos[1] + self.leftArmScoopingPos[2][1], 
                self.bowl_pos[2] + self.leftArmScoopingPos[2][2])
            quatL.x, quatL.y, quatL.z, quatL.w = (self.leftArmScoopingQuats[2][0], self.leftArmScoopingQuats[2][1], 
                self.leftArmScoopingQuats[2][2], self.leftArmScoopingQuats[2][3])
            self.setOrientGoal(posL, quatL, timeoutsL[2])
            print "Pausing for {} seconds ".format(pausesL[2])
            time.sleep(pausesL[2])

            print "#4 Scooping in bowl... "
            posL.x, posL.y, posL.z = (self.bowl_pos[0] + self.leftArmScoopingPos[3][0], 
                self.bowl_pos[1] + self.leftArmScoopingPos[3][1], 
                self.bowl_pos[2] + self.leftArmScoopingPos[3][2])
            quatL.x, quatL.y, quatL.z, quatL.w = (self.leftArmScoopingQuats[3][0], self.leftArmScoopingQuats[3][1], 
                self.leftArmScoopingQuats[3][2], self.leftArmScoopingQuats[3][3])
            self.setOrientGoal(posL, quatL, timeoutsL[3])
            print "Pausing for {} seconds ".format(pausesL[3])
            time.sleep(pausesL[3])

            print "#5 Moving out of bowl... "
            posL.x, posL.y, posL.z = (self.bowl_pos[0] + self.leftArmScoopingPos[4][0], 
                self.bowl_pos[1] + self.leftArmScoopingPos[4][1], 
                self.bowl_pos[2] + self.leftArmScoopingPos[4][2])
            quatL.x, quatL.y, quatL.z, quatL.w = (self.leftArmScoopingQuats[4][0], self.leftArmScoopingQuats[4][1], 
                self.leftArmScoopingQuats[4][2], self.leftArmScoopingQuats[4][3])
            self.setOrientGoal(posL, quatL, timeoutsL[4])
            print "Pausing for {} seconds ".format(pausesL[4])
            time.sleep(pausesL[4])

            print "Scooping action completed"

            runScoopingAns = raw_input("Run scooping again? [y/n] ")
            while runScoopingAns != 'y' and runScoopingAns != 'n':
                print "Please enter 'y' or 'n' ! "
                runScoopingAns = raw_input("Run scooping again? [y/n] ")
            if runScoopingAns == 'y':
                runScooping = True
            elif runScoopingAns == 'n':
                runScooping = False

        print "Done running scooping!"

    def feeding(self):
        posL = Point()
        quatL = Quaternion()

        runFeeding = True
        while runFeeding:
            print "Initializing left arm for feeding... "
            self.setPostureGoal(self.leftArmInitialJointAnglesFeeding)

            print "Current joint angles: "
            print self.getJointAngles()
            print "Current end effector pose: "
            print self.getEndeffectorPose()

            raw_input("Press anything to continue... ")

            print "Feeding action completed"

            runFeedingAns = raw_input("Run feeding again? [y/n] ")
            while runFeedingAns != 'y' and runFeedingAns != 'n':
                print "Please enter 'y' or 'n' ! "
                runFeedingAns = raw_input("Run feeding again? [y/n] ")
            if runFeedingAns == 'y':
                runFeeding = True
            elif runFeedingAns == 'n':
                runFeeding = False

        print "Done running feeding!"



    def run(self):
        self.chooseBowlPose()
        self.chooseHeadPose()
        whichTask = raw_input("Run scooping, feeding, or exit? [s/f/x] ")
        while whichTask != 's' and whichTask != 'f' and whichTask != 'x':
            print "Please enter 's' or 'f' or 'x' ! "
            whichTask = raw_input("Run scooping, feeding, or exit? [s/f/x] ")
        if whichTask == 's':
            print "Running scooping! "
            self.scooping()
        elif whichTask == 'f':
            print "Running feeding! "
            self.feeding()
        elif whichTask == 'x':
            print "Exiting program! "
            sys.exit()

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
            initKind = raw_input("Iniitialize for feeding or scooping? [f/s]")
            while initKind != 'f' and initKind != 's':
                print "Please enter 'f' or 's' !"
                initKind = raw_input("Iniitialize for feeding or scooping? [f/s]")
            if initKind == 'f':
                print "Initializing left arm for feeding"
                self.setPostureGoal(self.leftArmInitialJointAnglesFeeding)
            elif initKind == 's':
                print "Initializing left arm for scooping"
                self.setPostureGoal(self.leftArmInitialJointAnglesScooping)
        initRight = raw_input("Initialize right arm joint angles? [y/n]")
        if initRight == 'y':
            initKind = raw_input("Initialize for holding bowl or folding up? [b/f]")
            while initKind != 'b' and initKind != 'f':
                print "Please enter 'b' or 'f' !"
                initKind = raw_input("Initialize for holding bowl or folding up? [b/f]")
            if initKind == 'b':
                print "Initializing for holding bowl"
                self.setPostureGoalRight(self)
            elif initKind == 'f':
                print "Initializing for folding up out of the way"
                sel.setPostureGoalRight(self)
        print "initialization completed"

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

    #try:
        #arm = opt.arm1 #added/changed due to new launch file controlling both arms (arm1, arm2)
    #except:
        #arm = opt.arm

    arm = 'l'

    rospy.init_node('arm_reacher_feeding')
    ara = armReachAction(d_robot, controller, arm)
    rospy.spin()


#-------------------------------------------------------------


 #        print "MOVES 1 - Pointing down over bowl "
 #        (posL.x, posL.y, posL.z) = (self.bowl_pos[0] + self.bowlPosOffsets[0][0], self.bowl_pos[1] + self.bowlPosOffsets[0][1], self.bowl_pos[2] + self.bowlPosOffsets[0][2])
 #        (quatL.x, quatL.y, quatL.z, quatL.w) = (self.bowlQuatOffsets[0][0], self.bowlQuatOffsets[0][1], self.bowlQuatOffsets[0][2], self.bowlQuatOffsets[0][3])
 #        #self.setPositionGoal(posL, quatL, self.timeout)
 #        self.setOrientGoal(posL, quatL, self.timeoutsL[0])

 #        armReachAction.iteration += 1

 #        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

 #      print "--------------------------------"

 #        print "MOVES 2 - Moving down into bowl"
 #        (posL.x, posL.y, posL.z) = (self.bowl_pos[0] + self.bowlPosOffsets[1][0], self.bowl_pos[1] + self.bowlPosOffsets[1][1], self.bowl_pos[2] + self.bowlPosOffsets[1][2])
 #        (quatL.x, quatL.y, quatL.z, quatL.w) = (self.bowlQuatOffsets[1][0], self.bowlQuatOffsets[1][1], self.bowlQuatOffsets[1][2], self.bowlQuatOffsets[1][3])
 #        #self.setPositionGoal(posL, quatL, self.timeout)
 #        self.setOrientGoal(posL, quatL, self.timeoutsL[1])

 #        armReachAction.iteration += 1

 #        raw_input("Iteration # = %d. Enter anything to continue: " % armReachAction.iteration)


 #      print "--------------------------------"

 #        print "MOVES 3 - Pushing forward in bowl, scooping"
 #        (posL.x, posL.y, posL.z) = (self.bowl_pos[0] + self.bowlPosOffsets[2][0], self.bowl_pos[1] + self.bowlPosOffsets[2][1], self.bowl_pos[2] + self.bowlPosOffsets[2][2])
 #        (quatL.x, quatL.y, quatL.z, quatL.w) = (self.bowlQuatOffsets[2][0], self.bowlQuatOffsets[2][1], self.bowlQuatOffsets[2][2], self.bowlQuatOffsets[2][3])
 #        #self.setPositionGoal(posL, quatL, self.timeout)
 #        self.setOrientGoal(posL, quatL, self.timeoutsL[2])

 #        armReachAction.iteration += 1

 #        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

 #      print "--------------------------------"

 #        print "MOVES 4 - Scooping in bowl"
 #        (posL.x, posL.y, posL.z) = (self.bowl_pos[0] + self.bowlPosOffsets[3][0], self.bowl_pos[1] +  self.bowlPosOffsets[3][1], self.bowl_pos[2] + self.bowlPosOffsets[3][2])
 #        (quatL.x, quatL.y, quatL.z, quatL.w) = (self.bowlQuatOffsets[3][0], self.bowlQuatOffsets[3][1], self.bowlQuatOffsets[3][2], self.bowlQuatOffsets[3][3])
 #        #self.setPositionGoal(posL, quatL, self.timeout)
 #        self.setOrientGoal(posL, quatL, self.timeoutsL[3])

 #        armReachAction.iteration += 1

 #        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

 #      print "--------------------------------"

 #        print "MOVES 5 - Lifting above bowl"
 #        (posL.x, posL.y, posL.z) = (self.bowl_pos[0] + self.bowlPosOffsets[4][0], self.bowl_pos[1] + self.bowlPosOffsets[4][1], self.bowl_pos[2] + self.bowlPosOffsets[4][2])
 #        (quatL.x, quatL.y, quatL.z, quatL.w) = (self.bowlQuatOffsets[4][0], self.bowlQuatOffsets[4][1], self.bowlQuatOffsets[4][2], self.bowlQuatOffsets[4][3])
 #        #self.setPositionGoal(posL, quatL, self.timeout)
 #        self.setOrientGoal(posL, quatL, self.timeoutsL[4])

 #        armReachAction.iteration += 1

 #        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

 #      print "--------------------------------"

 #      print "MOVES 6 - Reaching to mouth"
 #      try:
 #          (posL.x, posL.y, posL.z) = (self.headPos[0] + self.headPosOffsets[0][0], self.headPos[1] + self.headPosOffsets[0][1], self.headPos[2] + self.headPosOffsets[0][2]);
 #          (quatL.x, quatL.y, quatL.z, quatL.w) = (self.headQuatOffsets[0][0], self.headQuatOffsets[0][1], self.headQuatOffsets[0][2], self.headQuatOffsets[0][3])
 #          self.setOrientGoal(posL, quatL, self.timeoutsL[5])

 #          armReachAction.iteration += 1

 #          raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)
 #        except:
 #            raw_input("Oops, can't get head_frame tf info, press Enter to continue: ")

 #      print "--------------------------------"

    # print "MOVES 7 - Moving left arm back to original position"
    # self.setPostureGoal(self.initialJointAnglesSideFacingFowardLEFT, 7)
    # armReachAction.iteration += 1

 #        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

 #        print "--------------------------------"

 #        print "MOVES 8 - Moving RIGHT ARM in front of face"

 #        posR.x, posR.y, posR.z = (self.headPos[0] + self.rightArmPosOffsets[0][0], self.headPos[1] + self.rightArmPosOffsets[0][1], self.headPos[2] + self.rightArmPosOffsets[0][2])
 #        quatR.x, quatR.y, quatR.z, quatR.w = (self.rightArmQuatOffsets[0][0], self.rightArmQuatOffsets[0][1], self.rightArmQuatOffsets[0][2], self.rightArmQuatOffsets[0][3])
 #        self.setOrientGoalRight(posR, quatR, 10) #Sends request to right arm server

 #        armReachAction.iteration += 1

 #        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

    # print "--------------------------------"

 #        print "MOVES 9 - Moving RIGHT ARM away from/above face"

 #        posR.x, posR.y, posR.z = (self.headPos[0] + self.rightArmPosOffsets[1][0], self.headPos[1] + self.rightArmPosOffsets[1][1], self.headPos[2] + self.rightArmPosOffsets[1][2])
 #        quatR.x, quatR.y, quatR.z, quatR.w = (self.rightArmQuatOffsets[1][0], self.rightArmQuatOffsets[1][1], self.rightArmQuatOffsets[1][2], self.rightArmQuatOffsets[1][3])
 #        self.setOrientGoalRight(posR, quatR, 10) #Sends request to right arm server

 #        armReachAction.iteration += 1

 #        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)

 #        print "--------------------------------"

    # print "MOVES 10 - Moving RIGHT ARM back to original position"
    # self.setPostureGoalRight(self.initialJointAnglesSideOfBodyRIGHT, 7)

    # armReachAction.iteration += 1

 #        raw_input("Iteration # %d. Enter anything to continue: " % armReachAction.iteration)