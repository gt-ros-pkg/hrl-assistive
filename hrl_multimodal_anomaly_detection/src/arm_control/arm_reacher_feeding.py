#!/usr/bin/env python

import time
import rospy
import numpy as np

import roslib
# roslib.load_manifest('sandbox_dpark_darpa_m3')
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf

import hrl_haptic_mpc.haptic_mpc_util as haptic_mpc_util

from hrl_srvs.srv import None_Bool, None_BoolResponse, Int_Int
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from sandbox_dpark_darpa_m3.lib.hrl_mpc_base import mpcBaseAction
from hrl_multimodal_anomaly_detection.srv import PosQuatTimeoutSrv, AnglesTimeoutSrv, String_String
import hrl_lib.quaternion as quatMath 
from std_msgs.msg import String

class armReachAction(mpcBaseAction):


    def __init__(self, d_robot, controller, arm):

        mpcBaseAction.__init__(self, d_robot, controller, arm)

        #Subscribers to publishers of bowl location data
        #MAY NEED TO REMAP ROOT TOPIC NAME GROUP!
        rospy.Subscriber('hrl_feeding_task/manual_bowl_location',
                         PoseStamped, self.bowlPoseManualCallback)
        rospy.Subscriber('hrl_feeding_task/manual_head_location',
                         PoseStamped, self.headPoseManualCallback)

        rospy.Subscriber('hrl_feeding_task/RYDS_CupLocation',
                         PoseStamped, self.bowlPoseKinectCallback)

        rospy.Subscriber('hrl_feeding_task/emergency_arm_stop', String, self.stopCallback)

        # service request
        self.reach_service = rospy.Service('/arm_reach_enable', String_String, self.serverCallback)

        self.scoopingStepsClient = rospy.ServiceProxy('/scooping_steps_service', None_Bool)

        rospy.Subscriber('InterruptAction', String, self.interrupt)
        self.interrupted = False

        rArmServersRunning = False

        # Service Proxies for controlling right arm
        # Mimicks a built-in function with "Right" appended
        # To make use in code easier and more intuitive
        try:
            self.setOrientGoalRight = rospy.ServiceProxy(
                '/setOrientGoalRightService', PosQuatTimeoutSrv)
            self.setStopRight = rospy.ServiceProxy(
                '/setStopRightService', None_Bool)
            self.setPostureGoalRight = rospy.ServiceProxy(
                '/setPostureGoalRightService', AnglesTimeoutSrv)
            print "Connected to right arm server! "
            rArmServersRunning = True
        except:
            print "Oops, can't connect to right arm server!"

        #Stored initialization joint angles
        self.leftArmInitialJointAnglesScooping = [1.570, 0, 1.570, -1.570, -4.71, 0, -1.570]
        self.leftArmInitialJointAnglesFeeding = [0, 0, 1.57, 0, 0, -1.45, 0]
        self.rightArmInitialJointAnglesScooping = [0, 0, 0, 0, 0, 0, 0]
        self.rightArmInitialJointAnglesFeeding = [0, 0, 0, 0, 0, 0, 0]
        #^^ THESE NEED TO BE UPDATED!!!

        #Variables...! #
        armReachAction.iteration = 0

        self.posL = Point()
        self.quatL = Quaternion()
        self.posR = Point()
        self.quatR = Quaternion()


        #Array of offsets from bowl/head positions
        #Used to perform motions relative to bowl/head positions
        self.leftArmScoopingPos = np.array([[-.015,	0,	  .15],
                                            [-.015,	0,	-.065], #Moving down into bowl
                                            [.01,	0,	-.045], #Moving forward in bowl
                                            [0,		0,	  .05], #While rotating spoon to scoop out
                                            [0,		0,    .15]]) #Moving up out of bowl

        self.leftArmFeedingPos = np.array([[0,    .2,   0],
                                           [0,   .01,   0],
                                           [0,    .2,   0]])

        self.leftArmScoopingEulers = np.array([[90,	-50,    -30],
                                               [90,	-50,	-30], #Moving down into bowl
                                               [90,	-30,	-30], #Moving forward in bowl
                                               [90,	  0,	-30], #Rotating spoon to scoop out of bowl
                                               [90,	  0,    -30]]) #Moving up out of bowl

        self.leftArmFeedingEulers = np.array([[90, 0, -90],
                                              [90, 0, -90],
                                              [90, 0, -90]])

        self.leftArmStopPos = np.array([[.7, .7, .5]])
        self.leftArmStopEulers = np.array([[90, 0, 0]])

        #converts the array of eulers to an array of quats

        self.leftArmScoopingQuats = self.euler2quatArray(self.leftArmScoopingEulers)
        self.leftArmFeedingQuats = self.euler2quatArray(self.leftArmFeedingEulers)
        self.leftArmStopQuats = self.euler2quatArray(self.leftArmStopEulers)

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
        self.timeoutsScooping = [6, 3, 3, 2, 2]
        self.timeoutsFeeding = [3, 3, 3]

        #Paused used between each motion
        #... for automatic movement
        self.pausesScooping = [6, 3, 3, 2, 2]
        self.pausesFeeding = [3, 3, 3]

        print "Calculated quaternions: \n"
        print "leftArmScoopingQuats -"
        print self.leftArmScoopingQuats
        print "leftArmFeedingQuats -"
        print self.leftArmFeedingQuats
        print "leftArmStopQuats -"
        print self.leftArmStopQuats

        try:
                print "--------------------------------"
                #raw_input("Register bowl &  head position! Then press Enter \m")
                #self.tf_lstnr.waitForTransform('/torso_lift_link', 
                      #'head_frame', rospy.Time.now(), rospy.Duration(10))
                (self.headPos, self.headQuat) = self.tf_lstnr.lookupTransform('/torso_lift_link', 
                                                                              'head_frame', 
                                                                              rospy.Time(0))
                print "Recived Kinect provided head position: \n"
                print self.headPos
                print self.headQuat
                print "--------------------------------"
                #raw_input("Press Enter to confirm.")
                #print "--------------------------------"
        except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print "Oops, can't get head_frame tf info from Kinect! \n Will try to use manual!"

        rate = rospy.Rate(100) # 25Hz, nominally.
        while not rospy.is_shutdown():
            if self.getJointAngles() != []:
                print "--------------------------------"
                print "Current left arm joint angles"
                print self.getJointAngles()
                print "Current left arm pose"
                print self.getEndeffectorPose()
                print "--------------------------------"
                break

        rospy.spin()

    def interrupt(self, data):
        print '\n\nAction Interrupted! Event Stop\n\n'
        self.interrupted = True

    def serverCallback(self, req):
        req = req.data
        self.interrupted = False

        if req == "leftArmInitScooping":
            self.setPostureGoal(self.leftArmInitialJointAnglesScooping, 10)
            return "Initialized left arm for scooping!"

        elif req == "leftArmInitFeeding":
            self.setPostureGoal(self.leftArmInitialJointAnglesFeeding, 10)
            return "Initialized left arm for feeding!"

        elif req == "rightArmInitScooping":
            self.setPostureGoal(self.rightArmInitialJointAnglesScooping, 10)
            return "Initialized right arm for scooping!"

        elif req == "rightArmInitFeeding":
            self.setPostureGoal(self.rightArmInitialJointAnglesFeeding, 10)
            return "Initialized right arm for feeding!"

        elif req == "getBowlPosType":
            if self.bowl_pos_kinect is None and self.bowl_pos_manual is not None:
                return "manual"
            elif self.bowl_pos_manual is None and self.bowl_pos_kinect is not None:
                return "kinect"
            elif self.bowl_pos_manual is not None and self.bowl_pos_kinect is not None:
                return "both"

        elif req == "getHeadPosType":
            if self.head_pos_kinect is None and self.head_pos_manual is not None:
                return "manual"
            elif self.head_pos_manual is None and self.head_pos_kinect is not None:
                return "kinect"
            elif self.head_pos_manual is not None and self.head_pos_kinect is not None:
                return "both"

        elif req == "chooseManualBowlPos":
            if self.bowl_pos_manual is not None:
                self.bowl_frame = self.bowl_frame_manual
                self.bowl_pos = self.bowl_pos_manual
                self.bowl_quat = self.bowl_quat_manual
                return "Chose manual bowl position"
            else:
                return "No manual bowl position available! \n Code won't work! \n Provide bowl position and try again!"

        elif req == "chooseKinectBowlPos":
            if self.bowl_pos_kinect is not None:
                self.bowl_frame = self.bowl_frame_kinect
                self.bowl_pos = self.bowl_pos_kinect
                self.bowl_quat = self.bowl_quat_kinect
                return "Chose kinect bowl position"
            else:
                return "No kinect bowl position available! \n Code won't work! \n Provide bowl position and try again!"

        elif req == "chooseManualHeadPos":
            if self.head_pos_manual is not None:
                self.head_frame = self.head_frame_manual
                self.head_pos = self.head_pos_manual
                self.head_quat = self.head_quat_manual
                return "Chose manual head position"
            else:
                return "No manual head position available! \n Code won't work! \n Provide head position and try again!"

        elif req == "chooseKinectHeadPos":
            if self.head_pos_kinect is not None:
                self.head_frame = self.head_frame_kinect
                self.head_pos = self.head_pos_kinect
                self.head_quat = self.head_quat_kinect
                return "Chose kinect head position"
            else:
                return "No kinect head position available! \n Code won't work! \n Provide head position and try again!"

        elif req == "runScooping":
            self.scooping()
            return "Finished scooping!"

        elif req == "runFeeding":
            self.feeding()
            return "Finished feeding!"

        else:
            return "Request not understood by server!!!"

    def bowlPoseManualCallback(self, data):

        self.bowl_frame_manual = data.header.frame_id
        self.bowl_pos_manual = np.matrix([[data.pose.position.x],
            [data.pose.position.y], [data.pose.position.z]])
        self.bowl_quat_manual = np.matrix([[data.pose.orientation.x], [data.pose.orientation.y],
            [data.pose.orientation.z], [data.pose.orientation.w]])

    def bowlPoseKinectCallback(self, data):

        #Takes in a PointStamped() type message, contains Header() and Pose(),
        #from Kinect bowl location publisher
        self.bowl_frame_kinect = data.header.frame_id
        self.bowl_pos_kinect = np.matrix([[data.pose.position.x + self.kinectBowlFoundPosOffsets[0]],
            [data.pose.position.y + self.kinectBowlFoundPosOffsets[1]],
            [data.pose.position.z + self.kinectBowlFoundPosOffsets[2]]])
        self.bowl_quat_kinect = np.matrix([[data.pose.orientation.x], [data.pose.orientation.y],
            [data.pose.orientation.z], [data.pose.orientation.w]])

    def headPoseManualCallback(self, data):

        self.head_frame_manual = data.header.frame_id
        self.head_pos_manual = np.matrix([[data.pose.position.x], 
		[data.pose.position.y], [data.pose.position.z]])
        self.head_quat_manual = np.matrix([[data.pose.orientation.x], [data.pose.orientation.y],
            [data.pose.orientation.z], [data.pose.orientation.w]])

    def headPoseKinectCallback(self, data):

        self.head_frame_kinect = data.header.frame_id
        self.head_pos_kinect = np.matrix([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.head_quat_kinect = np.matrix([[data.pose.orientation.x], [data.pose.orientation.y],
            [data.pose.orientation.z], [data.pose.orientation.w]])

    # def chooseBowlPose(self):

    #     if self.bowl_pos_kinect is None and self.bowl_pos_manual is not None:
    #         print "No Kinect provided bowl information, using manually provided bowl information"
    #         print 'Manually Provided Bowl Pos: '
    #         print self.bowl_pos_manual
    #         print 'Manually Provided Bowl Quaternions: '
    #         print self.bowl_quat_manual
            
            
    #     elif self.bowl_pos_manual is None and self.bowl_pos_kinect is not None:
    #         print "No manually provided bowl information, using Kinect provided bowl information"
    #         print 'Kinect Provided Bowl Pos: '
    #         print self.bowl_pos_kinect
    #         print 'Kinect Provided Bowl Quaternions: '
    #         print self.bowl_quat_kinect
            
    #     elif self.bowl_pos_manual is not None and self.bowl_pos_kinect is not None:
    #         which_bowl = raw_input("Use Kinect or manually provided bowl position? [k/m] ")
    #         while which_bowl != 'k' and which_bowl != 'm':
    #             which_bowl = raw_input("Use Kinect or manually provided bowl position? [k/m] ")
    #         if which_bowl == 'k':
                
    #             print 'Kinect Provided Bowl Pos: '
    #             print self.bowl_pos_kinect
    #             print 'Kinect Provided Bowl Quaternions: '
    #             print self.bowl_quat_kinect
                
    #             self.bowl_frame = self.bowl_frame_kinect
    #             self.bowl_pos = self.bowl_pos_kinect
    #             self.bowl_quat = self.bowl_quat_kinect
    #         elif which_bowl == 'm':
                
    #             print 'Manually Provided Bowl Pos: '
    #             print self.bowl_pos_manual
    #             print 'Manually Provided Bowl Quaternions: '
    #             print self.bowl_quat_manua
                
    #             self.bowl_frame = self.bowl_frame_manual
    #             self.bowl_pos = self.bowl_pos_manual
    #             self.bowl_frame = self.bowl_quat_manual
    #     else:
    #         print "No bowl information available, publish info before running client/run again!! "
            
    #         return False
    #         #sys.exit()

    # def chooseHeadPose(self):

    #     if self.head_pos_kinect is None and self.head_pos_manual is not None:
    #         print "No Kinect provided head information, using manually provided head information"
            
    #     elif self.head_pos_manual is None and self.head_pos_kinect is not None:
    #         print "No manually provided head information, using Kinect provided head information"
            
    #     elif self.head_pos_manual is not None and self.head_pos_kinect is not None:
    #         which_head = raw_input("Use Kinect or manually provided head position? [k/m] ")
    #         while which_head != 'k' and which_head != 'm':
    #             which_head = raw_input("Use Kinect or manually provided head position? [k/m] ")
    #         if which_head == 'k':
    #             self.head_frame = self.head_frame_kinect
    #             self.head_pos = self.head_pos_kinect
    #             self.head_quat = self.head_quat_kinect
    #         elif which_bowl == 'm':
    #             self.head_frame = self.head_frame_manual
    #             self.head_pos = self.head_pos_manual
    #             self.head_frame = self.head_quat_manual
    #     else:
    #         print "No head information available, publish info before running client/run again!! "
    #         #sys.exit()


    def scooping(self):

        #self.chooseBowlPose()

        scoopingPrints = ['#1 Moving over bowl...',
                          '#2 Moving down into bowl...',
                          '#3 Moving forward in bowl...', 
                          '#4 Scooping in bowl...',
                          '#5 Moving out of bowl...']

        for i in xrange(len(self.pausesScooping)):
            print "Scooping step #%d " % i
            print scoopingPrints[i]
            self.posL.x, self.posL.y, self.posL.z = (self.bowl_pos[0] + self.leftArmScoopingPos[i][0],
                self.bowl_pos[1] + self.leftArmScoopingPos[i][1],
                self.bowl_pos[2] + self.leftArmScoopingPos[i][2])
            self.quatL.x, self.quatL.y, self.quatL.z, self.quatL.w = (self.leftArmScoopingQuats[i][0],
                self.leftArmScoopingQuats[i][1],
                self.leftArmScoopingQuats[i][2],
                self.leftArmScoopingQuats[i][3])

            self.setOrientGoal(self.posL, self.quatL, 0.01) # self.timeoutsScooping[i]
            scoopingTimes = self.scoopingStepsClient()
            print scoopingTimes
            print "Pausing for {} seconds ".format(self.pausesScooping[i])
            sleepCounter = 0.0
            while sleepCounter < self.pausesScooping[i] and not self.interrupted:
                time.sleep(0.1)
                sleepCounter += 0.1
            if self.interrupted:
                print 'Scooping action completed!!'
                return True

        print "Scooping action completed Gerr"

        return True

    def feeding(self):

        #self.chooseHeadPose()

        feedingPrints = ['##1 Moving in front of mouth...',
                          '#2 Moving into mouth...',
                          '#3 Moving away from mouth...']

        for i in xrange(len(self.pausesFeeding)):
            print 'Feeding step #%d ' % i
            print feedingPrints[i]
            self.posL.x, self.posL.y, self.posL.z = (self.head_pos[0] + self.leftArmFeedingPos[i][0],
                self.head_pos[1] + self.leftArmFeedingPos[i][1],
                self.head_pos[2] + self.leftArmFeedingPos[i][2])
            self.quatL.x, self.quatL.y, self.quatL.z, self.quatL.w = (self.leftArmFeedingQuats[i][0],
                self.leftArmFeedingQuats[i][1],
                self.leftArmFeedingQuats[i][2],
                self.leftArmFeedingQuats[i][3])

            self.setOrientGoal(self.posL, self.quatL, 0.01) # self.timeoutsFeeding[i]
            print 'Pausing for {} seconds '.format(self.pausesFeeding[i])
            feedingCounter = 0.0
            while feedingCounter < self.pausesFeeding[i] and not self.interrupted:
                time.sleep(0.1)
                feedingCounter += 0.1
            if self.interrupted:
                print 'Feeding action completed!!'
                return True

        print "Feeding action completed Gerr"

        return True

    def stopCallback(self, msg):

        print "Stopping Motion..."
        self.setStop() #Stops Current Motion
        try:
            self.setStopRight() #Sends message to service node
        except:
            print "Couldn't stop right arm! "

        posStopL = Point()
        quatStopL = Quaternion()

        print "Moving left arm to safe position "
        (posStopL.x, posStopL.y, posStopL.z) = (self.leftArmStopPos[0][0], 
            self.leftArmStopPos[0][1], 
            self.leftArmStopPos[0][2])
        (quatStopL.x, quatStopL.y, quatStopL.z, quatStopL.w) = (self.leftArmStopQuats[0][0], 
            self.leftArmStopQuats[0][1], 
            self.leftArmStopQuats[0][2], 
            self.leftArmStopQuats[0][3])
        self.setOrientGoal(posStopL, quatStopL, 10)

    #converts an array of euler angles (in degrees) to array of quaternions
    def euler2quatArray(self, eulersIn): 

        (rows, cols) = np.shape(eulersIn)
        quatArray = np.zeros((rows, cols+1))
        for r in xrange(0, rows):
            rads = np.radians([eulersIn[r][0], eulersIn[r][2], eulersIn[r][1]]) #CHECK THIS ORDER!!!
            quats = quatMath.euler2quat(rads[2], rads[1], rads[0])
            quatArray[r][0], quatArray[r][1], quatArray[r][2], quatArray[r][3] = (quats[0],
                                                                                  quats[1], 
                                                                                  quats[2], 
                                                                                  quats[3])

        return quatArray

    # def initJoints(self):

    #     initLeft = raw_input("Initialize left arm joint angles? [y/n]")
    #     if initLeft == 'y':
    #         initKind = raw_input("Iniitialize for feeding or scooping? [f/s]")
    #         while initKind != 'f' and initKind != 's':
    #             print "Please enter 'f' or 's' !"
    #             initKind = raw_input("Iniitialize for feeding or scooping? [f/s]")
    #         if initKind == 'f':
    #             print "Initializing left arm for feeding"
    #             self.setPostureGoal(self.leftArmInitialJointAnglesFeeding, 10)
    #         elif initKind == 's':
    #             print "Initializing left arm for scooping"
    #             self.setPostureGoal(self.leftArmInitialJointAnglesScooping, 10)
    #     initRight = raw_input("Initialize right arm joint angles? [y/n]")
    #     if initRight == 'y':
    #         initKind = raw_input("Initialize for feeding or scooping? [f/s]")
    #         while initKind != 'f' and initKind != 's':
    #             print "Please enter 'f' or 's' !"
    #             initKind = raw_input("Initialize for scooping or feeding? [f/s]")
    #         if initKind == 'f':
    #             print "Initializing right arm for feeding"
    #             self.setPostureGoalRight(self.rightArmInitialJointAnglesFeeding)
    #         elif initKind == 's':
    #             print "Initializing right arm for scooping"
    #             self.setPostureGoalRight(self.rightArmInitialJointAnglesScooping)
    #     print "initialization completed"

    # def run(self):

    #     whichTask = raw_input("Run scooping, feeding, or exit? [s/f/x] ")
    #     while whichTask != 's' and whichTask != 'f' and whichTask != 'x':
    #         print "Please enter 's' or 'f' or 'x' ! "
    #         whichTask = raw_input("Run scooping, feeding, or exit? [s/f/x] ")
    #     if whichTask == 's':
    #         print "Running scooping! "
    #         self.scooping()
    #     elif whichTask == 'f':
    #         print "Running feeding! "
    #         self.feeding()
    #     elif whichTask == 'x':
    #         print "Exiting program! "
    #         sys.exit()

    #         return True

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
        #arm = opt.arm1 
        #added/changed due to new launch file controlling both arms (arm1, arm2)
    #except:
        #arm = opt.arm

    arm = 'l'

    rospy.init_node('arm_reacher_feeding')
    ara = armReachAction(d_robot, controller, arm)
    rospy.spin()


