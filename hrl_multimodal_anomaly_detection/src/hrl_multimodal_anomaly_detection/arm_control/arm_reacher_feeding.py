#!/usr/bin/env python

import sys, time, copy
import rospy
import numpy as np

import roslib
roslib.load_manifest('sandbox_dpark_darpa_m3')
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf

import hrl_haptic_mpc.haptic_mpc_util as haptic_mpc_util

from hrl_srvs.srv import None_Bool, None_BoolResponse, Int_Int
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from hrl_multimodal_anomaly_detection.srv import PosQuatTimeoutSrv, AnglesTimeoutSrv, String_String

from sandbox_dpark_darpa_m3.lib.hrl_mpc_base import mpcBaseAction
import hrl_lib.quaternion as quatMath 
from std_msgs.msg import String
import PyKDL

class armReachAction(mpcBaseAction):
    def __init__(self, d_robot, controller, arm, head_noise=False):
        mpcBaseAction.__init__(self, d_robot, controller, arm)

        #Variables...! #
        self.interrupted = False
        self.iteration = 0 ##?????????????????????????????

        self.posL = Point()
        self.quatL = Quaternion()
        self.posR = Point()
        self.quatR = Quaternion()
        
        #Declares bowl positions options ## looks redundant variables
        self.bowl_pos_manual = None
        self.bowl_pos_kinect = None
        self.mouth_pos_manual = None
        self.mouth_pos_kinect = None

        # Artificial error
        self.head_noise = head_noise
        
        self.initCommsForArmReach()                            
        self.initParamsForArmReach()

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
            rate.sleep()
            
        rospy.loginfo("Arm Reach Action is initialized.")
                            
    def initCommsForArmReach(self):
        #Subscribers to publishers of bowl location data
        #MAY NEED TO REMAP ROOT TOPIC NAME GROUP!
        rospy.Subscriber('/ar_track_alvar/bowl_cen_pose',
                         PoseStamped, self.bowlPoseManualCallback)
        rospy.Subscriber('/ar_track_alvar/mouth_pose',
                         PoseStamped, self.mouthPoseManualCallback)
        rospy.Subscriber('InterruptAction', String, self.interruptCallback)

        ## rospy.Subscriber('hrl_feeding_task/RYDS_CupLocation',
        ##                  PoseStamped, self.bowlPoseKinectCallback)

        #rospy.Subscriber('hrl_feeding_task/emergency_arm_stop', String, self.stopCallback)

        

        # service
        self.reach_service = rospy.Service('arm_reach_enable', String_String, self.serverCallback)
        self.scoopingStepsClient = rospy.ServiceProxy('/scooping_steps_service', None_Bool)

        # Service Proxies for controlling right arm
        # Mimicks a built-in function with "Right" appended
        # To make use in code easier and more intuitive
        ## try:
        ##     self.setOrientGoalRight = rospy.ServiceProxy(
        ##         '/setOrientGoalRightService', PosQuatTimeoutSrv)
        ##     self.setStopRight = rospy.ServiceProxy(
        ##         '/setStopRightService', None_Bool)
        ##     self.setPostureGoalRight = rospy.ServiceProxy(
        ##         '/setPostureGoalRightService', AnglesTimeoutSrv)
        ##     print "Connected to right arm server!"
        ## except:
        ##     print "Oops, can't connect to right arm server!"        
            
        rospy.loginfo("ROS-based communications are set up .")
                                    
    def initParamsForArmReach(self):
        
        #Stored initialization joint angles
        self.leftArmInitialJointAnglesScooping = [1.570, 0, 1.570, -1.570, -4.71, 0, -1.570]
        self.leftArmInitialJointAnglesFeeding = [0.397, 0.272, 1.088, -2.11, -3.78, -0.658, -2.288] # not used
        self.rightArmInitialJointAnglesScooping = [0, 0, 0, 0, 0, 0, 0]
        self.rightArmInitialJointAnglesFeeding = [0, 0, 0, 0, 0, 0, 0]
        #^^ THESE NEED TO BE UPDATED!!!
        
        #Array of offsets from bowl/mouth positions
        #Used to perform motions relative to bowl/mouth positions > It should use relative frame
        self.leftArmScoopingPos = np.array([[-.015,	-0.02,	  .15],
                                            [-.015,	-0.02,	-.055], #Moving down into bowl
                                            [.02,	-0.02,	-.025], #Moving forward in bowl
                                            [0,	 -0.03,	  .20], #While rotating spoon to scoop out
                                            [0,	 -0.03,    .25]]) #Moving up out of bowl

        # It uses the l_gripper_spoon_frame aligned with mouth
        self.leftArmFeedingPos = np.array([[-0.15, 0, 0.0],
                                           [0.07, 0, 0.0],
                                           [-0.15, 0, 0.0]])
        ## self.leftArmFeedingPos = np.array([[-0.2, 0, 0.0],
        ##                                    [0.05, 0, 0.0],
        ##                                    [-0.2, 0, 0.0]])

        self.leftArmScoopingEulers = np.array([[90,	-50,    -30],
                                               [90,	-50,	-30], #Moving down into bowl
                                               [90,	-30,	-30], #Moving forward in bowl
                                               [90,	  0,	-30], #Rotating spoon to scoop out of bowl
                                               [90,	  0,    -30]]) #Moving up out of bowl

        self.leftArmFeedingEulers = np.array([[90, 0, -75],
                                              [90, 0, -75],
                                              [90, 0, -75]])

        self.leftArmStopPos = np.array([[.7, .7, .5]])
        self.leftArmStopEulers = np.array([[90.0, 0, 0]])

        #converts the array of eulers to an array of quats
        self.leftArmScoopingQuats = self.euler2quatArray(self.leftArmScoopingEulers)
        self.leftArmFeedingQuats = self.euler2quatArray(self.leftArmFeedingEulers)
        self.leftArmStopQuats = self.euler2quatArray(self.leftArmStopEulers)

        #Timeouts used in setOrientGoal() function for each motion
        self.timeoutsScooping = [6, 3, 3, 2, 2]
        self.timeoutsFeeding = [3, 3, 3]

        #Paused used between each motion
        #... for automatic movement
        self.pausesScooping = [0, 0, 0, 0, 0]
        self.pausesFeeding = [0., 0.0, 0.5]

        print "Calculated quaternions: \n"
        print "leftArmScoopingQuats -"
        print self.leftArmScoopingQuats
        print "leftArmFeedingQuats -"
        print self.leftArmFeedingQuats
        print "leftArmStopQuats -"
        print self.leftArmStopQuats

        rospy.loginfo("Parameters are loaded.")
                
        
    def serverCallback(self, req):
        req = req.data
        self.interrupted = False

        if req == "leftArmInitScooping":
            self.setPostureGoal(self.leftArmInitialJointAnglesScooping, 10)
            return "Initialized left arm for scooping!"

        elif req == "leftArmInitFeeding":
            self.setPostureGoal(self.leftArmInitialJointAnglesFeeding, 10)
            ## self.posL.x, self.posL.y, self.posL.z = 0.5, -0.1, 0
            ## self.quatL.x, self.quatL.y, self.quatL.z, self.quatL.w = (self.leftArmFeedingQuats[0][0],
            ##                                                           self.leftArmFeedingQuats[0][1],
            ##                                                           self.leftArmFeedingQuats[0][2],
            ##                                                           self.leftArmFeedingQuats[0][3])
            ## self.setOrientGoal(self.posL, self.quatL, 10)
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
            if self.mouth_pos_kinect is None and self.mouth_pos_manual is not None:
                return "manual"
            elif self.mouth_pos_manual is None and self.mouth_pos_kinect is not None:
                return "kinect"
            elif self.mouth_pos_manual is not None and self.mouth_pos_kinect is not None:
                return "both"

        elif req == "chooseManualBowlPos":
            if self.bowl_pos_manual is not None:
                self.bowl_frame = self.bowl_frame_manual
                self.bowl_pos = self.bowl_pos_manual
                self.bowl_quat = self.bowl_quat_manual
                return "Chose manual bowl position"
            else:
                return "No manual bowl position available! \n Code won't work! \n \
                Provide bowl position and try again!"

        elif req == "chooseKinectBowlPos":
            if self.bowl_pos_kinect is not None:
                self.bowl_frame = self.bowl_frame_kinect
                self.bowl_pos = self.bowl_pos_kinect
                self.bowl_quat = self.bowl_quat_kinect
                return "Chose kinect bowl position"
            else:
                return "No kinect bowl position available! \n Code won't work! \n \
                Provide bowl position and try again!"

        elif req == "chooseManualHeadPos":
            if self.mouth_pos_manual is not None:
                self.mouth_frame = self.mouth_frame_manual
                self.mouth_pos = self.mouth_pos_manual
                self.mouth_quat = self.mouth_quat_manual
                return "Chose manual head position"
            else:
                return "No manual head position available! \n Code won't work! \n \
                Provide head position and try again!"

        elif req == "chooseKinectHeadPos":
            if self.mouth_pos_kinect is not None:
                self.mouth_frame = self.mouth_frame_kinect
                self.mouth_pos = self.mouth_pos_kinect
                self.mouth_quat = self.mouth_quat_kinect
                return "Chose kinect head position"
            else:
                return "No kinect head position available! \n Code won't work! \n \
                Provide head position and try again!"

        elif req == 'initArmScooping':
            self.scooping([0])
            return 'Initialized'

        elif req == "runScooping":
            self.scooping(xrange(1, 5))
            return "Finished scooping!"

        elif req == 'initArmFeeding':
            self.feeding([0])
            return 'Initialized'

        elif req == "runFeeding":
            self.feeding(xrange(1, 3))
            return "Finished feeding!"

        else:
            return "Request not understood by server!!!"

    def bowlPoseManualCallback(self, data):
        self.bowl_frame_manual = data.header.frame_id
        self.bowl_pos_manual = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.bowl_quat_manual = np.array([data.pose.orientation.x, data.pose.orientation.y, 
                                          data.pose.orientation.z, data.pose.orientation.w])

    def bowlPoseKinectCallback(self, data):

        #Takes in a PointStamped() type message, contains Header() and Pose(),
        #from Kinect bowl location publisher
        self.bowl_frame_kinect = data.header.frame_id
        self.bowl_pos_kinect = np.array([data.pose.position.x + self.kinectBowlFoundPosOffsets[0],
                                         data.pose.position.y + self.kinectBowlFoundPosOffsets[1],
                                         data.pose.position.z + self.kinectBowlFoundPosOffsets[2]])
        self.bowl_quat_kinect = np.array([data.pose.orientation.x, data.pose.orientation.y,
                                          data.pose.orientation.z, data.pose.orientation.w])
        
    def mouthPoseManualCallback(self, data):

        self.mouth_frame_manual = data.header.frame_id
        self.mouth_pos_manual = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.mouth_quat_manual = np.array([data.pose.orientation.x, data.pose.orientation.y,
                                           data.pose.orientation.z, data.pose.orientation.w])

    def mouthPoseKinectCallback(self, data):

        self.mouth_frame_kinect = data.header.frame_id
        self.mouth_pos_kinect = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.mouth_quat_kinect = np.array([data.pose.orientation.x, data.pose.orientation.y,
                                           data.pose.orientation.z, data.pose.orientation.w])

    def scooping(self, iterations):

        #self.chooseBowlPose()

        scoopingPrints = ['#1 Moving over bowl...',
                          '#2 Moving down into bowl...',
                          '#3 Moving forward in bowl...', 
                          '#4 Scooping in bowl...',
                          '#5 Moving out of bowl...']

        for i in iterations:
            print "Scooping step #%d " % i
            print scoopingPrints[i]
            self.posL.x, self.posL.y, self.posL.z = (self.bowl_pos[0] + self.leftArmScoopingPos[i][0],
                self.bowl_pos[1] + self.leftArmScoopingPos[i][1],
                self.bowl_pos[2] + self.leftArmScoopingPos[i][2])
            self.quatL.x, self.quatL.y, self.quatL.z, self.quatL.w = (self.leftArmScoopingQuats[i][0],
                self.leftArmScoopingQuats[i][1],
                self.leftArmScoopingQuats[i][2],
                self.leftArmScoopingQuats[i][3])

            self.setOrientGoal(self.posL, self.quatL, self.timeoutsScooping[i])

            # Daehyung: It should be removed from this server. This program should not depend on any data 
            #           recording program. This program can publish data, but it should not request answer 
            #           other program except arm_reacher_client.
            ## scoopingTimes = self.scoopingStepsClient()
            ## print scoopingTimes
            
            print "Pausing for {} seconds ".format(self.pausesScooping[i])
            if self.interrupted:
                break

        print "Scooping action completed"

        return True

    def feeding(self, iterations):

        #self.chooseHeadPose()

        feedingPrints = ['#1 Moving in front of mouth...',
                          '#2 Moving into mouth...',
                          '#3 Moving away from mouth...']

        mouth_pos = self.mouth_pos_manual
        mouth_quat = self.mouth_quat_manual                

        for i in iterations:
            print 'Feeding step #%d ' % i
            print feedingPrints[i]

            mouth_rot = PyKDL.Rotation.Quaternion(mouth_quat[0], mouth_quat[1], mouth_quat[2], mouth_quat[3])

            spoon_x = -mouth_rot.UnitZ()
            spoon_y = PyKDL.Vector(0, 0, 1.0)
            spoon_z = spoon_x * spoon_y
            spoon_y = spoon_z * spoon_x
            spoon_rot = PyKDL.Rotation(spoon_x, spoon_y, spoon_z)

            spoon_offset = PyKDL.Vector(self.leftArmFeedingPos[i][0], self.leftArmFeedingPos[i][1], 
                                        self.leftArmFeedingPos[i][2])
            spoon_offset = spoon_rot * spoon_offset

            head_pos_noise = np.zeros(3)
            if self.head_noise and i==1:
                print "Head pose noise on feeding sequence 1"
                min_rad = 0.03
                max_rad = 0.08
                x = 0.0
                
                y = np.random.uniform(min_rad, max_rad, 1)                           
                y_sign = np.random.uniform(-1.0, 1.0, 1)
                if y_sign < 0.0: y = y*-1.0
                    
                z = np.random.uniform(min_rad, max_rad, 1)                           
                z_sign = np.random.uniform(-1.0, 1.0, 1)
                if z_sign < 0.0: z = z*-1.0
                    
                noise_offset = PyKDL.Vector(x,y,z)
                head_pos_noise = spoon_rot * noise_offset
                print "pos offset: ", x, y, z

                min_ang = -15.*np.pi/180.0
                max_ang = 15.*np.pi/180.0
                roll = 0.0
                pitch = np.random.uniform(min_ang, max_ang, 1)                           
                yaw = 0.0 
                noise_rot = PyKDL.Rotation.RPY(roll, pitch, yaw )                           
                spoon_rot = spoon_rot * noise_rot

                print "rpy error: ", roll , pitch, yaw
                ## print "before: ", self.quatL
                ## ## quatL = quatMath.quat_QuTem(spoon_rot.GetQuaternion(), 1, [0.0001,0.0001,0.0001])[0]
                ## self.quatL.x, self.quatL.y, self.quatL.z, self.quatL.w = (quatL[0],quatL[1],quatL[2],quatL[3])
                ## print "after: ", self.quatL
            
            self.posL.x, self.posL.y, self.posL.z = (mouth_pos[0] + spoon_offset[0] + head_pos_noise[0],
                                                     mouth_pos[1] + spoon_offset[1] + head_pos_noise[1],
                                                     mouth_pos[2] + spoon_offset[2] + head_pos_noise[2])
            self.quatL.x, self.quatL.y, self.quatL.z, self.quatL.w = (spoon_rot.GetQuaternion()[0],
                                                                      spoon_rot.GetQuaternion()[1],
                                                                      spoon_rot.GetQuaternion()[2],
                                                                      spoon_rot.GetQuaternion()[3])
            
            self.setOrientGoal(self.posL, self.quatL, self.timeoutsFeeding[i])
            print 'Pausing for {} seconds '.format(self.pausesFeeding[i])

            # Daehyung: don't use time sleep. use rospy.sleep. the time sleep will stop communication either. 
            ## time.sleep(self.pausesFeeding[i])
            if self.interrupted:
                break

        print "Feeding action completed"

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


    def interruptCallback(self, data):
        print '\n\nAction Interrupted! Event Stop\n\n'
        print 'Interrupt Data:', data.data
        self.interrupted = True

        print "Stopping Motion..."
        self.setStop() #Stops Current Motion
        try:
            self.setStopRight() #Sends message to service node
        except:
            print "Couldn't stop right arm! "

        rospy.sleep(0.1)
        
        posStopL = Point()
        quatStopL = Quaternion()

        # TODO: location should be replaced into the last scooping or feeding starts.
        print "Moving left arm to safe position "
        if data.data == 'InterruptHead':
            self.feeding([0])
            self.setPostureGoal(self.leftArmInitialJointAnglesFeeding, 10)
        else:
            self.scooping([0])
            self.setPostureGoal(self.leftArmInitialJointAnglesScooping, 10)

        
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

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    haptic_mpc_util.initialiseOptParser(p)
    opt = haptic_mpc_util.getValidInput(p)

    # Initial variables
    d_robot    = 'pr2'
    controller = 'static'
    #controller = 'actionlib'
    arm        = 'l'

    # Manual anomaly by a robot
    head_noise = False
    
    rospy.init_node('arm_reacher_feeding')
    ara = armReachAction(d_robot, controller, arm, head_noise=head_noise)
    rospy.spin()


