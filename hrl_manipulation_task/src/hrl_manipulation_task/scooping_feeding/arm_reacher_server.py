#!/usr/bin/env python

# System
import sys, time, copy
import rospy
import numpy as np

# ROS
import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf
import PyKDL
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import String

# HRL library
import hrl_haptic_mpc.haptic_mpc_util as haptic_mpc_util
import hrl_lib.quaternion as quatMath 
from hrl_srvs.srv import None_Bool, None_BoolResponse, Int_Int
from hrl_multimodal_anomaly_detection.srv import PosQuatTimeoutSrv, AnglesTimeoutSrv, String_String

# Personal library
from sandbox_dpark_darpa_m3.lib.hrl_mpc_base import mpcBaseAction


class armReachAction(mpcBaseAction):
    def __init__(self, d_robot, controller, arm):
        mpcBaseAction.__init__(self, d_robot, controller, arm)

        #Variables...! #
        if arm == 'l':  self.arm = 'left'
        else:  self.arm = 'right'
        self.interrupted = False

        self.bowl_frame_kinect  = None
        self.mouth_frame_kinect = None
        self.default_frame    = PyKDL.Frame()

        self.initCommsForArmReach()                            
        self.initParamsForArmReach()

        rate = rospy.Rate(100) # 25Hz, nominally.
        while not rospy.is_shutdown():
            if self.getJointAngles() != []:
                print "--------------------------------"
                print "Current "+self.arm+" arm joint angles"
                print self.getJointAngles()
                print "Current "+self.arm+" arm pose"
                print self.getEndeffectorPose()
                print "--------------------------------"
                break
            rate.sleep()
            
        rospy.loginfo("Arm Reach Action is initialized.")
                            
    def initCommsForArmReach(self):

        # publishers and subscribers
        rospy.Subscriber('InterruptAction', String, self.stopCallback)
        rospy.Subscriber('/ar_track_alvar/bowl_cen_pose',
                         PoseStamped, self.bowlPoseCallback)
        rospy.Subscriber('/ar_track_alvar/mouth_pose',
                         PoseStamped, self.mouthPoseCallback)
        
        # service
        self.reach_service = rospy.Service('arm_reach_enable', String_String, self.serverCallback)
        ## self.scoopingStepsClient = rospy.ServiceProxy('/scooping_steps_service', None_Bool)

        rospy.loginfo("ROS-based communications are set up .")
                                    
    def initParamsForArmReach(self):
        '''
        Industrial movment commands generally follows following format, 
        
               Movement type, joint or pose(pos+euler), timeout, relative_frame(not implemented)

        In this code, we allow to use following movement types,

        MOVEP: straight motion without orientation control (ex. MOVEP pos-euler timeout relative_frame)
        MOVES: straight motion with orientation control (ex. MOVES pos-euler timeout relative_frame)
        MOVET: MOVES with respect to the current tool frame (ex. MOVET pos-euler timeout) (experimental!!)
        MOVEJ: joint motion (ex. MOVEJ joint timeout)
        PAUSE: Add pause time between motions (ex. PAUSE duration)

        joint or pose: we use degree and meter unit. The order of euler angle follows z-y-x order.
        timeout or duration: we use second
        relative_frame: Not implemented. You can put your custome PyKDL frame variable or 'self.default_frame'
        '''
        
        self.motions = {}

        ## test motoins --------------------------------------------------------
        self.motions['test_orient'] = {}
        self.motions['test_orient']['left'] = \
          [['MOVEJ', '[1.570, 0, 1.570, -1.570, -4.71, 0, -1.570]', 5.0],\
          ['MOVET', '[ 0, 0, 0, 1.0, 0, 0]', 5.0],\
          ['MOVET', '[ 0, 0, 0, -1.0, 0, 0]', 5.0],\
          ['MOVET', '[ 0, 0, 0, 0, 0.5, 0]', 5.0],\
          ['MOVET', '[ 0, 0, 0, 0, -0.5, 0]', 5.0]]
        self.motions['test_orient']['right'] =\
          [['MOVEJ', '[-1.570, 0, -1.570, -1.570, -4.71, 0, -1.570]', 5.0] ]

        self.motions['test_pos'] = {}
        self.motions['test_pos']['left'] = \
          [['MOVEJ', '[1.570, 0, 1.570, -1.570, -4.71, 0, -1.570]', 5.0],\
          ['MOVET', '[ 0, 0, 0.3, 0, 0, 0]', 5.0],\
          ['MOVET', '[ 0, 0, -0.3, 0, 0, 0]', 5.0],\
          ['MOVET', '[ 0, 0.3, 0, 0, 0, 0]', 5.0],\
          ['MOVET', '[ 0, -0.3, 0, 0, 0, 0]', 5.0]]
        self.motions['test_pos']['right'] = []
        
        ## Scooping motoins --------------------------------------------------------
        # Used to perform motions relative to bowl/mouth positions > It should use relative frame                        
        self.motions['initScooping'] = {}
        self.motions['initScooping']['left'] = \
          [['MOVEJ', '[1.570,     0, 1.570, -1.570, -4.71, 0, -1.570]', 10.0] ] 
        self.motions['initScooping']['right'] = \
          [['MOVEJ', '[1.570,     0, 1.570, -1.570, -4.71, 0, -1.570]', 10.0] ]
          
        self.motions['runScooping'] = {}
        self.motions['runScooping']['left'] = \
          [['MOVEJ', [1.570,     0, 1.570, -1.570, -4.71, 0, -1.570], 10.0],\
           ['MOVES', '[-.015+self.bowl_frame.p.x(), -0.02+self.bowl_frame.p.y(),  .15+self.bowl_frame.p.z(),  90, -50, -30]', 6, 'self.default_frame'], 
           ['MOVES', '[-.015+self.bowl_frame.p.x(), -0.02+self.bowl_frame.p.y(), -.055+self.bowl_frame.p.z(), 90, -50,	-30]', 3, 'self.default_frame'], #Moving down into bowl
           ['MOVES', '[  .02+self.bowl_frame.p.x(), -0.02+self.bowl_frame.p.y(), -.025+self.bowl_frame.p.z(), 90, -30,	-30]', 3, 'self.default_frame'], #Moving forward in bowl
           ['MOVES', '[    0+self.bowl_frame.p.x(), -0.03+self.bowl_frame.p.y(),   .20+self.bowl_frame.p.z(), 90,   0,	-30]', 2, 'self.default_frame'], #While rotating spoon to scoop out
           ['MOVES', '[    0+self.bowl_frame.p.x(), -0.03+self.bowl_frame.p.y(),   .25+self.bowl_frame.p.z(), 90,   0,	-30]', 2, 'self.default_frame']  #Moving up out of bowl
           ]
        self.motions['runScooping']['right'] = \
          []
        
        ## Feeding motoins --------------------------------------------------------
        # It uses the l_gripper_spoon_frame aligned with mouth
        ## self.motions['initFeeding'] = {}
        ## self.motions['initFeeding']['left'] = \
        ##   [['MOVEJ', [0.397, 0.272, 1.088, -2.11, -3.78, -0.658, -2.288], 10.0]] 
        ## self.motions['initFeeding']['right'] = \
        ##   []
          
        ## self.motions['runFeeding'] = {}
        ## self.motions['runFeeding']['left'] = \        
        ##   [['MOVEJ', [0.397, 0.272, 1.088, -2.11, -3.78, -0.658, -2.288], 10.0],
        ##    ['MOVES', [-.015, 0.0, 0.0, 90, 0, -75], 3, 'self.mouth_frame'], 
        ##    ['MOVES', [-.07,  0.0, 0.0, 90, 0, -75], 3, 'self.mouth_frame'], 
        ##    ['PAUSE', 0.5],
        ##    ['MOVES', [-.015, 0.0, 0.0, 90, 0, -75], 3, 'self.mouth_frame'], 
        ##    ]
        ## self.motions['runFeeding']['right'] = \
        ##   []
                                                    
        rospy.loginfo("Parameters are loaded.")
                
        
    def serverCallback(self, req):
        req = req.data
        self.interrupted = False

        if req == "getBowlPos":
            if self.bowl_frame_kinect is not None:
                self.bowl_frame = self.bowl_frame_kinect
                return "Chose kinect bowl position"
            elif self.bowl_frame_kinect is None:
                # Get frame info from right arm and upate bowl_pos                
                self.bowl_frame = PyKDL.Frame()  # TODO: Need to update!!!
                return "Chose bowl position from kinematics using tf"                
            else:
                return "No kinect head position available! \n Code won't work! \n \
                Provide head position and try again!"
        elif req == "getHeadPos":
            if self.mouth_frame_kinect is not None:
                self.mouth_frame = self.mouth_frame_kinect
                return "Chose kinect head position"
            else:
                return "No kinect head position available! \n Code won't work! \n \
                Provide head position and try again!"
        else:
            self.parsingMovements(self.motions[req][self.arm])
            return "Completed to execute "+req 

        ## else:
        ##     return "Request not understood by server!!!"

        
    def parsingMovements(self, motions):
        
        for i, motion in enumerate(motions):
            pos  = Point()
            quat = Quaternion()
        
            if motion[0] == 'MOVEP':   
                poseData  = eval(motion[1])            
                if len(motion)>3: frameData  = eval(motion[3])            
                pos.x = poseData[0]
                pos.y = poseData[1]
                pos.z = poseData[2]
                self.setPositionGoal(pos, quat, motion[2])
                
            elif motion[0] == 'MOVES':
                poseData  = eval(motion[1])            
                if len(motion)>3: frameData  = eval(motion[3])            
                pos.x = poseData[0]
                pos.y = poseData[1]
                pos.z = poseData[2]

                quatArray = quatMath.euler2quat(poseData[3], poseData[4], poseData[5]) 
                quat.w, quat.x, quat.y, quat.z = (quatArray[0], quatArray[1],\
                                                  quatArray[2], quatArray[3])
                self.setOrientGoal(pos, quat, motion[2])

            elif motion[0] == 'MOVET':
                poseData  = eval(motion[1])            

                [cur_pos, cur_quat] = self.getEndeffectorPose()
                M = PyKDL.Rotation.Quaternion(cur_quat[0], cur_quat[1], cur_quat[2], cur_quat[3])

                # position 
                pos_offset = PyKDL.Vector(poseData[0], poseData[1], poseData[2])
                pos_offset = M * pos_offset

                pos.x = cur_pos[0] + pos_offset[0]
                pos.y = cur_pos[1] + pos_offset[1]
                pos.z = cur_pos[2] + pos_offset[2]

                # orientation
                rot_offset = PyKDL.Rotation.EulerZYX(poseData[3], poseData[4], poseData[5])
                rot_offset = M * rot_offset
                quat.x = rot_offset.GetQuaternion()[0]
                quat.y = rot_offset.GetQuaternion()[1]
                quat.z = rot_offset.GetQuaternion()[2]
                quat.w = rot_offset.GetQuaternion()[3]
                                
                self.setOrientGoal(pos, quat, motion[2])                
                
            elif motion[0] == 'MOVEJ': 
                self.setPostureGoal(eval(motion[1]), motion[2])
                
            elif motion[0] == 'PAUSE': 
                rospy.sleep(motions[1])
                print "Pausing for ", motions[1], " seconds "

                
            if self.interrupted:
                break
                

    def bowlPoseCallback(self, data):
        p = PyKDL.Vector(data.pose.position.x, data.pose.position.y, data.pose.position.z)
        M = PyKDL.Rotation.Quaternion(data.pose.orientation.x, data.pose.orientation.y, 
                                      data.pose.orientation.z, data.pose.orientation.w)
        self.bowl_frame_kinect = PyKDL.Frame(M,p)
            
    def mouthPoseCallback(self, data):

        p = PyKDL.Vector(data.pose.position.x, data.pose.position.y, data.pose.position.z)
        M = PyKDL.Rotation.Quaternion(data.pose.orientation.x, data.pose.orientation.y, 
                                      data.pose.orientation.z, data.pose.orientation.w)

        spoon_x = -M.UnitZ()
        spoon_y = PyKDL.Vector(0, 0, 1.0)
        spoon_z = spoon_x * spoon_y
        spoon_y = spoon_z * spoon_x
        spoon_rot = PyKDL.Rotation(spoon_x, spoon_y, spoon_z)

        self.mouth_frame_kinect = PyKDL.Frame(spoon_rot,p)
        

    def stopCallback(self, msg):
        print '\n\nAction Interrupted! Event Stop\n\n'
        print 'Interrupt Data:', msg.data
        self.interrupted = True

        print "Stopping Motion..."
        self.setStop() #Stops Current Motion
        try:
            self.setStopRight() #Sends message to service node
        except:
            print "Couldn't stop right arm! "

        ## posStopL = Point()
        ## quatStopL = Quaternion()

        ## # TODO: location should be replaced into the last scooping or feeding starts.
        ## print "Moving left arm to safe position "
        ## if data.data == 'InterruptHead':
        ##     self.feeding([0])
        ##     self.setPostureGoal(self.lInitAngFeeding, 10)
        ## else:
        ##     self.scooping([0])
        ##     self.setPostureGoal(self.lInitAngScooping, 10)

        

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    haptic_mpc_util.initialiseOptParser(p)
    opt = haptic_mpc_util.getValidInput(p)

    # Initial variables
    d_robot    = 'pr2'
    controller = 'static'
    #controller = 'actionlib'
    arm        = opt.arm

    rospy.init_node('arm_reacher_feeding_and_scooping')
    ara = armReachAction(d_robot, controller, arm)
    rospy.spin()


