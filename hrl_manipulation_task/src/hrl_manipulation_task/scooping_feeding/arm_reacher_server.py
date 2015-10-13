#!/usr/bin/env python

# System
import sys, time, copy
import rospy
import numpy as np

# ROS
import roslib
roslib.load_manifest('hrl_manipulation_task')
import tf
import PyKDL
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import String

# HRL library
import hrl_haptic_mpc.haptic_mpc_util as haptic_mpc_util
import hrl_lib.quaternion as quatMath 
from hrl_srvs.srv import None_Bool, None_BoolResponse, Int_Int, String_String

# Personal library
from sandbox_dpark_darpa_m3.lib.hrl_mpc_base import mpcBaseAction


class armReachAction(mpcBaseAction):
    def __init__(self, d_robot, controller, arm, verbose=False):
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
                if verbose:
                    print "--------------------------------"
                    print "Current "+self.arm+" arm joint angles"
                    print self.getJointAngles()
                    print "Current "+self.arm+" arm pose"
                    print self.getEndeffectorPose()
                    print "Current "+self.arm+" arm orientation (w/ euler rpy)"
                    print self.getEndeffectorRPY()*180.0/np.pi
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

        #TOOL: Set a tool frame for MOVET. Defualt is 0 which is end-effector frame.

        joint or pose: we use radian and meter unit. The order of euler angle follows z-y-x order.
        timeout or duration: we use second
        relative_frame: You can put your custome PyKDL frame variable or you can use 'self.default_frame'
        '''
        
        self.motions = {}

        ## test motoins --------------------------------------------------------
        self.motions['test_orient'] = {}
        self.motions['test_orient']['left'] = \
          [['MOVEJ', '[1.570, 0, 1.570, -1.570, -4.71, 0, -1.570]', 10.0],\
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


        self.motions['test_debug'] = {}
        self.motions['test_debug']['left'] = \
        [#['MOVEJ', '[1.738, 0.201, 1.385, -2.035, -3.850, -0.409, -2.292]', 30.0],\
           ['MOVES', '[0.4, 0, 0, 0, 0, 0.5]', 20., 'self.bowl_frame']]
        ##    ['MOVES', '[0.489, 0.706, 0.122, 1.577, -0.03, -0.047]', 10., 'self.default_frame']]
        self.motions['test_debug']['right'] =\
          [['MOVEJ', '[-1.570, 0, -1.570, -1.570, -4.71, 0, -1.570]', 5.0] ]

        
        ## Testing Motions ---------------------------------------------------------
        # Used to test and find the best optimal procedure to scoop the target.
        self.motions['testingMotion'] = {}
        self.motions['testingMotion']['left'] = \
          [['MOVEJ', '[0.051, 0.219, 0.135, -2.115, -3.052, -1.928, -1.64]', 5.0],\
          ['MOVEJ', '[0.054, 0.038, 0.298, -2.118, -3.090, -1.872, -1.39]', 10.0],\
          ['MOVEJ', '[0.645, 0.016, 0.279, -2.118, -3.127, -1.803, -2.176]', 10.0],\
          ['MOVEJ', '[0.051, 0.219, 0.135, -2.115, -3.053, -1.928, -1.64]', 10.0]]

        #  ['MOVES', '[ 0.521, -0.137, -0.041, 38, -99, -4]', 5.0]]
        self.motions['testingMotion']['right'] = []
        ##  [['MOVEJ', '[1.570, 0, 1.570, -1.570, -4.71, 0, -1.570]', 5.0]]
        ## Scooping motoins --------------------------------------------------------
        # Used to perform motions relative to bowl/mouth positions > It should use relative frame                        
        self.motions['initScooping'] = {}
        self.motions['initScooping']['left'] = \
          [['MOVEJ', '[0.677, 0.147, 0.625, -2.019, -3.544, -1.737, -2.077]', 10.0] ] 
        self.motions['initScooping']['right'] = \
          [['MOVEJ', '[-0.38766331112969, 0.10069929321857, -1.1663864746779473, -1.3783351499208258, -6.952309470970393, -0.7927793084206439, 2.179890685990797]', 10.0] ]
          
        self.motions['runScooping'] = {}
        self.motions['runScooping']['left'] = \
          [['MOVEJ', '[0.5778482746916489, -0.33053534611272795, 0.9224823852146335, -1.6170620529922632, -3.868802912203407, -1.4730600051592582, -1.234066882640653]', 10.0],\
           ['MOVEJ', '[0.5175751638731203, -0.17877205406333624, 0.8937788712944239, -1.8017894807176447, -3.6269442416631135, -2.011787453832436, -1.8558093508638567]', 10.0],\
           ['MOVEJ', '[0.5778482746916489, -0.33053534611272795, 0.9224823852146335, -1.6170620529922632, -3.868802912203407, -1.4730600051592582, -1.234066882640653]', 10.0]
        #  [['MOVES', '[0.62, 0.15, 0.1, 1.161, 1.238, -2.103]', 10., 'self.default_frame'],
        #   ['MOVES', '[0.62, 0.15, 0.1, 1.161, 1.238, -1.103]', 10., 'self.default_frame'], 
        #   ['MOVES', '[0.62, 0.15, 0.1, 1.161, 1.238, -2.103]', 10., 'self.default_frame'], 
                    
          ## [['MOVES', '[0.588, 0.158, 0.092, 46.8, 66., 48.92]', 10., 'self.default_frame'], 
          ##  ['MOVES', '[0.588, 0.158, 0.0, 46.8, 66., 48.92]', 10., 'self.default_frame'], 
          ##  ['MOVES', '[0.588, 0.158, 0.092, 46.8, 66., 48.92]', 10., 'self.default_frame']
           
           ## ['MOVES', '[-.015+self.bowl_frame.p.x(), -0.02+self.bowl_frame.p.y(),  .15+self.bowl_frame.p.z(),  90, -50, -30]', 6, 'self.default_frame'], 
           ## ['MOVES', '[-.015+self.bowl_frame.p.x(), -0.02+self.bowl_frame.p.y(), -.055+self.bowl_frame.p.z(), 90, -50,	-30]', 3, 'self.default_frame'], #Moving down into bowl
           ## ['MOVES', '[  .02+self.bowl_frame.p.x(), -0.02+self.bowl_frame.p.y(), -.025+self.bowl_frame.p.z(), 90, -30,	-30]', 3, 'self.default_frame'], #Moving forward in bowl
           ## ['MOVES', '[    0+self.bowl_frame.p.x(), -0.03+self.bowl_frame.p.y(),   .20+self.bowl_frame.p.z(), 90,   0,	-30]', 2, 'self.default_frame'], #While rotating spoon to scoop out
           ## ['MOVES', '[    0+self.bowl_frame.p.x(), -0.03+self.bowl_frame.p.y(),   .25+self.bowl_frame.p.z(), 90,   0,	-30]', 2, 'self.default_frame']  #Moving up out of bowl
           ]
        self.motions['runScooping']['right'] = \
          []
        
        ## Feeding motoins --------------------------------------------------------
        # It uses the l_gripper_spoon_frame aligned with mouth
        self.motions['initFeeding'] = {}
        self.motions['initFeeding']['left'] = \
          [['MOVEJ', '[0.39, 0.26, 0.61, -2.07, -3.36, -1.82, -2.33]', 10.0]] 
        self.motions['initFeeding']['right'] = \
          []
          
        self.motions['runFeeding'] = {}
        self.motions['runFeeding']['left'] = \
          [['MOVEJ', '[1.570, 0, 1.570, -1.570, -4.71, 0, -1.570]', 5.0],
           ['MOVET', '[0.0, 0., 0., 90, 0, 0]', 10.], 
           ['MOVET', '[0.0, 0., 0., -90, 0, 0]', 10.]
           ## ['PAUSE', 0.5],
           ## ['MOVES', '[0.79, 0.35, 0.05, -59., 0.4, 77.]', 10., 'self.default_frame'], 
           ## ['MOVES', [-.015, 0.0, 0.0, 90, 0, -75], 3, 'self.mouth_frame'], 
           ## ['MOVES', [-.07,  0.0, 0.0, 90, 0, -75], 3, 'self.mouth_frame'], 
           ## ['MOVES', [-.015, 0.0, 0.0, 90, 0, -75], 3, 'self.mouth_frame'], 
           ]
        self.motions['runFeeding']['right'] = \
          []
                                                    
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
                # 1. right arm ('r_gripper_tool_frame') from tf
                self.tf_lstnr.waitForTransform(self.torso_frame, 'r_gripper_tool_frame', rospy.Time(0), \
                                               rospy.Duration(5.0))
                [self.----pos, self.---quat] = \
                    self.tf_lstnr.lookupTransform(self.torso_frame, 'r_gripper_tool_frame', rospy.Time(0))

                
                # 2. add offset 
                # 3. Store in bowl_frame
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
            print "Exec: ", motion
            pos  = Point()
            quat = Quaternion()
        
            if motion[0] == 'MOVEP':   
                poseData  = eval(motion[1])            
                if len(motion)>3: frameData  = eval(motion[3])   
                else: frameData = PyKDL.Frame()         

                poseFrame = array2KDLframe(poseData)
                poseFrame = frameConversion(poseFrame, frameData)
                    
                pos.x = poseFrame.p[0]
                pos.y = poseFrame.p[1]
                pos.z = poseFrame.p[2]
                self.setPositionGoal(pos, quat, motion[2])
                
            elif motion[0] == 'MOVES':
                poseData  = eval(motion[1])            
                if len(motion)>3: frameData  = eval(motion[3])            
                else: frameData = PyKDL.Frame()         

                poseFrame = array2KDLframe(poseData)
                poseFrame = frameConversion(poseFrame, frameData)
                    
                pos.x = poseFrame.p[0]
                pos.y = poseFrame.p[1]
                pos.z = poseFrame.p[2]

                quat.x = poseFrame.M.GetQuaternion()[0]
                quat.y = poseFrame.M.GetQuaternion()[1]
                quat.z = poseFrame.M.GetQuaternion()[2]
                quat.w = poseFrame.M.GetQuaternion()[3]
                
                self.setOrientGoal(pos, quat, motion[2])

            elif motion[0] == 'MOVET':
                poseData  = eval(motion[1])            

                [cur_pos, cur_quat] = self.getEndeffectorPose()
                M = PyKDL.Rotation.Quaternion(cur_quat[0], cur_quat[1], cur_quat[2], cur_quat[3]) # R_0e

                # position 
                pos_offset = PyKDL.Vector(poseData[0], poseData[1], poseData[2])
                pos_offset = M * pos_offset

                pos.x = cur_pos[0] + pos_offset[0]
                pos.y = cur_pos[1] + pos_offset[1]
                pos.z = cur_pos[2] + pos_offset[2]

                # orientation
                M.DoRotX(poseData[3])
                M.DoRotY(poseData[4])
                M.DoRotZ(poseData[5])
                rot_offset = M
                ## rot_offset = PyKDL.Rotation.RPY(poseData[3], poseData[4], poseData[5]) #R_ee'
                ## rot_offset = M * rot_offset
                quat.x = rot_offset.GetQuaternion()[0]
                quat.y = rot_offset.GetQuaternion()[1]
                quat.z = rot_offset.GetQuaternion()[2]
                quat.w = rot_offset.GetQuaternion()[3]

                self.setOrientGoal(pos, quat, motion[2])                
                
            elif motion[0] == 'MOVEJ': 
                self.setPostureGoal(eval(motion[1]), motion[2])
                
            elif motion[0] == 'PAUSE': 
                rospy.sleep(motion[1])
                print "Pausing for ", str(motion[1]), " seconds "

                
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


    ## def tfBroadcaster(self, ps):

    ##     quat = tf.transformations.quaternion_matrix([ps.pose.orientation.x,
    ##                                                  ps.pose.orientation.y,
    ##                                                  ps.pose.orientation.z,
    ##                                                  ps.pose.orientation.w])

    ##     self.br = tf.TransformBroadcaster()
    ##     self.br.sendTransform((ps.pose.position.x, ps.pose.position.y, ps.pose.position.z),
    ##                           quat,
    ##                           rospy.Time.now(),
    ##                           "goal_viz",
    ##                           "torso_lift_link")
        

def frameConversion(cur_pose, cur_frame):

    pos = cur_frame * cur_pose.p
    rot = cur_frame.M * cur_pose.M
    pose = PyKDL.Frame(rot, pos)
    
    return pose

def array2KDLframe(pose_array):

    p = PyKDL.Vector(pose_array[0], pose_array[1], pose_array[2])
    M = PyKDL.Rotation.RPY(pose_array[3], pose_array[4], pose_array[5])

    return PyKDL.Frame(M,p)

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
    if opt.arm == 'l': verbose = True
    else: verbose = False
        

    rospy.init_node('arm_reacher_feeding_and_scooping')
    ara = armReachAction(d_robot, controller, arm, verbose)
    rospy.spin()


