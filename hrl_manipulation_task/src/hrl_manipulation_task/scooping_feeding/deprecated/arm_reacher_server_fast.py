#!/usr/bin/env python
#
# Copyright (c) 2014, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

# System
import sys, time, copy, random
import numpy as np

# ROS
import rospy
import PyKDL
import actionlib

# Msg
from geometry_msgs.msg import Pose, PoseStamped, Point, PointStamped, Quaternion
from std_msgs.msg import String, Empty, Int64, String
import pr2_controllers_msgs.msg
from hrl_msgs.msg import FloatArray
from hrl_srvs.srv import None_Bool, None_BoolResponse, String_String

# HRL library
import hrl_haptic_mpc.haptic_mpc_util as haptic_mpc_util
## import hrl_lib.quaternion as quatMath

# Personal library - need to move neccessary libraries to a new package
from sandbox_dpark_darpa_m3.lib.hrl_mpc_base import mpcBaseAction
import sandbox_dpark_darpa_m3.lib.hrl_dh_lib as dh
QUEUE_SIZE = 10


class armReachAction(mpcBaseAction):
    def __init__(self, d_robot, controller, arm, viz=False, verbose=False):
        mpcBaseAction.__init__(self, d_robot, controller, arm)

        #Variables...! #
        self.stop_motion = False
        self.verbose = verbose
        self.highBowlDiff = np.array([0, 0, 0])
        self.bowlPosition = np.array([0, 0, 0])
        # vertical x side x depth
        self.mouthManOffset = np.array([-0.03, 0.0, 0.04]) # -0.03, 0., 0.05
        self.mouthNoise     = np.array([0., 0., 0.])
        self.mouthOffset    = self.mouthManOffset+self.mouthNoise 

        self.viz = viz
        self.bowl_frame_kinect  = None
        self.mouth_frame_vision = None
        self.bowl_frame         = None
        self.mouth_frame        = None
        self.default_frame      = PyKDL.Frame()

        self.initCommsForArmReach()
        self.initParamsForArmReach()
        self.setMotions()


        if self.arm_name == 'left':
            self.feeding_depth_pub.publish( Int64(int(self.mouthManOffset[2]*100.0)) )
            self.feeding_horiz_pub.publish( Int64(int(-self.mouthManOffset[1]*100.0)) )            
            self.feeding_vert_pub.publish( Int64(int(self.mouthManOffset[0]*100.0)) )

        rate = rospy.Rate(5)
        print_flag = True
        while not rospy.is_shutdown():
            if self.getJointAngles() != []:
                if verbose and print_flag:
                    print "--------------------------------"
                    print "Current "+self.arm_name+" arm joint angles"
                    print self.getJointAngles()
                    print "Current "+self.arm_name+" arm pose"
                    print self.getEndeffectorPose(tool=self.cur_tool)
                    print "Current "+self.arm_name+" arm orientation (w/ euler rpy)"
                    print self.getEndeffectorRPY(tool=self.cur_tool) #*180.0/np.pi
                    print "--------------------------------"
                    print_flag = False
                self.pubCurEEPose()
                ## break

            # temp
            ## if self.arm_name == 'left':
            ##     self.getBowlFrame()                
                
            rate.sleep()

        rospy.loginfo("Arm Reach Action is initialized.")

    def initCommsForArmReach(self):

        # publishers
        self.bowl_pub\
        = rospy.Publisher('/hrl_manipulation_task/arm_reacher/bowl_cen_pose', PoseStamped,\
                          queue_size=QUEUE_SIZE, latch=True)
        self.mouth_pub\
        = rospy.Publisher('/hrl_manipulation_task/arm_reacher/mouth_pose', PoseStamped,\
                          queue_size=QUEUE_SIZE, latch=True)
        self.ee_pose_pub\
        = rospy.Publisher('/hrl_manipulation_task/arm_reacher/'+self.arm_name+\
                          '_ee_pose', PoseStamped, queue_size=QUEUE_SIZE, latch=True)
            
        self.bowl_height_init_pub\
        = rospy.Publisher('/hrl_manipulation_task/arm_reacher/init_bowl_height', Empty,\
                          queue_size=QUEUE_SIZE, latch=True)
        self.kinect_pause\
        = rospy.Publisher('/head_mount_kinect/pause_kinect', String,\
                          queue_size=QUEUE_SIZE, latch=False)

        if self.arm_name == 'left':
            self.feeding_depth_pub\
            = rospy.Publisher('/feeding/manipulation_task/mouth_depth_offset',\
                              Int64, queue_size=QUEUE_SIZE, latch=True)
            self.feeding_horiz_pub\
            = rospy.Publisher('/feeding/manipulation_task/mouth_horiz_offset',\
                              Int64, queue_size=QUEUE_SIZE, latch=True)
            self.feeding_vert_pub\
            = rospy.Publisher('/feeding/manipulation_task/mouth_vert_offset',\
                              Int64, queue_size=QUEUE_SIZE, latch=True)

        # subscribers
        rospy.Subscriber('/manipulation_task/InterruptAction', String, self.stopCallback)
        rospy.Subscriber('/hrl_manipulation_task/bowl_highest_point', Point,\
                         self.highestBowlPointCallback)
        rospy.Subscriber('/hrl_manipulation_task/mouth_pose',
                         PoseStamped, self.mouthPoseCallback)
        rospy.Subscriber('/hrl_manipulation_task/mouth_noise', FloatArray, self.mouthNoiseCallback)
        if self.arm_name == 'left':
            rospy.Subscriber('/feeding/manipulation_task/mouth_depth_request',\
                             Int64, self.feedingDepthCallback)
            rospy.Subscriber('/feeding/manipulation_task/mouth_horiz_request',\
                             Int64, self.feedingHorizCallback)
            rospy.Subscriber('/feeding/manipulation_task/mouth_vert_request',\
                             Int64, self.feedingVertCallback)

        # service
        self.reach_service = rospy.Service('arm_reach_enable', String_String, self.serverCallback)

        if self.verbose: rospy.loginfo("ROS-based communications are set up .")

    def initParamsForArmReach(self):

        ## Off set : 11 cm x direction, - 4 cm z direction.
        ## self.bowl_pos_offset    = rospy.get_param('/hrl_manipulation_task/sub_ee_pos_offset')
        ## self.bowl_orient_offset = rospy.get_param('/hrl_manipulation_task/sub_ee_orient_offset')
        self.org_tool = copy.copy(self.cur_tool)

        if self.arm_name == 'left':
            r_tool_id = rospy.get_param('/right/haptic_mpc'+self.robot_path+'/tool_id', 0)        
            pos = rospy.get_param('right/haptic_mpc'+self.robot_path+'/tool_frame_'+str(r_tool_id)+'/pos', None)
            rpy = rospy.get_param('right/haptic_mpc'+self.robot_path+'/tool_frame_'+str(r_tool_id)+'/rpy', None)
            p = PyKDL.Vector(pos['x'], pos['y'], pos['z'])
            M = PyKDL.Rotation.RPY(rpy['rx'], rpy['ry'], rpy['rz'])
            self.r_tool_offset = PyKDL.Frame(M,p)
             
        pass

    def setMotions(self):
        '''
        Industrial movment commands generally follows following format,

               Movement type, joint or pose(pos+euler), timeout, relative_frame, threshold

        In this code, we allow to use following movement types,

        MOVEP: straight motion without orientation control (ex. MOVEP pos-euler timeout relative_frame)
        MOVES: straight motion with orientation control (ex. MOVES pos-euler timeout relative_frame)
        MOVEL: straight (linear) motion with orientation control (ex. MOVEL pos-euler timeout relative_frame)
        MOVET: MOVES with respect to the current tool frame (ex. MOVET pos-euler timeout) (experimental!!)
        MOVEJ: joint motion (ex. MOVEJ joint timeout)
        PAUSE: Add pause time between motions (ex. PAUSE duration)
        STOP:  Stop at the current pose (ex. STOP)
        TOOL:  Set a tool frame. Defualt is 0 which is end-effector frame.

        joint or pose: we use radian and meter unit. The order of euler angle follows original x-y-z order (RPY).
        timeout or duration: we use second
        relative_frame: You can put your custome PyKDL frame variable or you can use 'self.default_frame'
        '''

        self.motions = {}

        ## Init arms ---------------------------------------------------------------
        self.motions['initArms'] = {}
        self.motions['initArms']['left']  = [['MOVEJ', '[0.6447, 0.1256, 0.721, -2.12, 1.574, -0.7956, 1.1291]',\
                                              10.0]]
        self.motions['initArms']['right'] = [['MOVEJ', '[-0.59, 0.0, -1.574, -1.041, 0.0, -1.136, -1.65]', 10.0]]


        ## Stabbing motoins --------------------------------------------------------
        # Used to perform motions relative to bowl/mouth positions > It should use relative frame
        # [shoulder (towards left shoulder), arm pitch on shoulder (towards ground),
        # whole arm roll (rotates right), elbow pitch (rotates towards outer arm),
        # elbow roll (rotates left), wrist pitch (towards top of forearm), wrist roll (rotates right)]
        # (represents positive values)
        self.motions['initStabbing1'] = {}
        self.motions['initStabbing1']['left'] \
          = [['PAUSE', 2.0],
             ['MOVEJ', '[0.6447, 0.1256, 0.721, -2.12, 1.574, -0.7956, 1.1291]', 5.0]]
        self.motions['initStabbing1']['right'] = \
          [['MOVEJ', '[-0.59, 0.131, -1.55, -1.041, 0.098, -1.136, -1.4]', 3.0*2],
           ['TOOL', 0],
           ['MOVES', '[0.7, -0.15, -0., -3.1415, 0.0, 1.574]', 5.],
           ['TOOL', self.org_tool]
          ]
           ## ['MOVES', '[0.7, -0.15, -0., -3.1415, 0.0, 1.574]', 5.]]

        self.motions['initStabbing2'] = {}
        self.motions['initStabbing2']['left'] = [
            #['MOVES', '[0.7, -0.15, -0., -3.1415, 0.0, 1.574]', 3.],
            ['MOVES', '[-0.02+self.highBowlDiff[0], 0.01-self.highBowlDiff[1], -0.08, 0, -0.1, 0]', 5*1.5, 'self.bowl_frame']]
        self.motions['initStabbing2']['right'] = []

        self.motions['initStabbing12'] = {}
        self.motions['initStabbing12']['left'] = \
          [['PAUSE', 1.0],
           ['MOVES', '[-0.02, 0.01, -0.08, 0, -0.1, 0]', 7, 'self.bowl_frame']]
        self.motions['initStabbing12']['right'] = \
          self.motions['initStabbing1']['right'][1:]

        # [Y (from center of bowl away from Pr2), X (towards right gripper), Z (towards floor) ,
        # roll?, pitch (tilt downwards), yaw?]
        self.motions['runStabbing'] = {}
        self.motions['runStabbing']['left'] = \
          [['MOVES', '[-0.02+self.highBowlDiff[0], 0.01-self.highBowlDiff[1], 0.03, 0, -0.1, 0]', 5, 'self.bowl_frame'],
           ['STOP'],
           ['MOVET', '[0.0,0.0,-0.15,0,0.0,0]',5]]
        
           #['MOVES', '[-0.02+self.highBowlDiff[0], 0.0-self.highBowlDiff[1], -0.1, 0, -0.1, 0]', 5, 'self.bowl_frame']]
           
        
           ##['MOVES', '[-0.03+self.highBowlDiff[0]+random.choice([-0.01, 0, 0.01]), 0.0-self.highBowlDiff[1]++random.choice([-0.015, 0, 0.015]), 0.04, 0, 0., 0]', 5, 'self.bowl_frame'],
           ## ['MOVES', '[-0.03+self.highBowlDiff[0], 0.0-self.highBowlDiff[1],  0.0, 0, 0., 0]', 2,
           ##  'self.bowl_frame'],
           ## ['MOVES', '[-0.03+self.highBowlDiff[0], 0.0-self.highBowlDiff[1], -0.15, 0, 0., 0]', 3,
           ##  'self.bowl_frame'],]
        self.motions['runStabbing']['right'] = \
          [['PAUSE', 5.0],
           ['STOP'],
           ['PAUSE', 3.0]] + self.motions['initStabbing1']['right'][1:] 


        ## Scooping motoins --------------------------------------------------------
        # Used to perform motions relative to bowl/mouth positions > It should use relative frame
        # [shoulder (towards left shoulder), arm pitch on shoulder (towards ground),
        # whole arm roll (rotates right), elbow pitch (rotates towards outer arm),
        # elbow roll (rotates left), wrist pitch (towards top of forearm), wrist roll (rotates right)] (represents positive values)
        self.motions['initScooping1'] = {}
        self.motions['initScooping1']['left'] =\
        [['MOVEJ', '[0.6447, 0.1256, 0.721, -2.12, 1.574, -0.7956, 0.4291]', 5.0]]
        self.motions['initScooping1']['right'] =\
        [['MOVEJ', '[-0.59, 0.131, -1.55, -1.041, 0.098, -1.136, -1.6]', 5.0],
         ['TOOL', 0],
         ['MOVES', '[0.7, 0., 0., -3.1415, 0.0, 1.574]', 5.],
         ['TOOL', self.org_tool]]

        self.motions['initScooping2'] = {}
        self.motions['initScooping2']['left'] = \
          [['MOVES', '[-0.04, 0.0, -0.15, 0, 0.5, 0]', 5, 'self.bowl_frame']]
        self.motions['initScooping2']['right'] =\
          self.motions['initScooping1']['right'][1:]

        self.motions['initScooping12'] = {}
        self.motions['initScooping12']['left'] = \
          [['MOVES', '[-0.04, 0.0, -0.15, 0, 0.5, 0]', 7, 'self.bowl_frame']]
          ## self.motions['initScooping2']['left'][0]]
          ## [self.motions['initScooping1']['left'][0],
        self.motions['initScooping12']['right'] = \
          self.motions['initScooping1']['right'][1:]



        # [Y (from center of bowl away from Pr2), X (towards right gripper), Z (towards floor) ,
        #  roll?, pitch (tilt downwards), yaw?]
        self.motions['runScooping'] = {}
        self.motions['runScooping']['left'] = \
          [['MOVES', '[-0.04, 0.04-self.highBowlDiff[1],  0.04, -0.2, 0.8, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVEL', '[ 0.04, -0.01-self.highBowlDiff[1],  0.03, -0.1, 0.8, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVES', '[ 0.04-0.005, -0.01-self.highBowlDiff[1],  -0.08, 0, 1.5, 0]', 3, 'self.bowl_frame'],]

        self.motions['runScooping_pspoon'] = {}
        self.motions['runScooping_pspoon']['left'] = \
          [['MOVES', '[-0.04-0.01, 0.035-self.highBowlDiff[1],  0.04, -0.2, 0.5, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVEL', '[ 0.04-0.01, -0.01-self.highBowlDiff[1],  0.03, -0.1, 0.8, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVES', '[ 0.04-0.01, -0.0-self.highBowlDiff[1],  -0.01, 0, 1.4, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVES', '[ 0.04-0.04, -0.0-self.highBowlDiff[1],  -0.1, 0, 1.4, 0]', 3, 'self.bowl_frame'],]

        ## Clean spoon motoins --------------------------------------------------------
        self.motions['cleanSpoon1'] = {}
        self.motions['cleanSpoon1']['left'] = \
          [['MOVEL', '[ 0.06+0.01, 0.03,  -0.09, 0, 1.5, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVEL', '[ 0.06+0.01, 0.03,  -0.02, 0, 1.5, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVEL', '[ 0.06-0.05, 0.03,  -0.02, 0, 1.5, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVEL', '[ 0.06-0.01, 0.0,  -0.1, 0, 1.5, 0]', 3, 'self.bowl_frame'],]
        self.motions['cleanSpoon1']['right'] = []
                                                

        ## Feeding motoins for a silicon spoon----------------------------------------------------
        # It uses the l_gripper_spoon_frame aligned with mouth

        self.motions['initFeeding1'] = {}
        self.motions['initFeeding1']['left'] =\
          [['MOVEJ', '[0.5447, 0.1256, 0.721, -2.12, 1.574, -0.7956, 1.0291]', 5.0],]        
        self.motions['initFeeding1']['right'] =\
          [['TOOL', 0],
           ['MOVES', '[0.22, 0., -0.55, 0., -1.85, 0.]', 5., 'self.mouth_frame'],
           ['TOOL', self.org_tool]]

        self.motions['initFeeding2'] = {}
        self.motions['initFeeding2']['left'] =\
        [['MOVEL', '[-0.06, -0.1, -0.2, -0.6, 0., 0.]', 5., 'self.mouth_frame']]

        self.motions['initFeeding3'] = {}
        self.motions['initFeeding3']['left'] =\
        [['MOVEL', '[-0.005+self.mouthOffset[0], self.mouthOffset[1], -0.15+self.mouthOffset[2], 0., 0., 0.]',
          5., 'self.mouth_frame'],\
        ['PAUSE', 0.0]]

        self.motions['initFeeding13'] = {}
        self.motions['initFeeding13']['right'] = [
            ['PAUSE', 4.0]]+self.motions['initFeeding1']['right']
        self.motions['initFeeding13']['left'] = [self.motions['initFeeding1']['left'][0],
                                               self.motions['initFeeding3']['left'][0]]        
        
        self.motions['runFeeding'] = {}
        self.motions['runFeeding']['left'] =\
        [['MOVEL', '[self.mouthOffset[0], self.mouthOffset[1], self.mouthOffset[2], 0., 0., 0.]',
          3., 'self.mouth_frame'],\
        ['PAUSE', 0.0],
        ['MOVEL', '[self.mouthOffset[0], self.mouthOffset[1], -0.15+self.mouthOffset[2], 0., 0., 0.]',
         4., 'self.mouth_frame']]


        ## Feeding motoins for a metalic fork----------------------------------------------------
        self.motions['initFeeding1_fork'] = {}
        self.motions['initFeeding1_fork']['left'] =\
          [['MOVEJ', '[0.2447, 0.1256, 0.721, -1.9, 1.374, -0.7956, 1.0291]', 7.0],]        

        self.motions['initFeeding13_fork'] = {}
        self.motions['initFeeding13_fork']['left'] = [self.motions['initFeeding1_fork']['left'][0],
                                                      self.motions['initFeeding3']['left'][0]]        


        ## Feeding motoins for a plastic spoon----------------------------------------------------

        self.motions['initFeeding1_pspoon'] = {}
        self.motions['initFeeding1_pspoon']['left'] =\
          [['MOVEJ', '[0.7447, 0.1256, 0.721, -1.9, 1.374, -0.7956, 1.0291]', 7.0],]        

        self.motions['initFeeding13_pspoon'] = {}
        self.motions['initFeeding13_pspoon']['left'] = \
          [self.motions['initFeeding1_pspoon']['left'][0],
           ['MOVEL', '[-0.005+self.mouthOffset[0], self.mouthOffset[1], -0.15+self.mouthOffset[2], 0., -0.2, 0.]',
            5., 'self.mouth_frame']]
          
        self.motions['runFeeding_pspoon'] = {}
        self.motions['runFeeding_pspoon']['left'] =\
        [['MOVEL', '[self.mouthOffset[0], self.mouthOffset[1], self.mouthOffset[2], 0., -0.2, 0.]',
          3., 'self.mouth_frame'],\
        ['PAUSE', 0.0],
        ['MOVEL', '[self.mouthOffset[0], self.mouthOffset[1], -0.15+self.mouthOffset[2], 0., -0.4, 0.]',
         4., 'self.mouth_frame']]



        ## Feeding motoins for a plastic spoon 2 (white)----------------------------------------------------
        
        self.motions['initFeeding3_pspoon2'] = {}
        self.motions['initFeeding3_pspoon2']['left'] =\
        [['MOVEL', '[-0.005+self.mouthOffset[0], self.mouthOffset[1], -0.15+self.mouthOffset[2], 0., -0.2, 0.]',
          5., 'self.mouth_frame'],\
        ['PAUSE', 0.0]]

        self.motions['runFeeding_pspoon2'] = {}
        self.motions['runFeeding_pspoon2']['left'] =\
        [['MOVEL', '[self.mouthOffset[0], self.mouthOffset[1], self.mouthOffset[2], 0., -0.2, 0.]',
          3., 'self.mouth_frame'],\
        ['PAUSE', 0.0],
        ['MOVEL', '[self.mouthOffset[0], self.mouthOffset[1], -0.15+self.mouthOffset[2], 0., -0.2, 0.]',
         4., 'self.mouth_frame']]


        rospy.loginfo("Parameters are loaded.")

    def serverCallback(self, req):
        task = req.data
        self.stop_motion = False

        if task == 'returnBowlPos':
            return self.bowl_frame

        if task == "getBowlPos":
            print '\n\n----getBowlPos called!-----\n\n'
            if self.bowl_frame_kinect is not None:
                print 'getBowlPos called 1!'
                self.bowl_frame = copy.deepcopy(self.bowl_frame_kinect)
                return "Choose kinect bowl position"
            elif self.bowl_frame_kinect is None:
                print 'getBowlPos called 2!'
                self.bowl_frame = copy.deepcopy(self.getBowlFrame())
                return "Choose bowl position from kinematics using tf"
            else:
                return "No kinect position available! \n Code won't work! \n \
                Provide head position and try again!"

        elif task == "getBowlPosRandom":
            if self.bowl_frame_kinect is not None:
                self.bowl_frame = copy.deepcopy(self.bowl_frame_kinect)
                return "Choose kinect bowl position"
            elif self.bowl_frame_kinect is None:
                self.bowl_frame = copy.deepcopy(self.getBowlFrame(addNoise=True))
                return "Choose bowl position from kinematics using tf"

        elif task == "getHeadPos":
            if self.mouth_frame_vision is not None:
                self.mouth_frame = copy.deepcopy(self.mouth_frame_vision)
                return self.arm_name+"Chose head position from vision"
            else:
                return "No kinect head position available! \n Code won't work! \n \
                Provide head position and try again!"

        elif task == "getBowlHighestPoint":
            self.bowl_height_init_pub.publish(Empty())
            return "Completed to get the highest point in the bowl."

        elif task == "lookAtBowl":
            self.lookAt(self.bowl_frame)
            # Account for the time it takes to turn the head
            rospy.sleep(2)
            return "Completed to move head"

        elif task == "lookAtMouth":
            self.lookAt(self.mouth_frame, tag_base='head')
            return "Completed to move head"

        elif task == 'lookToRight':
            # self.lookToRightSide()
            ## rospy.sleep(2.0)
            return 'Completed head movement to right'

        else:
            if task.find('initScooping')>=0 or task.find('initStabbing')>=0:
                self.kinect_pause.publish('start')
            elif task.find('initFeeding')>=0:
                self.kinect_pause.publish('pause')
            self.parsingMovements(self.motions[task][self.arm_name])
            return "Completed to execute "+task


    def highestBowlPointCallback(self, data):
        if not self.arm_name == 'left':
            return
        # Find difference between current highest point in bowl and center of bowl
        print 'Highest Point original position:', [data.x, data.y, data.z]
        print 'Bowl Position:', self.bowlPosition
        
        # Subtract 0.01 to account for the bowl center position being slightly off center
        # x: left
        # y: out
        self.highBowlDiff = np.array([data.x, data.y, data.z])
        ## self.highBowlDiff = np.array([data.x, data.y, data.z]) - self.bowlPosition -\
        ##   np.array([0.03,0,0])
          
        if np.linalg.norm(self.highBowlDiff) > 0.15: self.highBowlDiff = np.array([0.0,0,0])
        print '-'*25
        print 'Highest bowl point difference:', self.highBowlDiff
        print '-'*25


    def bowlPoseCallback(self, data):
        p = PyKDL.Vector(data.pose.position.x, data.pose.position.y, data.pose.position.z)
        M = PyKDL.Rotation.Quaternion(data.pose.orientation.x, data.pose.orientation.y,
                                      data.pose.orientation.z, data.pose.orientation.w)
        self.bowl_frame_kinect = PyKDL.Frame(M,p)


    def mouthPoseCallback(self, data):

        p = PyKDL.Vector(data.pose.position.x, data.pose.position.y, data.pose.position.z)
        M = PyKDL.Rotation.Quaternion(data.pose.orientation.x, data.pose.orientation.y,
                                      data.pose.orientation.z, data.pose.orientation.w)

        # get upright mouth frame
        tx = PyKDL.Vector(1.0, 0.0, 0.0)
        ty = PyKDL.Vector(0.0, 1.0, 0.0)

        # Projection to xy plane
        px = PyKDL.dot(tx, M.UnitZ())
        py = PyKDL.dot(ty, M.UnitZ())
        mouth_z = PyKDL.Vector(px, py, 0.0)
        mouth_z.Normalize()
        mouth_x = PyKDL.Vector(0.0, 0.0, 1.0)

        if self.viz:
            # only viz
            # (optional) publish pose for visualization
            ps = dh.gen_pose_stamped(PyKDL.Frame(M,p), 'torso_lift_link', rospy.Time.now() )
            self.mouth_pub.publish(ps)

        # fix mouth direction to y-direction of robot frame (temp??)
        mouth_z = PyKDL.Vector(0.0, 1.0, 0.0)
        mouth_y = mouth_z * mouth_x

        M = PyKDL.Rotation(mouth_x, mouth_y, mouth_z)
        self.mouth_frame_vision = PyKDL.Frame(M,p)


    def mouthNoiseCallback(self, msg):
        offset = msg.data
        self.mouthNoise[0] = offset[0]
        self.mouthNoise[1] = offset[1]
        self.mouthNoise[2] = offset[2]
        self.mouthOffset = self.mouthManOffset+self.mouthNoise
        

    def feedingDepthCallback(self, msg):
        print "Feeding depth requested ", msg.data
        self.mouthManOffset[2] = float(msg.data)/100.0
        self.feeding_depth_pub.publish( Int64(int(self.mouthManOffset[2]*100.0)) )
        self.mouthOffset = self.mouthManOffset+self.mouthNoise

        
    def feedingHorizCallback(self, msg):
        print "Feeding horizonal offset requested ", msg.data
        self.mouthManOffset[1] = -float(msg.data)/100.0
        self.feeding_horiz_pub.publish( Int64(int(-self.mouthManOffset[1]*100.0)) )
        self.mouthOffset = self.mouthManOffset+self.mouthNoise


    def feedingVertCallback(self, msg):
        print "Feeding vertical offset requested ", msg.data
        self.mouthManOffset[0] = float(msg.data)/100.0
        self.feeding_vert_pub.publish( Int64(int(self.mouthManOffset[0]*100.0)) )
        self.mouthOffset = self.mouthManOffset+self.mouthNoise
        

    def stopCallback(self, msg):
        print '\n\nAction Interrupted! Event Stop\n\n'
        print 'Interrupt Data:', msg.data
        self.stop_motion = True

        print "Stopping Motion..."
        self.setStop() #Stops Current Motion


    def getBowlFrame(self, addNoise=False):
        # Get frame info from right arm and upate bowl_pos

        ## # 1. right arm ('r_gripper_tool_frame') from tf
        self.tf_lstnr.waitForTransform(self.torso_frame, 'r_gripper_tool_frame', rospy.Time(0), rospy.Duration(5.0))
        [pos, quat] = self.tf_lstnr.lookupTransform(self.torso_frame, 'r_gripper_tool_frame', rospy.Time(0))

        p = PyKDL.Vector(pos[0],pos[1],pos[2])
        M = PyKDL.Rotation.Quaternion(quat[0], quat[1], quat[2], quat[3])

        ## # 2. add offset to called TF value. Make sure Orientation is up right.
        ## p = p + M*PyKDL.Vector(self.bowl_pos_offset['x'], self.bowl_pos_offset['y'], self.bowl_pos_offset['z'])
        ## M.DoRotX(self.bowl_orient_offset['rx'])
        ## M.DoRotY(self.bowl_orient_offset['ry'])
        ## M.DoRotZ(self.bowl_orient_offset['rz'])

        ## print 'Bowl frame:', p

        ## # 2.* add noise for random training
        ## if addNoise:
        ##     p = p + PyKDL.Vector(random.uniform(-0.1, 0.1),
        ##                          random.uniform(-0.1, 0.1),
        ##                          random.uniform(-0.1, 0.1))
        ## self.bowlPosition = np.array([p[0], p[1], p[2]])


        #
        frame = PyKDL.Frame(M,p) * self.r_tool_offset
        self.bowlPosition = np.array([frame.p[0], frame.p[1], frame.p[2]])

        if self.viz:
            # 4. (optional) publish pose for visualization
            ## ps = dh.gen_pose_stamped(PyKDL.Frame(M,p), 'torso_lift_link', rospy.Time.now() )
            ps = dh.gen_pose_stamped(frame, 'torso_lift_link', rospy.Time.now() )
            self.bowl_pub.publish(ps)

        return frame
        ## return PyKDL.Frame(M,p)

    def lookAt(self, target, tag_base='head'):

        t = time.time()
        head_frame  = rospy.get_param('hrl_manipulation_task/head_audio_frame')
        headClient = actionlib.SimpleActionClient("/head_traj_controller/point_head_action",
                                                  pr2_controllers_msgs.msg.PointHeadAction)
        headClient.wait_for_server()
        rospy.logout('Connected to head control server')

        ## print '1:', time.time() - t
        ## t = time.time()

        pos = Point()

        ps  = PointStamped()
        ps.header.frame_id = self.torso_frame

        head_goal_msg = pr2_controllers_msgs.msg.PointHeadGoal()
        head_goal_msg.pointing_frame = head_frame
        head_goal_msg.pointing_axis.x = 1
        head_goal_msg.pointing_axis.y = 0
        head_goal_msg.pointing_axis.z = 0
        head_goal_msg.min_duration = rospy.Duration(1.0)
        head_goal_msg.max_velocity = 1.0

        if target is None:
            ## tag_id = rospy.get_param('hrl_manipulation_task/'+tag_base+'/artag_id')

            while not rospy.is_shutdown() and self.mouth_frame_vision is None:
                rospy.loginfo("Search "+tag_base+" tag")

                pos.x = 0.8
                pos.y = 0.4
                pos.z = 0.0

                ps.point = pos
                head_goal_msg.target = ps

                headClient.send_goal(head_goal_msg)
                headClient.wait_for_result()
                # rospy.sleep(2.0)

            self.mouth_frame = copy.deepcopy(self.mouth_frame_vision)
            target = self.mouth_frame


        pos.x = target.p.x()
        pos.y = target.p.y()
        pos.z = target.p.z()

        ## print '2:', time.time() - t
        ## t = time.time()

        ps.point = pos
        head_goal_msg.target = ps

        headClient.send_goal(head_goal_msg)
        return "Success"

    def lookToRightSide(self):
        t = time.time()
        head_frame  = rospy.get_param('hrl_manipulation_task/head_audio_frame')
        headClient = actionlib.SimpleActionClient("/head_traj_controller/point_head_action",
                                                  pr2_controllers_msgs.msg.PointHeadAction)
        headClient.wait_for_server()
        rospy.logout('Connected to head control server')

        pos = Point()

        ps  = PointStamped()
        ps.header.frame_id = self.torso_frame

        head_goal_msg = pr2_controllers_msgs.msg.PointHeadGoal()
        head_goal_msg.pointing_frame = head_frame
        head_goal_msg.pointing_axis.x = 1
        head_goal_msg.pointing_axis.y = 0
        head_goal_msg.pointing_axis.z = 0
        head_goal_msg.min_duration = rospy.Duration(1.0)
        head_goal_msg.max_velocity = 1.0

        pos.x = 0
        pos.y = -2
        pos.z = 0

        ps.point = pos
        head_goal_msg.target = ps

        headClient.send_goal(head_goal_msg)
        ## headClient.wait_for_result() # TODO: This takes about 5 second -- very slow!
        return "Success"

    def pubCurEEPose(self):
        frame = self.getEndeffectorFrame(tool=self.cur_tool)
        ps = dh.gen_pose_stamped(frame, 'torso_lift_link', stamp = rospy.Time.now())
        self.ee_pose_pub.publish(ps)


if __name__ == '__main__':

    # TODO: optparse should be replaced to our own one.
    import optparse
    p = optparse.OptionParser()
    haptic_mpc_util.initialiseOptParser(p)
    opt = haptic_mpc_util.getValidInput(p)

    # Initial variables
    d_robot    = 'pr2'
    controller = 'static'
    ## controller = 'actionlib'
    arm        = opt.arm
    if opt.arm == 'l':
        verbose = True
    else:
        verbose = False


    rospy.init_node('arm_reacher_feeding_and_scooping')
    ara = armReachAction(d_robot, controller, arm, viz=True, verbose=verbose)
    rospy.spin()


