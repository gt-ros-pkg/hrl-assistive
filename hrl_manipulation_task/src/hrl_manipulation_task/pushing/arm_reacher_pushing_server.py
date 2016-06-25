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
import sys, time, copy
import random
import numpy as np

# ROS
import rospy, roslib
import tf
import PyKDL
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import String

# HRL library
import hrl_haptic_mpc.haptic_mpc_util as haptic_mpc_util
import hrl_lib.quaternion as quatMath 
from hrl_srvs.srv import None_Bool, String_String

# Personal library
from sandbox_dpark_darpa_m3.lib.hrl_mpc_base import mpcBaseAction
from sandbox_dpark_darpa_m3.lib import hrl_dh_lib as dh

class armReachAction(mpcBaseAction):
    def __init__(self, d_robot, controller, arm, tool_id=0, verbose=False):
        mpcBaseAction.__init__(self, d_robot, controller, arm, tool_id)

        #Variables...! #
        self.stop_motion = False
        self.verbose     = verbose
        self.tag0_frame  = None

        self.default_frame      = PyKDL.Frame()

        self.initCommsForArmReach()                            
        self.initParamsForArmReach()

        rate = rospy.Rate(100) # 25Hz, nominally.
        while not rospy.is_shutdown():
            if self.getJointAngles() != []:
                if verbose:
                    print "--------------------------------"
                    print "Current "+self.arm_name+" arm joint angles"
                    print self.getJointAngles()
                    print "Current "+self.arm_name+" arm pose"
                    print self.getEndeffectorPose(tool=tool_id)
                    print "Current "+self.arm_name+" arm orientation (w/ euler rpy)"
                    print self.getEndeffectorRPY(tool=tool_id) #*180.0/np.pi
                    print "--------------------------------"
                break
            rate.sleep()
            
        rospy.loginfo("Arm Reach Action is initialized.")
        print "Current "+self.arm_name+" arm joint angles"
        print self.getJointAngles()
        print "Current "+self.arm_name+" arm pose"
        print self.getEndeffectorPose(tool=tool_id)
                            
    def initCommsForArmReach(self):

        # publishers and subscribers
        rospy.Subscriber('/hrl_manipulation_task/InterruptAction', String, self.stopCallback)
        rospy.Subscriber('/ar_track_alvar/pose_0', PoseStamped, self.mainTagPoseCallback)
        
        # service
        self.reach_service = rospy.Service('arm_reach_enable', String_String, self.serverCallback)
        ## rospy.Subscriber('/ar_track_alvar/artag_vision_pose_0', PoseStamped, self.tagPoseCallback)

        if self.verbose: rospy.loginfo("ROS-based communications are set up .")
                                    
    def initParamsForArmReach(self):
        '''
        Industrial movment commands generally follows following format, 
        
               Movement type, joint or pose(pos+euler), timeout, relative_frame

        In this code, we allow to use following movement types,

        MOVEP: point-to-point motion without orientation control (ex. MOVEP pos-euler timeout relative_frame)
        MOVES: point-to-point motion with orientation control (ex. MOVES pos-euler timeout relative_frame)
        MOVEL: straight (linear) motion with orientation control (ex. MOVEL pos-euler timeout relative_frame)
        MOVET: MOVES with respect to the current tool frame (ex. MOVET pos-euler timeout) (experimental!!)
        MOVEJ: joint motion (ex. MOVEJ joint timeout)
        PAUSE: Add pause time between motions (ex. PAUSE duration)

        #TOOL: Set a tool frame for MOVET. Defualt is 0 which is end-effector frame.

        joint or pose: we use radian and meter unit. The order of euler angle follows original x-y-z order (RPY).
        timeout or duration: we use second
        relative_frame: You can put your custome PyKDL frame variable or you can use 'self.default_frame'
        '''
        
        self.motions = {}

        ## test motoins --------------------------------------------------------
        # It uses the l_gripper_push_frame
        self.motions['initTest'] = {}
        self.motions['initTest']['left'] = \
          [['MOVEJ', '[0.44, 0.98, 0.53, -2.06, 3.12, -1.05, 2.84]', 5.0],
           ## ['MOVEL', '[0.7, 0.190, 0.14, -0.04, 0.026, -0.105, 0.9929 ]', 3.0],
           ['PAUSE', 1.0],
           #['MOVEL', '[-0.2, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0]', 5.0, 'self.microwave_white_frame'],           
           #['MOVEL', '[-0.05, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0]', 2.5, 'self.microwave_white_frame'],           
           #['MOVEL', '[-0.2, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0]', 3.0, 'self.microwave_white_frame'],           
           ## ['PAUSE', 1.0],
           ## ['MOVEL', '[-0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]', 5.0, 'self.microwave_white_frame'],           
           ## ['MOVEL', '[0.7, 0.190, 0.04, -0.04, 0.026, -0.105, 0.9929 ]', 3.0],
           ['MOVET', '[0., 0.0, 0.1, 0.0, 0.0, 0.0 ]', 3.0],
           ['PAUSE', 1.0],
           ['MOVET', '[-0., 0.0, -0.1, 0.0, 0.0, 0.0 ]', 3.0],
          ] 
        ## 
           
        self.motions['initTest']['right'] = []
        
        ## Pushing white microwave motoins --------------------------------------------------------
        # It uses the l_gripper_push_frame
        self.motions['initMicroWhite'] = {}
        self.motions['initMicroWhite']['left'] = \
          [['MOVEJ', '[0.44, 0.98, 0.53, -2.06, 3.12, -1.05, 2.84]', 5.0],
           ['MOVEL', '[-0.2, -0.0, -0.1, 0.0, 0.0, 0.0, 1.0]', 5.0, 'self.main_tag_frame'], 
           ['PAUSE', 1.0]
            ] 
        self.motions['initMicroWhite']['right'] = []

        self.motions['runMicroWhite'] = {}
        self.motions['runMicroWhite']['left'] = \
          [['MOVEL', '[-0.05, -0.0, -0.1, 0.0, 0.0, 0.0, 1.0]', 2.5, 'self.main_tag_frame'],
           ['MOVEL', '[-0.2, -0.0, -0.1, 0.0, 0.0, 0.0, 1.0]', 3.0, 'self.main_tag_frame']] 
        self.motions['runMicroWhite']['right'] = []

        ## Pushing black microwave motoins --------------------------------------------------------
        # It uses the l_gripper_push_frame
        self.motions['initMicroBlack'] = {}
        self.motions['initMicroBlack']['left'] = \
          [['MOVEJ', '[0.44, 0.98, 0.53, -2.06, 3.12, -1.05, 2.84]', 5.0],
           ['MOVEL', '[-0.2, -0.0, -0.1, 0.0, 0.0, 0.0, 1.0]', 5.0, 'self.main_tag_frame'], 
           ['PAUSE', 1.0]
            ] 
        self.motions['initMicroBlack']['right'] = []

        self.motions['runMicroBlack'] = {}
        self.motions['runMicroBlack']['left'] = \
          [['MOVEL', '[-0.05, -0.0, -0.1, 0.0, 0.0, 0.0, 1.0]', 2.5, 'self.main_tag_frame'],
           ['MOVEL', '[-0.2, -0.0, -0.1, 0.0, 0.0, 0.0, 1.0]', 3.0, 'self.main_tag_frame']] 
        self.motions['runMicroBlack']['right'] = []

        ## Pushing heatgun tool case motoins --------------------------------------------------
        # It uses the l_gripper_push_frame
        self.motions['initToolCase'] = {}
        self.motions['initToolCase']['left'] = \
          [['MOVEJ', '[0.82, 0.43, 1.52, -1.26, 2.04, -1.49, 0.0]', 5.0] ] 
        self.motions['initToolCase']['right'] = []

        self.motions['runToolCase1'] = {}
        self.motions['runToolCase1']['left'] = \
          [['MOVEL', '[-0.03, 0.08, 0.05, 0.0, 1.57, 3.14]', 5.0, 'self.main_tag_frame'], 
           ['PAUSE', 1.0] ]
        self.motions['runToolCase1']['right'] = []
        ## 
        
        self.motions['runToolCase2'] = {}
        self.motions['runToolCase2']['left'] = \
          [['MOVEL', '[-0.03, 0.08, -0.03, 0.0, 1.57, 3.14]', 2.5, 'self.main_tag_frame'],
           ['MOVEL', '[-0.03, 0.08, 0.05, 0.0, 1.57, 3.14]', 3.0, 'self.main_tag_frame']] 
        self.motions['runToolCase2']['right'] = []
        
        ## Pushing cabinet motoins --------------------------------------------------------
        self.motions['initCabinet'] = {}
        self.motions['initCabinet']['left'] = \
          [] 
        self.motions['initCabinet']['right'] = \
        [['MOVEJ', '[-1.19, 0.667, -0.36, -1.63, 4.32, -1.02, -2.007]', 5.0]]

        self.motions['runCabinet'] = {}
        self.motions['runCabinet']['left'] = \
          []
        self.motions['runCabinet']['right'] = \
          [['MOVET', '[0., 0., -0.2, 0., 0., 0.]', 10., 'self.default_frame'],
           ['MOVET', '[0.2, 0., 0, 0., 0., 0.]', 10., 'self.default_frame'],
           ['MOVET', '[-0.2, 0., 0, 0., 0., 0.]', 10., 'self.default_frame'],
           ['MOVET', '[0., 0., 0.2, 0., 0., 0.]', 10., 'self.default_frame'],
           ]
                                                    
        rospy.loginfo("Parameters are loaded.")
                
        
    def serverCallback(self, req):
        task = req.data
        self.stop_motion = False

        if task == 'getMainTagPos':
            self.main_tag_frame = copy.deepcopy(self.tag0_frame)
            return "Get a main tag position"        
        else:        
            self.parsingMovements(self.motions[task][self.arm_name])
            return "Completed to execute "+task

    
    def stopCallback(self, msg):
        print '\n\nAction Interrupted! Event Stop\n\n'
        print 'Interrupt Data:', msg.data
        self.stop_motion = True

        print "Stopping Motion..."
        self.setStop() #Stops Current Motion
        try:
            self.setStopRight() #Sends message to service node
        except:
            rospy.loginfo("Couldn't stop "+self.arm_name+" arm! ")


    def mainTagPoseCallback(self, data):

        ## tag_frame = dh.pose2KDLframe(data.pose)
        ## M = PyKDL.Rotation.Quaternion(0,0,0,1)
        ## self.tag0_frame = PyKDL.Frame(M, tag_frame.p)
        
        self.tag0_frame = dh.pose2KDLframe(data.pose)
        


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    haptic_mpc_util.initialiseOptParser(p)
    opt = haptic_mpc_util.getValidInput(p)

    # Initial variables
    d_robot    = 'pr2'
    ## controller = 'static' # quasistatic model predictive controller
    controller = 'actionlib'
    arm        = opt.arm
    tool_id    = 2
    if opt.arm == 'l': verbose = False
    else: verbose = True
        
    rospy.init_node('arm_reacher_pushing')
    ara = armReachAction(d_robot, controller, arm, tool_id, verbose)
    rospy.spin()


