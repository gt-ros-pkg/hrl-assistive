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
import numpy as np

# ROS
import rospy, roslib
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
        self.stop_motion = False
        self.verbose = verbose

        self.bowl_frame_kinect  = None
        self.mouth_frame_kinect = None
        self.default_frame      = PyKDL.Frame()

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
                    print self.getEndeffectorPose(tool=1)
                    print "Current "+self.arm+" arm orientation (w/ euler rpy)"
                    print self.getEndeffectorRPY(tool=1) #*180.0/np.pi
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

        if self.verbose: rospy.loginfo("ROS-based communications are set up .")
                                    
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

        joint or pose: we use radian and meter unit. The order of euler angle follows original z-y-x order (RPY).
        timeout or duration: we use second
        relative_frame: You can put your custome PyKDL frame variable or you can use 'self.default_frame'
        '''
        
        self.motions = {}

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
          [['MOVEJ', '[0.785, 0, 1.57, -2.356, -3.14, -1.0, 0.0]', 10.0]] 
        self.motions['initFeeding']['right'] = \
          []

        # another initial posture
        #['MOVEJ', '[1.57, 0, 1.57, -2.356, -3.14, -0.5, 0.0]', 10.0]
        
        self.motions['runFeeding'] = {}
        self.motions['runFeeding']['left'] = \
          [['MOVES', '[0.0, 0.0, -0.1, 0., 0., 0.]', 20., 'self.mouth_frame'],
           ['MOVES', '[0.0, 0.0, 0.0, 0., 0., 0.]', 10., 'self.mouth_frame'],
           ['MOVES', '[0.0, 0.0, -0.1, 0., 0., 0.]', 10., 'self.mouth_frame'],
           ['MOVEJ', '[0.785, 0., 1.57, -2.356, -3.14, -1.0, 0.0]', 10.0],           
           ]
        self.motions['runFeeding']['right'] = \
          []
                                                    
        rospy.loginfo("Parameters are loaded.")
                
        
    def serverCallback(self, req):
        req = req.data
        self.stop_motion = False

        if req == "getBowlPos":
            if self.bowl_frame_kinect is not None:
                self.bowl_frame = self.bowl_frame_kinect
                return "Chose kinect bowl position"
            elif self.bowl_frame_kinect is None:
                # Get frame info from right arm and upate bowl_pos                
                # 1. right arm ('r_gripper_tool_frame') from tf
                self.tf_lstnr.waitForTransform(self.torso_frame, 'r_gripper_tool_frame', rospy.Time(0), \
                                               rospy.Duration(5.0))
                ## [self.----pos, self.---quat] = \
                ##     self.tf_lstnr.lookupTransform(self.torso_frame, 'r_gripper_tool_frame', rospy.Time(0))

                
                # 2. add offset to called TF value. Make sure Orientation is up right. 
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

        
                
    def bowlPoseCallback(self, data):
        p = PyKDL.Vector(data.pose.position.x, data.pose.position.y, data.pose.position.z)
        M = PyKDL.Rotation.Quaternion(data.pose.orientation.x, data.pose.orientation.y, 
                                      data.pose.orientation.z, data.pose.orientation.w)
        self.bowl_frame_kinect = PyKDL.Frame(M,p)

        
    def mouthPoseCallback(self, data):

        p = PyKDL.Vector(data.pose.position.x, data.pose.position.y, data.pose.position.z)
        M = PyKDL.Rotation.Quaternion(data.pose.orientation.x, data.pose.orientation.y, 
                                      data.pose.orientation.z, data.pose.orientation.w)
        M.DoRotX(np.pi)        
        self.mouth_frame_kinect = PyKDL.Frame(M,p)
        

    def stopCallback(self, msg):
        print '\n\nAction Interrupted! Event Stop\n\n'
        print 'Interrupt Data:', msg.data
        self.stop_motion = True

        print "Stopping Motion..."
        self.setStop() #Stops Current Motion
        try:
            self.setStopRight() #Sends message to service node
        except:
            rospy.loginfo("Couldn't stop "+self.arm+" arm! ")

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
    if opt.arm == 'l': verbose = True
    else: verbose = False
        

    rospy.init_node('arm_reacher_feeding_and_scooping')
    ara = armReachAction(d_robot, controller, arm, verbose)
    rospy.spin()


