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
        self.bowl_pub = rospy.Publisher('/hrl_manipulation_task/bowl_cen_pose', PoseStamped, latch=True)
        self.mouth_pub = rospy.Publisher('/hrl_manipulation_task/mouth_pose', PoseStamped, latch=True)
        
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
          [['MOVEJ', '[0.4447, 0.1256, 0.721, -2.12, 1.574, -0.7956, 0.8291]', 10.0] ] 
        self.motions['initScooping']['right'] = \
          [['MOVEJ', '[-0.848, 0.175, -1.676, -1.627, -0.097, -0.777, -1.704]', 5.0],
           ['MOVES', '[0.6, -0.15, -0.1, -3.1415, 0.0, 1.57]', 5.],
           ['PAUSE', 2.0]]
          
        self.motions['runScooping'] = {}
        self.motions['runScooping']['left'] = \
          [['MOVES', '[-0.06, 0.0, -0.1, 0, 0.7, 0]', 5, 'self.bowl_frame'],
           ['MOVES', '[-0.06, 0.0,  0.04, 0, 0.7, 0]', 5, 'self.bowl_frame'],
           ['MOVES', '[ 0.02, 0.0,  0.04, 0, 1.2, 0]', 5, 'self.bowl_frame'],
           ['MOVES', '[ 0.0,  0.0, -0.1, 0, 1.2, 0]', 5, 'self.bowl_frame'] ]
        self.motions['runScooping']['right'] = \
          []
        
        ## Feeding motoins --------------------------------------------------------
        # It uses the l_gripper_spoon_frame aligned with mouth
        self.motions['initFeeding'] = {}
        self.motions['initFeeding']['left'] = \
          [['MOVEJ', '[0.645, -0.198, 1.118, -2.121, 1.402, -0.242, 0.939]', 10.0],
           ['MOVES', '[0.705, 0.348, -0.029, 0.98, -1.565, -2.884]', 10.0, 'self.default_frame'], 
           ['PAUSE', 2.0] ]           
        self.motions['initFeeding']['right'] = \
          [['MOVEJ', '[-1.57, 0.0, -1.57, -1.69, 0.0, -0.748, -1.57]', 5.0]]

        self.motions['runFeeding'] = {}
        self.motions['runFeeding']['left'] = \
          [['MOVES', '[0.0, 0.0, -0.1, 0., 0., 0.]', 5., 'self.mouth_frame'],                     
           ['MOVES', '[0.0, 0.0, -0.04, 0., 0., 0.]', 5., 'self.mouth_frame'],
           ['MOVES', '[0.0, 0.0, -0.1, 0., 0., 0.]', 5., 'self.mouth_frame'],                     
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
                self.bowl_frame = copy.copy(self.getBowlFrame())
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

        # get upright mouth frame        
        tx = PyKDL.Vector(1.0, 0.0, 0.0)
        ty = PyKDL.Vector(0.0, 1.0, 0.0)

        # Projection to xy plane
        px = PyKDL.dot(tx, M.UnitZ())
        py = PyKDL.dot(ty, M.UnitZ())
        mouth_z = PyKDL.Vector(px, py, 0.0)
        mouth_z.Normalize()

        mouth_x = PyKDL.Vector(0.0, 0.0, 1.0)
        mouth_y = mouth_z * mouth_x
        M = PyKDL.Rotation(mouth_x, mouth_y, mouth_z)
        self.mouth_frame_kinect = PyKDL.Frame(M,p)

        # 4. (optional) publish pose for visualization        
        ps = PoseStamped()
        ps.header.frame_id = 'torso_lift_link'
        ps.header.stamp = rospy.Time.now()
        ps.pose.position.x = p[0]
        ps.pose.position.y = p[1]
        ps.pose.position.z = p[2]
        
        ps.pose.orientation.x = M.GetQuaternion()[0]
        ps.pose.orientation.y = M.GetQuaternion()[1]
        ps.pose.orientation.z = M.GetQuaternion()[2]
        ps.pose.orientation.w = M.GetQuaternion()[3]                
        self.mouth_pub.publish(ps)
        
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


    def getBowlFrame(self):
        # Get frame info from right arm and upate bowl_pos                

        # 1. right arm ('r_gripper_tool_frame') from tf
        self.tf_lstnr.waitForTransform(self.torso_frame, 'r_gripper_tool_frame', rospy.Time(0), \
                                       rospy.Duration(5.0))
        [pos, quat] = self.tf_lstnr.lookupTransform(self.torso_frame, 'r_gripper_tool_frame', \
                                                   rospy.Time(0))
        p = PyKDL.Vector(pos[0],pos[1],pos[2])
        M = PyKDL.Rotation.Quaternion(quat[0], quat[1], quat[2], quat[3])

        # 2. add offset to called TF value. Make sure Orientation is up right. 
        ## Off set : 11 cm x direction, - 5 cm z direction. 
        p = p + M*PyKDL.Vector(0.11, 0, 0.04)
        M.DoRotZ(np.pi/2.0)        
        ## RPY(np.pi, -np.py, 0.0)

        # 4. (optional) publish pose for visualization
        ps = PoseStamped()
        ps.header.frame_id = 'torso_lift_link'
        ps.header.stamp = rospy.Time.now()
        ps.pose.position.x = p[0]
        ps.pose.position.y = p[1]
        ps.pose.position.z = p[2]
        
        ps.pose.orientation.x = M.GetQuaternion()[0]
        ps.pose.orientation.y = M.GetQuaternion()[1]
        ps.pose.orientation.z = M.GetQuaternion()[2]
        ps.pose.orientation.w = M.GetQuaternion()[3]        
        self.bowl_pub.publish(ps)
        
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
    if opt.arm == 'l': verbose = False
    else: verbose = True
        

    rospy.init_node('arm_reacher_feeding_and_scooping')
    ara = armReachAction(d_robot, controller, arm, verbose)
    rospy.spin()


