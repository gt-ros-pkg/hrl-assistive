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
import rospy
import PyKDL
import actionlib

# Msg
from geometry_msgs.msg import Pose, PoseStamped, Point, PointStamped, Quaternion
from std_msgs.msg import String, Empty, Int64
import pr2_controllers_msgs.msg
from hrl_msgs.msg import FloatArray
from hrl_srvs.srv import None_Bool, None_BoolResponse, String_String

# HRL library
import hrl_haptic_mpc.haptic_mpc_util as haptic_mpc_util
import hrl_lib.quaternion as quatMath

# Personal library - need to move neccessary libraries to a new package
from sandbox_dpark_darpa_m3.lib.hrl_mpc_base import mpcBaseAction
import sandbox_dpark_darpa_m3.lib.hrl_dh_lib as dh
QUEUE_SIZE = 10


class armReachAction(mpcBaseAction):
    def __init__(self, d_robot, controller, arm, tool_id=0, verbose=False):
        mpcBaseAction.__init__(self, d_robot, controller, arm, tool_id)

        #Variables...! #
        self.stop_motion = False
        self.verbose = verbose
        self.highBowlDiff = np.array([0, 0, 0])
        self.bowlPosition = np.array([0, 0, 0])
        # vertical x side x depth
        self.mouthManOffset = np.array([-0.03, 0.0, 0.04]) # -0.02, 0., 0.05
        ## self.mouthManOffset = np.array([-0.04, 0.02, 0.09]) # -0.02, 0., 0.05
        self.mouthNoise     = np.array([0., 0., 0.])
        self.mouthOffset    = self.mouthManOffset+self.mouthNoise 

        # exp 1:
        #self.mouthOffset  = [-0.04, 0.03, -0.02] # 
        #self.mouthOffset  = [-0.04, -0.12, 0.02] #
        #self.mouthOffset  = [-0.04, 0.03, 0.1] #
        #self.mouthOffset  = [-0.04, 0.06, 0.02] # 
        

        self.bowl_pub = None
        self.mouth_pub = None
        self.bowl_frame_kinect  = None
        self.mouth_frame_vision = None
        self.bowl_frame         = None
        self.mouth_frame        = None
        self.default_frame      = PyKDL.Frame()

        self.initCommsForArmReach()
        self.initParamsForArmReach()
        self.setMotions()

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
                ## break
                self.pubCurEEPose()
            rate.sleep()

        rospy.loginfo("Arm Reach Action is initialized.")

    def initCommsForArmReach(self):

        # publishers
        self.bowl_pub = rospy.Publisher('/hrl_manipulation_task/arm_reacher/bowl_cen_pose', PoseStamped,
                                        queue_size=QUEUE_SIZE, latch=True)
        self.mouth_pub = rospy.Publisher('/hrl_manipulation_task/arm_reacher/mouth_pose', PoseStamped,
                                         queue_size=QUEUE_SIZE, latch=True)
        self.ee_pose_pub = rospy.Publisher('/hrl_manipulation_task/arm_reacher/'+self.arm_name+\
                                           '_ee_pose', PoseStamped,
                                           queue_size=QUEUE_SIZE, latch=True)
        self.bowl_height_init_pub = rospy.Publisher('/hrl_manipulation_task/arm_reacher/init_bowl_height', Empty,
                                        queue_size=QUEUE_SIZE, latch=True)
        self.kinect_pause = rospy.Publisher('/head_mount_kinect/pause_kinect', String, queue_size=QUEUE_SIZE, latch=False)

        if self.arm_name == 'left':
            self.feeding_dist_pub = rospy.Publisher('/feeding/manipulation_task/feeding_dist_state', Int64, queue_size=QUEUE_SIZE, latch=True)
            msg = Int64()
            msg.data = int(self.mouthManOffset[2]*100.0)
            self.feeding_dist_pub.publish(msg)

        # subscribers
        rospy.Subscriber('/manipulation_task/InterruptAction', String, self.stopCallback)
        rospy.Subscriber('/hrl_manipulation_task/bowl_highest_point', Point, self.highestBowlPointCallback)
        ## rospy.Subscriber('/ar_track_alvar/bowl_cen_pose',
        ##                  PoseStamped, self.bowlPoseCallback)
        rospy.Subscriber('/hrl_manipulation_task/mouth_pose',
                         PoseStamped, self.mouthPoseCallback)
        rospy.Subscriber('/hrl_manipulation_task/mouth_noise', FloatArray, self.mouthNoiseCallback)
        if self.arm_name == 'left':
            rospy.Subscriber('/feeding/manipulation_task/feeding_dist_request', Int64, self.feedingDistCallback)

        # service
        self.reach_service = rospy.Service('arm_reach_enable', String_String, self.serverCallback)

        if self.verbose: rospy.loginfo("ROS-based communications are set up .")

    def initParamsForArmReach(self):

        ## Off set : 11 cm x direction, - 5 cm z direction.
        self.bowl_pos_offset    = rospy.get_param('/hrl_manipulation_task/sub_ee_pos_offset')
        self.bowl_orient_offset = rospy.get_param('/hrl_manipulation_task/sub_ee_orient_offset')


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

        #TOOL: Set a tool frame for MOVET. Defualt is 0 which is end-effector frame.

        joint or pose: we use radian and meter unit. The order of euler angle follows original x-y-z order (RPY).
        timeout or duration: we use second
        relative_frame: You can put your custome PyKDL frame variable or you can use 'self.default_frame'
        '''

        self.motions = {}

        self.motions['movestest'] = {}
        self.motions['movestest']['left'] = [['MOVEJ', '[0.6447, 0.1256, 0.721, -2.12, 1.574, -0.7956, 1.1291]', 3.0],\
                                             ['MOVEL', '[0, 0.0, 0.2, 0., 0, 0]', 4.0, 'self.getEndeffectorFrame(self.cur_tool)'],\
                                             ['MOVEL', '[0, 0.0, -0.2, 0., 0, 0]', 4.0, 'self.getEndeffectorFrame(self.cur_tool)']]


        ## Testing Motions ---------------------------------------------------------
        # Used to test and find the best optimal procedure to scoop the target.
        self.motions['test'] = {}
        self.motions['test']['left'] = [['MOVEJ', '[0.255, 0.358, 0.559, -1.682, 1.379, -1.076, 1.695]', 7.0],\
                                        ['PAUSE', 2.0],
                                        ['MOVET', '[-0.05, -0.2, -0.15, 0.6, 0., 0.]', 5.0]]

        self.motions['testingMotion'] = {}
        self.motions['testingMotion']['left'] = \
          [['MOVEJ', '[0.051, 0.219, 0.135, -2.115, -3.052, -1.928, -1.64]', 2.0],
          ['MOVEJ', '[0.054, 0.038, 0.298, -2.118, -3.090, -1.872, -1.39]', 2.0],
          ['MOVEJ', '[0.645, 0.016, 0.279, -2.118, -3.127, -1.803, -2.176]', 2.0],
          ['MOVEJ', '[0.051, 0.219, 0.135, -2.115, -3.053, -1.928, -1.64]', 2.0]]
        self.motions['testingMotion']['right'] = []


        ## Init arms ---------------------------------------------------------------
        self.motions['initArms'] = {}
        self.motions['initArms']['left']  = [['MOVEJ', '[0.6447, 0.1256, 0.721, -2.12, 1.574, -0.7956, 1.1291]', 10.0]]
        self.motions['initArms']['right'] = [['MOVEJ', '[-0.59, 0.0, -1.574, -1.041, 0.0, -1.136, -1.65]', 10.0]]

        ## Init arms ---------------------------------------------------------------
        self.motions['cleanSpoon1'] = {}
        self.motions['cleanSpoon1']['left']  = [['MOVEL', '[ 0.0, 0.0,  -0.01, 0, 1.4, 0]', 3, 'self.bowl_frame'],\
                                                ['PAUSE', 3.0],\
                                                ['MOVEL', '[ 0.0, 0.0,  -0.1, 0, 1.4, 0]', 3, 'self.bowl_frame']]
        self.motions['cleanSpoon1']['right'] = [['PAUSE', 3.0],\
                                                ['MOVET', '[0., -0.1, 0.0, 0., 0., 0.]', 3., 'self.default_frame'],\
                                                ['PAUSE', 2.0],\
                                                ['MOVET', '[0., 0.1, 0.0, 0., 0., 0.]', 3., 'self.default_frame'] ]
                                                

        ## Scooping motoins --------------------------------------------------------
        # Used to perform motions relative to bowl/mouth positions > It should use relative frame
        # [shoulder (towards left shoulder), arm pitch on shoulder (towards ground), whole arm roll (rotates right), elbow pitch (rotates towards outer arm),
        # elbow roll (rotates left), wrist pitch (towards top of forearm), wrist roll (rotates right)] (represents positive values)
        self.motions['initScooping1'] = {}
        self.motions['initScooping1']['left'] = [['PAUSE', 2.0],
                                                 ['MOVEJ', '[0.6447, 0.1256, 0.721, -2.12, 1.574, -0.7956, 1.1291]', 5.0]]
        self.motions['initScooping1']['right'] = [['MOVEJ', '[-0.59, 0.131, -1.55, -1.041, 0.098, -1.136, -1.4]', 5.0],
                                                  ['MOVES', '[0.7, -0.15, -0., -3.1415, 0.0, 1.574]', 2.]]
        ## 
                                                  

        self.motions['initScooping2'] = {}
        self.motions['initScooping2']['left'] = [['MOVES', '[-0.04, 0.0, -0.15, 0, 0.5, 0]', 3, 'self.bowl_frame']]
        self.motions['initScooping2']['right'] = []

        # only for training setup
        self.motions['initScooping2Random'] = {}
        self.motions['initScooping2Random']['left'] = []
        self.motions['initScooping2Random']['right'] = \
          [['MOVES', '[0.7+random.uniform(-0.1, 0.1), -0.15+random.uniform(-0.1, 0.1),-0.1+random.uniform(-0.1, 0.1), -3.1415, 0.0, 1.57]', 2.],]

        # [Y (from center of bowl away from Pr2), X (towards right gripper), Z (towards floor) , roll?, pitch (tilt downwards), yaw?]
        self.motions['runScooping'] = {}
        self.motions['runScoopingRight'] = {}
        self.motions['runScoopingLeft'] = {}
        self.motions['runScooping']['left'] = \
          [['MOVES', '[-0.05, 0.0-self.highBowlDiff[1],  0.04, 0, 0.6, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVEL', '[ 0.05, 0.0-self.highBowlDiff[1],  0.03, 0, 0.8, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVES', '[ 0.05, 0.0-self.highBowlDiff[1],  -0.1, 0, 1.3, 0]', 3, 'self.bowl_frame'],]


        self.motions['runWiping'] = {}
        self.motions['runWiping']['left'] = \
          [['MOVES', '[-0.05, 0.0-self.highBowlDiff[1],  0.04, 0, 0.7, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVEL', '[ 0.05, 0.0-self.highBowlDiff[1],  0.03, 0, 0.8, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVES', '[ 0.05-0.01, 0.0-self.highBowlDiff[1],  -0.065, 0, 1.5, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVES', '[ 0.05-0.01, 0.05-self.highBowlDiff[1],  -0.065, 0, 1.5, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVES', '[ 0.05-0.0, 0.05-self.highBowlDiff[1],  -0.04, 0, 1.5, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVES', '[ 0.05-0.05, 0.05-self.highBowlDiff[1],  -0.04, 0, 1.5, 0]', 3, 'self.bowl_frame'],
           ['PAUSE', 0.0],
           ['MOVES', '[ 0.05, 0.0-self.highBowlDiff[1],  -0.1, 0, 1.5, 0]', 3, 'self.bowl_frame'],]


        ## Feeding motoins --------------------------------------------------------
        # It uses the l_gripper_spoon_frame aligned with mouth
        self.motions['initFeeding'] = {}
        self.motions['initFeeding']['left'] = [['MOVEJ', '[0.327, 0.205, 1.05, -2.08, 2.57, -1.29, 0.576]', 5.0]]
        self.motions['initFeeding']['right'] = [['MOVES', '[0.22, 0., -0.55, 0., -1.85, 0.]', 5., 'self.mouth_frame'],
                                                ['PAUSE', 2.0]]

        self.motions['initFeeding1'] = {}
        self.motions['initFeeding1']['left'] = [['MOVEJ', '[0.6447, 0.1256, 0.721, -2.12, 1.574, -0.7956, 1.1291]', 5.0],]
                                                ## ['MOVET', '[-0.05, -0.2, -0.15, 0.6, 0., 0.]', 5.0]]
                                                ## ['MOVEL', '[ -0.1, -0.1, -0.3, -0.8, 0., 0.]', 7., 'self.mouth_frame'],
                                                ## ['PAUSE', 2.0]]

        self.motions['initFeeding2'] = {}
        self.motions['initFeeding2']['left'] = [['MOVEL', '[-0.06, -0.1, -0.2, -0.6, 0., 0.]', 5., 'self.mouth_frame']]

        self.motions['initFeeding3'] = {}
        self.motions['initFeeding3']['left'] = [['MOVEL', '[-0.005+self.mouthOffset[0], self.mouthOffset[1], -0.15+self.mouthOffset[2], 0., 0., 0.]', 5., 'self.mouth_frame'],\
                                                ['PAUSE', 1.0]]
        self.motions['runFeeding'] = {}
        self.motions['runFeeding']['left'] = [['MOVEL', '[self.mouthOffset[0], self.mouthOffset[1], self.mouthOffset[2], 0., 0., 0.]', 3., 'self.mouth_frame'],\
                                              ['PAUSE', 0.0],
                                              ['MOVEL', '[self.mouthOffset[0], self.mouthOffset[1], -0.2+self.mouthOffset[2], 0., 0., 0.]', 4., 'self.mouth_frame']]
        ## self.motions['runFeeding']['left'] = [['MOVES', '[-0.02, 0.0, 0.05, 0., 0., 0.]', 5., 'self.mouth_frame'],\
        ##                                       ['PAUSE', 0.5],
        ##                                       ['MOVES', '[-0.02, 0.0, -0.1, 0., 0., 0.]', 5., 'self.mouth_frame']]

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

        elif task == "lookAtBowl":
            self.lookAt(self.bowl_frame)
            # Account for the time it takes to turn the head
            rospy.sleep(2)
            self.bowl_height_init_pub.publish(Empty())
            return "Completed to move head"

        elif task == "lookAtMouth":
            self.lookAt(self.mouth_frame, tag_base='head')
            return "Completed to move head"

        elif task == 'lookToRight':
            # self.lookToRightSide()
            ## rospy.sleep(2.0)
            return 'Completed head movement to right'

        else:
            if task == 'initScooping1':
                self.kinect_pause.publish('start')
            elif task == 'initFeeding':
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
        self.highBowlDiff = np.array([data.x, data.y, data.z]) - self.bowlPosition - 0.015
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
        

    def feedingDistCallback(self, msg):
        print "Feeding distance requested ", msg.data
        self.mouthManOffset[2] = float(msg.data)/100.0
        msg = Int64()
        msg.data = int(self.mouthManOffset[2]*100.0)
        self.feeding_dist_pub.publish(msg)
        
        

    def stopCallback(self, msg):
        print '\n\nAction Interrupted! Event Stop\n\n'
        print 'Interrupt Data:', msg.data
        self.stop_motion = True

        print "Stopping Motion..."
        self.setStop() #Stops Current Motion
        ## try:
        ##     self.setStopRight() #Sends message to service node
        ## except:
        ##     rospy.loginfo("Couldn't stop "+self.arm_name+" arm! ")

        ## posStopL = Point()
        ## quatStopL = Quaternion()
        ## rospy.sleep(2.0)
        ## import hrl_lib.util as ut
        ## ut.get_keystroke('Hit a key to proceed next')

        # TODO: location should be replaced into the last scooping or feeding starts.
        #print "Moving left arm to safe position "
        ## self.parsingMovements(self.motions['initScooping1'][self.arm_name])
        ## ut.get_keystroke('Hit a key to proceed next')

        ## if data.data == 'InterruptHead':
        ##     self.feeding([0])
        ##     self.setPostureGoal(self.lInitAngFeeding, 10)
        ## else:
        ##     self.scooping([0])
        ##     self.setPostureGoal(self.lInitAngScooping, 10)


    def getBowlFrame(self, addNoise=False):
        # Get frame info from right arm and upate bowl_pos

        # 1. right arm ('r_gripper_tool_frame') from tf
        self.tf_lstnr.waitForTransform(self.torso_frame, 'r_gripper_tool_frame', rospy.Time(0), rospy.Duration(5.0))
        [pos, quat] = self.tf_lstnr.lookupTransform(self.torso_frame, 'r_gripper_tool_frame', rospy.Time(0))

        p = PyKDL.Vector(pos[0],pos[1],pos[2])
        M = PyKDL.Rotation.Quaternion(quat[0], quat[1], quat[2], quat[3])

        # 2. add offset to called TF value. Make sure Orientation is up right.
        p = p + M*PyKDL.Vector(self.bowl_pos_offset['x'], self.bowl_pos_offset['y'], self.bowl_pos_offset['z'])
        M.DoRotX(self.bowl_orient_offset['rx'])
        M.DoRotY(self.bowl_orient_offset['ry'])
        M.DoRotZ(self.bowl_orient_offset['rz'])

        print self.bowl_pos_offset
        print pos
        print quat
        print 'Bowl frame:', p

        # 2.* add noise for random training
        if addNoise:
            p = p + PyKDL.Vector(random.uniform(-0.1, 0.1),
                                 random.uniform(-0.1, 0.1),
                                 random.uniform(-0.1, 0.1))

        self.bowlPosition = np.array([p[0], p[1], p[2]])

        # 4. (optional) publish pose for visualization
        ps = dh.gen_pose_stamped(PyKDL.Frame(M,p), 'torso_lift_link', rospy.Time.now() )
        self.bowl_pub.publish(ps)

        return PyKDL.Frame(M,p)

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
        ## headClient.wait_for_result() # TODO: This takes about 5 second -- very slow!
        # rospy.sleep(1.0)

        ## print '3:', time.time() - t

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
    #controller = 'actionlib'
    arm        = opt.arm
    if opt.arm == 'l':
        tool_id = 1
        verbose = True
    else:
        tool_id = 0
        verbose = False


    rospy.init_node('arm_reacher_feeding_and_scooping')
    ara = armReachAction(d_robot, controller, arm, tool_id, verbose)
    rospy.spin()


