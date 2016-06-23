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
import PyKDL
from geometry_msgs.msg import Pose, PoseStamped, Point, PointStamped, Quaternion
from std_msgs.msg import String
import pr2_controllers_msgs.msg
import actionlib

# HRL library
import hrl_haptic_mpc.haptic_mpc_util as haptic_mpc_util
import hrl_lib.quaternion as quatMath 
from hrl_srvs.srv import None_Bool, None_BoolResponse, String_String

# Personal library
from sandbox_dpark_darpa_m3.lib.hrl_mpc_base import mpcBaseAction
QUEUE_SIZE = 10


class armReachAction(mpcBaseAction):
    def __init__(self, d_robot, controller, arm, tool_id=0, verbose=False):
        mpcBaseAction.__init__(self, d_robot, controller, arm, tool_id)

        #Variables...! #
        self.stop_motion = False
        self.verbose = verbose

        self.bowl_frame_kinect  = None
        self.mouth_frame_vision = None
        self.bowl_frame         = None
        self.mouth_frame        = None
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
                    print self.getEndeffectorPose(tool=1)
                    print "Current "+self.arm_name+" arm orientation (w/ euler rpy)"
                    print self.getEndeffectorRPY(tool=1) #*180.0/np.pi
                    print "--------------------------------"
                break
            rate.sleep()
            
        rospy.loginfo("Arm Reach Action is initialized.")
                            
    def initCommsForArmReach(self):

        # publishers
        self.bowl_pub = rospy.Publisher('/hrl_manipulation_task/arm_reacher/bowl_cen_pose', PoseStamped,
                                        queue_size=QUEUE_SIZE, latch=True)
        self.mouth_pub = rospy.Publisher('/hrl_manipulation_task/arm_reacher/mouth_pose', PoseStamped,
                                         queue_size=QUEUE_SIZE, latch=True)

        # subscribers
        rospy.Subscriber('/hrl_manipulation_task/InterruptAction', String, self.stopCallback)
        ## rospy.Subscriber('/ar_track_alvar/bowl_cen_pose',
        ##                  PoseStamped, self.bowlPoseCallback)
        rospy.Subscriber('/hrl_manipulation_task/mouth_pose',
                         PoseStamped, self.mouthPoseCallback)
        
        # service
        self.reach_service = rospy.Service('arm_reach_enable', String_String, self.serverCallback)

        if self.verbose: rospy.loginfo("ROS-based communications are set up .")
                                    
    def initParamsForArmReach(self):
        '''
        Industrial movment commands generally follows following format, 
        
               Movement type, joint or pose(pos+euler), timeout, relative_frame, threshold

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

        self.motions['test'] = {}
        self.motions['test']['right'] = [['MOVES', '[0.7, -0.15, -0.1, -3.1415, 0.0, 1.57]', 5.], ]
        
        ## Testing Motions ---------------------------------------------------------
        # Used to test and find the best optimal procedure to scoop the target.
        self.motions['testingMotion'] = {}
        self.motions['testingMotion']['left'] = \
          [['MOVEJ', '[0.051, 0.219, 0.135, -2.115, -3.052, -1.928, -1.64]', 2.0],
          ['MOVEJ', '[0.054, 0.038, 0.298, -2.118, -3.090, -1.872, -1.39]', 2.0],
          ['MOVEJ', '[0.645, 0.016, 0.279, -2.118, -3.127, -1.803, -2.176]', 2.0],
          ['MOVEJ', '[0.051, 0.219, 0.135, -2.115, -3.053, -1.928, -1.64]', 2.0]]

        #  ['MOVES', '[ 0.521, -0.137, -0.041, 38, -99, -4]', 5.0]]
        self.motions['testingMotion']['right'] = []
        ##  [['MOVEJ', '[1.570, 0, 1.570, -1.570, -4.71, 0, -1.570]', 5.0]]
        
        ## Scooping motoins --------------------------------------------------------
        # Used to perform motions relative to bowl/mouth positions > It should use relative frame
        # [shoulder (towards left shoulder), arm pitch on shoulder (towards ground), whole arm roll (rotates right), elbow pitch (rotates towards outer arm),
        # elbow roll (rotates left), wrist pitch (towards top of forearm), wrist roll (rotates right)] (represents positive values)
        self.motions['initScooping1'] = {}
        self.motions['initScooping1']['left'] = [['MOVEJ', '[0.6447, 0.1256, 0.721, -2.12, 1.574, -0.7956, 0.8291]', 5.0]]
        self.motions['initScooping1']['right'] = [['MOVEJ', '[-0.59, 0.131, -1.55, -1.041, 0.098, -1.136, -1.702]', 5.0]]
          #['MOVEJ', '[-0.649, 0.125, -1.715, -1.135, 0.247, -1.128, -1.797]', 5.0]
          #['MOVEJ', '[-0.848, 0.175, -1.676, -1.627, -0.097, -0.777, -1.704]', 5.0],

        self.motions['initScooping2'] = {}
        self.motions['initScooping2']['left'] = [['MOVES', '[-0.04, 0.0, -0.15, 0, 0.5, 0]', 3, 'self.bowl_frame']]
        self.motions['initScooping2']['right'] = [['MOVES', '[0.7, -0.15, -0.1, -3.1415, 0.0, 1.57]', 2.]]

        # only for training setup
        self.motions['initScooping2Random'] = {}
        self.motions['initScooping2Random']['left'] = []
        self.motions['initScooping2Random']['right'] = \
          [['MOVES', '[0.7+random.uniform(-0.1, 0.1), -0.15+random.uniform(-0.1, 0.1),-0.1+random.uniform(-0.1, 0.1), -3.1415, 0.0, 1.57]', 2.],]

        # [Y (from center of bowl away from Pr2, X (towards right gripper), Z (towards floor) , roll?, pitch (tilt downwards), yaw?]
        self.motions['runScooping'] = {}
        self.motions['runScoopingRight'] = {}
        self.motions['runScoopingLeft'] = {}
        self.motions['runScooping']['left'] = \
          [['MOVES', '[-0.05, 0.0,  0.045, 0, 0.6, 0]', 3, 'self.bowl_frame'],
           ['MOVES', '[ 0.05, 0.0,  0.03, 0, 0.8, 0]', 1, 'self.bowl_frame'],
           ['MOVES', '[ 0.05, 0.0,  -0.1, 0, 1.3, 0]', 3, 'self.bowl_frame'],]
           # ['MOVES', '[ 0.05, -0.03,  -0.1, 0, 1.3, 0]', 3, 'self.bowl_frame'],]
        
        ## Feeding motoins --------------------------------------------------------
        # It uses the l_gripper_spoon_frame aligned with mouth
        self.motions['initFeeding'] = {}
        self.motions['initFeeding']['left'] = [['MOVEJ', '[0.645, -0.198, 1.118, -2.121, 1.402, -0.242, 0.939]', 10.0],
           ['MOVES', '[0.705, 0.348, -0.029, 0.98, -1.565, -2.884]', 10.0]]
        self.motions['initFeeding']['right'] = [['MOVET', '[0.,0.1,-0.25,0.,-0.23,-0.1]', 5.0],
                                                ['PAUSE', 1.0]]
        ## self.motions['initFeeding']['right'] = [['MOVEJ', '[-1.0, 0.125, -1.715, -1.135, 0.247, -1.128, -1.797]', 5.0],]

        self.motions['runFeeding1'] = {}
        self.motions['runFeeding1']['left'] = [['MOVES', '[0.0, 0.0, -0.07, 0., 0., 0.]', 5., 'self.mouth_frame'], ['PAUSE', 1.0]]

        self.motions['runFeeding2'] = {}
        self.motions['runFeeding2']['left'] = [['MOVES', '[0.0, 0.0, 0.02, 0., 0., 0.]', 5., 'self.mouth_frame'], ['PAUSE', 0.5],
           ['MOVES', '[0.0, 0.0, -0.15, 0., 0., 0.]', 5., 'self.mouth_frame'],]
          
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
                return "Chose kinect head position"
            else:
                return "No kinect head position available! \n Code won't work! \n \
                Provide head position and try again!"
                
        elif task == "lookAtBowl":
            self.lookAt(self.bowl_frame)
            return "Completed to move head"
        
        elif task == "lookAtMouth":
            self.lookAt(self.mouth_frame, tag_base='head')                            
            return "Completed to move head"
        
        else:
            self.parsingMovements(self.motions[task][self.arm_name])
            return "Completed to execute "+task 
        
                
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
        self.mouth_frame_vision = PyKDL.Frame(M,p)

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
        ## Off set : 11 cm x direction, - 5 cm z direction. 
        pos_offset  = rospy.get_param('hrl_manipulation_task/sub_ee_pos_offset')        
        orient_offset = rospy.get_param('hrl_manipulation_task/sub_ee_orient_offset')        

        p = p + M*PyKDL.Vector(pos_offset['x'], pos_offset['y'], pos_offset['z'])
        M.DoRotX(orient_offset['rx'])
        M.DoRotY(orient_offset['ry'])
        M.DoRotZ(orient_offset['rz'])

        print 'Bowl frame:', p

        # 2.* add noise for random training 
        if addNoise:
            p = p + PyKDL.Vector(random.uniform(-0.1, 0.1),
                                 random.uniform(-0.1, 0.1),
                                 random.uniform(-0.1, 0.1))        
        
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

    def lookAt(self, target, tag_base='head'):

        t = time.time()
        head_frame  = rospy.get_param('hrl_manipulation_task/head_audio_frame')        
        headClient = actionlib.SimpleActionClient("/head_traj_controller/point_head_action",
                                                  pr2_controllers_msgs.msg.PointHeadAction)
        headClient.wait_for_server()
        rospy.logout('Connected to head control server')

        print '1:', time.time() - t
        t = time.time()

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

                   
        print '2:', time.time() - t
        t = time.time()

        pos.x = target.p.x()
        pos.y = target.p.y()
        pos.z = target.p.z()
                                        
        ps.point = pos
        head_goal_msg.target = ps
        
        headClient.send_goal(head_goal_msg)
        # headClient.wait_for_result() # TODO: This takes about 5 second -- very slow!
        # rospy.sleep(1.0)

        print '3:', time.time() - t

        return "Success"
        
    

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
    tool_id    = 1
    if opt.arm == 'l': verbose = False
    else: verbose = True
        

    rospy.init_node('arm_reacher_feeding_and_scooping')
    ara = armReachAction(d_robot, controller, arm, tool_id, verbose)
    rospy.spin()


