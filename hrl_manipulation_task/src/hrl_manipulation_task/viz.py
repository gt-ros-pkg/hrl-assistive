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
# system
import rospy, roslib
import os, threading, copy

# util
import numpy as np
import PyKDL

import hrl_common_code_darpa_m3.visualization.draw_scene as ds
from pykdl_utils.kdl_kinematics import create_kdl_kin

# ROS message
from pr2_controllers_msgs.msg import JointTrajectoryControllerState
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import JointState

import warnings

class viz():

    def __init__(self, verbose=False):
        warnings.simplefilter("always", DeprecationWarning)
        self.verbose = verbose
        self.lock = threading.Lock()

        self.draw_ee_rf = ds.SceneDraw("hrl_anomaly_detection/ee_rf_field", "/torso_lift_link")
        self.draw_farm_rf = ds.SceneDraw("hrl_anomaly_detection/farm_rf_field", "/torso_lift_link")
        self.ee_pos = None
        self.forearm_pos = None

        self.initParams()
        self.initComms()
        print "Initialization complete"

        
    def initParams(self):
        '''
        Get parameters
        '''
        self.torso_frame = 'torso_lift_link'
        self.main_arm    = rospy.get_param('/hrl_manipulation_task/arm')
        if self.main_arm == 'l': self.sub_arm = 'r'
        else: self.sub_arm = 'l'

        self.main_ee_frame         = rospy.get_param('hrl_manipulation_task/main_ee_frame')
        self.main_ee_pos_offset    = rospy.get_param('hrl_manipulation_task/main_ee_pos_offset', None)        
        self.main_ee_orient_offset = rospy.get_param('hrl_manipulation_task/main_ee_orient_offset', None)
        self.main_joint_names      = rospy.get_param('/hrl_manipulation_task/main_joints')

        self.sub_ee_frame         = rospy.get_param('hrl_manipulation_task/sub_ee_frame')
        self.sub_ee_pos_offset    = rospy.get_param('hrl_manipulation_task/sub_ee_pos_offset', None)        
        self.sub_ee_orient_offset = rospy.get_param('hrl_manipulation_task/sub_ee_orient_offset', None)        
        self.sub_joint_names      = rospy.get_param('/hrl_manipulation_task/sub_joints')

        self.main_arm_kdl = create_kdl_kin(self.torso_frame, self.main_ee_frame)
        self.sub_arm_kdl  = create_kdl_kin(self.torso_frame, self.sub_ee_frame)

        p = PyKDL.Vector(self.main_ee_pos_offset['x'], \
                         self.main_ee_pos_offset['y'], \
                         self.main_ee_pos_offset['z'])
        M = PyKDL.Rotation.RPY(self.main_ee_orient_offset['rx'], self.main_ee_orient_offset['ry'], \
                               self.main_ee_orient_offset['rz'])
        self.main_offset = PyKDL.Frame(M,p)

        p = PyKDL.Vector(self.sub_ee_pos_offset['x'], \
                         self.sub_ee_pos_offset['y'], \
                         self.sub_ee_pos_offset['z'])
        M = PyKDL.Rotation.RPY(self.sub_ee_orient_offset['rx'], self.sub_ee_orient_offset['ry'], \
                               self.sub_ee_orient_offset['rz'])
        self.sub_offset = PyKDL.Frame(M,p)

    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''
        rospy.Subscriber('joint_states', JointState, self.joint_states_callback)


    #callback function: when a joint_states message arrives, save the values
    def joint_states_callback(self, msg):
        time_stamp   = msg.header.stamp
        jnt_name     = msg.name
        jnt_position = msg.position
        jnt_velocity = msg.velocity
        jnt_effort   = msg.effort

        main_positions = []
        main_velocities = []
        main_efforts = []
        
        for joint_name in self.main_joint_names:
        
            if joint_name in jnt_name:
                index    = jnt_name.index(joint_name)
                position = jnt_position[index]
                velocity = jnt_velocity[index]
                effort   = jnt_effort[index]
                #unless it's not found
            else:
                rospy.logerr("Joint %s not found!", (joint_name,))
                self.lock.release()
                return

            main_positions.append(position)
            main_velocities.append(velocity)
            main_efforts.append(effort)

        sub_positions = []
        sub_velocities = []
        sub_efforts = []
        
        for joint_name in self.sub_joint_names:
        
            if joint_name in jnt_name:
                index    = jnt_name.index(joint_name)
                position = jnt_position[index]
                velocity = jnt_velocity[index]
                effort   = jnt_effort[index]
                #unless it's not found
            else:
                rospy.logerr("Joint %s not found!", (joint_name,))
                self.lock.release()
                return

            sub_positions.append(position)
            sub_velocities.append(velocity)
            sub_efforts.append(effort)

        self.lock.acquire()
        
        self.time                = time_stamp.to_sec() #- self.init_time
        self.main_jnt_positions  = np.array([main_positions]).T
        self.sub_jnt_positions  = np.array([sub_positions]).T

        self.ee_pos, self.ee_quat         = self.getEEFrame(main_positions)
        self.target_pos, self.target_quat = self.getTargetFrame(sub_positions)
        self.forearm_pos                  = self.getForearmFrame(main_positions)


    def getEEFrame(self, joint_angles):

        mPose = self.main_arm_kdl.forward(joint_angles)

        p = PyKDL.Vector(mPose[0,3],mPose[1,3],mPose[2,3])
        M = PyKDL.Rotation(mPose[0,0],mPose[0,1],mPose[0,2], mPose[1,0],mPose[1,1],mPose[1,2], \
                           mPose[2,0],mPose[2,1],mPose[2,2] )
        poseFrame = PyKDL.Frame(M,p)*self.main_offset

        ee_pos  = np.array( [[poseFrame.p[0], poseFrame.p[1], poseFrame.p[2]]] ).T
        ee_quat = np.array( [[ poseFrame.M.GetQuaternion()[0], poseFrame.M.GetQuaternion()[1],\
                               poseFrame.M.GetQuaternion()[2], poseFrame.M.GetQuaternion()[3] ]] ).T
                    
        return ee_pos, ee_quat


    def getTargetFrame(self, joint_angles):

        mPose = self.sub_arm_kdl.forward(joint_angles)

        p = PyKDL.Vector(mPose[0,3],mPose[1,3],mPose[2,3])
        M = PyKDL.Rotation(mPose[0,0],mPose[0,1],mPose[0,2], mPose[1,0],mPose[1,1],mPose[1,2], \
                           mPose[2,0],mPose[2,1],mPose[2,2] )
        poseFrame = PyKDL.Frame(M,p)*self.sub_offset

        ee_pos  = np.array( [[poseFrame.p[0], poseFrame.p[1], poseFrame.p[2]]] ).T
        ee_quat = np.array( [[ poseFrame.M.GetQuaternion()[0], poseFrame.M.GetQuaternion()[1],\
                               poseFrame.M.GetQuaternion()[2], poseFrame.M.GetQuaternion()[3] ]] ).T
                    
        return ee_pos, ee_quat

    def getForearmFrame(self, joint_angles):
        mPose1 = self.main_arm_kdl.forward(joint_angles, end_link='l_forearm_link', \
                                 base_link='torso_lift_link')
        mPose2 = self.main_arm_kdl.forward(joint_angles, end_link='l_wrist_flex_link', \
                                 base_link='torso_lift_link')
        rf_pos = (np.array(mPose1[:3,3])+np.array(mPose2[:3,3]))/2.0
                                 
        return rf_pos 

        

    def pubReceptiveField(self):

        if self.forearm_pos is None: return

        ee_pos = copy.copy(self.ee_pos)
        forearm_pos = copy.copy(self.forearm_pos)
        
        pos_x = ee_pos[0,0]
        pos_y = ee_pos[1,0]
        pos_z = ee_pos[2,0]
        scale_x = scale_y = scale_z = 0.3
        start_id = 10
        self.draw_ee_rf.pub_body([pos_x, pos_y, pos_z],
                                 [0,0,0,1],
                                 [scale_x,scale_y,scale_z], 
                                 [0.0, 1.0, 0.0, 0.7], 
                                 start_id+0, 
                                 self.draw_ee_rf.Marker.SPHERE)

        pos_x = forearm_pos[0,0]
        pos_y = forearm_pos[1,0]
        pos_z = forearm_pos[2,0]
        scale_x = scale_y = scale_z = 0.3
        start_id = 11
        self.draw_farm_rf.pub_body([pos_x, pos_y, pos_z],
                                   [0,0,0,1],
                                   [scale_x,scale_y,scale_z], 
                                   [0.0, 1.0, 0.0, 0.7], 
                                   start_id+0, 
                                   self.draw_farm_rf.Marker.SPHERE)

    def run(self):
        rate = rospy.Rate(10) # 25Hz, nominally.    
        while not rospy.is_shutdown():

            self.pubReceptiveField()
            rate.sleep()


if __name__ == '__main__':
    warnings.simplefilter("always", DeprecationWarning)
    rospy.init_node('rf_viz')

    v = viz()
    v.run()
