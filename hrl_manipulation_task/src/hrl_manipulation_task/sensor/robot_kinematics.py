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

from pykdl_utils.kdl_kinematics import create_kdl_kin

# util
import numpy as np
import PyKDL

# ROS message
from std_msgs.msg import Bool, Empty, Int32, Int64, Float32, Float64, String
from sensor_msgs.msg import JointState

class robot_kinematics(threading.Thread):
    def __init__(self, verbose=False):
        super(robot_kinematics, self).__init__()        
        self.daemon = True
        self.cancelled = False
        self.isReset = False
        self.verbose = verbose

        self.lock = threading.Lock()

        self.initVars()
        self.initParams()
        self.initComms()
        
        if self.verbose: print "Robot_kinematics>> initialization complete"

            
    def initVars(self):
        with self.lock:
            self.init_time = 0.0        
            self.counter = 0
            self.counter_prev = 0

            self.enable_log = False
            
            # instant data
            self.time = None
            self.main_jnt_positions = None
            self.main_jnt_velocities = None
            self.main_jnt_efforts   = None

            self.sub_jnt_positions = None
            self.sub_jnt_velocities = None
            self.sub_jnt_efforts   = None
                        
            self.ee_pos    = None
            self.ee_quat   = None
            self.target_pos    = None
            self.target_quat   = None

            # Declare containers
            self.time_data = []
            self.kinematics_ee_pos  = None
            self.kinematics_ee_quat = None
            self.kinematics_main_jnt_pos = None
            self.kinematics_main_jnt_vel = None
            self.kinematics_main_jnt_eff = None
            self.kinematics_target_pos = None
            self.kinematics_target_quat = None
            
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
        self.main_jnt_velocities = np.array([main_velocities]).T
        self.main_jnt_efforts    = np.array([main_efforts]).T

        self.sub_jnt_positions  = np.array([sub_positions]).T
        self.sub_jnt_velocities = np.array([sub_velocities]).T
        self.sub_jnt_efforts    = np.array([sub_efforts]).T

        self.ee_pos, self.ee_quat         = self.getEEFrame(main_positions)
        self.target_pos, self.target_quat = self.getTargetFrame(sub_positions)

        if self.enable_log:
            self.time_data.append(self.time)

            if self.kinematics_ee_pos is None:
                self.kinematics_ee_pos  = self.ee_pos
                self.kinematics_ee_quat = self.ee_quat
                self.kinematics_main_jnt_pos = self.main_jnt_positions 
                self.kinematics_main_jnt_vel = np.zeros((len(self.main_jnt_positions),1))
                self.kinematics_main_jnt_eff = self.main_jnt_efforts

                self.kinematics_target_pos  = self.target_pos
                self.kinematics_target_quat = self.target_quat

            else:
                self.kinematics_ee_pos  = np.hstack([self.kinematics_ee_pos, self.ee_pos])
                self.kinematics_ee_quat = np.hstack([self.kinematics_ee_quat, self.ee_quat])
                self.kinematics_main_jnt_vel = np.hstack([self.kinematics_main_jnt_vel, \
                                                          self.kinematics_main_jnt_pos - \
                                                          self.kinematics_main_jnt_pos[:,-1:] ]) #delta
                self.kinematics_main_jnt_pos = np.hstack([self.kinematics_main_jnt_pos, self.main_jnt_positions])
                self.kinematics_main_jnt_eff = np.hstack([self.kinematics_main_jnt_eff, self.main_jnt_efforts])

                self.kinematics_target_pos = np.hstack([self.kinematics_target_pos, self.target_pos])
                self.kinematics_target_quat= np.hstack([self.kinematics_target_quat, self.target_quat])
                
        self.counter += 1
        self.lock.release()

                                                                                                            
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
        poseFrame = PyKDL.Frame(M,p)*self.main_offset

        ee_pos  = np.array( [[poseFrame.p[0], poseFrame.p[1], poseFrame.p[2]]] ).T
        ee_quat = np.array( [[ poseFrame.M.GetQuaternion()[0], poseFrame.M.GetQuaternion()[1],\
                               poseFrame.M.GetQuaternion()[2], poseFrame.M.GetQuaternion()[3] ]] ).T
                    
        return ee_pos, ee_quat

                
    ## def run(self):
    ##     """Overloaded Thread.run, runs the update
    ##     method once per every xx milliseconds."""
    ##     rate = rospy.Rate(20)
    ##     while not self.cancelled and not rospy.is_shutdown():
    ##         if self.isReset:
                
    ##             if self.counter > self.counter_prev:
    ##                 self.counter_prev = self.counter
    ##             else:
    ##                 continue
                
    ##             self.time_data.append(rospy.get_rostime().to_sec() - self.init_time)
    ##             self.lock.acquire()
                
    ##             if self.kinematics_ee_pos is None:
    ##                 self.kinematics_ee_pos  = self.ee_pos
    ##                 self.kinematics_ee_quat = self.ee_quat
    ##                 self.kinematics_main_jnt_pos = self.main_jnt_positions 
    ##                 self.kinematics_main_jnt_vel = np.zeros((len(self.main_jnt_positions),1))
    ##                 self.kinematics_main_jnt_eff = self.main_jnt_efforts

    ##                 self.kinematics_target_pos  = self.target_pos
    ##                 self.kinematics_target_quat = self.target_quat
                    
    ##             else:
    ##                 self.kinematics_ee_pos  = np.hstack([self.kinematics_ee_pos, self.ee_pos])
    ##                 self.kinematics_ee_quat = np.hstack([self.kinematics_ee_quat, self.ee_quat])
    ##                 self.kinematics_main_jnt_vel = np.hstack([self.kinematics_main_jnt_vel, self.kinematics_main_jnt_pos - \
    ##                                                      self.kinematics_main_jnt_pos[:,-1:] ]) #delta
    ##                 self.kinematics_main_jnt_pos = np.hstack([self.kinematics_main_jnt_pos, self.main_jnt_positions])
    ##                 self.kinematics_main_jnt_eff = np.hstack([self.kinematics_main_jnt_eff, self.main_jnt_efforts])

    ##                 self.kinematics_target_pos = np.hstack([self.kinematics_target_pos, self.target_pos])
    ##                 self.kinematics_target_quat= np.hstack([self.kinematics_target_quat, self.target_quat])
                    
    ##             self.lock.release()
    ##         rate.sleep()
                
                
    ## def cancel(self):
    ##     """End this timer thread"""
    ##     self.cancelled = True
    ##     self.isReset = False


    def reset(self, init_time):
        self.init_time = init_time
        self.isReset = True

        # Reset containers
        self.time_data = []
        self.kinematics_ee_pos  = None
        self.kinematics_ee_quat = None
        self.kinematics_main_jnt_pos = None
        self.kinematics_main_jnt_vel = None
        self.kinematics_main_jnt_eff = None
        self.kinematics_target_pos = None
        self.kinematics_target_quat = None

        self.counter = 0
        self.counter_prev = 0
        

    def isReady(self):

        if self.ee_pos is not None and self.ee_quat is not None:
          return True
        else:
          return False
        
