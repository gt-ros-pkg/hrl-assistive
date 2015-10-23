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
import rospy
import roslib
roslib.load_manifest('hrl_manipulation_task')
import os, threading, copy

# util
import numpy as np
import PyKDL

# ROS message
import tf
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

            self.ee_pos    = None
            self.ee_quat   = None
            self.jnt_name     = None
            self.jnt_position = None
            self.jnt_velocity = None
            self.jnt_effort   = None
            self.jnt_positions  = None
            self.jnt_velocities = None
            self.jnt_efforts    = None

            # Declare containers
            self.time_data = []
            self.kinematics_ee_pos  = None
            self.kinematics_ee_quat = None
            self.kinematics_jnt_pos = None
            self.kinematics_jnt_vel = None
            self.kinematics_jnt_eff = None
            self.Kinematics_target_pos = None
            self.Kinematics_target_quat = None
            
    def initParams(self):
        '''
        Get parameters
        '''
        self.torso_frame = 'torso_lift_link'
        self.ee_frame    = rospy.get_param('/hrl_manipulation_task/end_effector_frame')
        self.joint_names = rospy.get_param('/hrl_manipulation_task/joints')

        self.target_frame = rospy.get_param('/hrl_manipulation_task/target_frame', None)
        self.target_pos_offset    = rospy.get_param('hrl_manipulation_task/target_pos_offset', None)        
        self.target_orient_offset = rospy.get_param('hrl_manipulation_task/target_orient_offset', None)        

        
    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''
        # tf
        try:
            self.tf_lstnr = tf.TransformListener()
        except rospy.ServiceException, e:
            rospy.loginfo("ServiceException caught while instantiating a TF listener. Seems to be normal")

        rospy.Subscriber('joint_states', JointState, self.joint_states_callback)


    #callback function: when a joint_states message arrives, save the values
    def joint_states_callback(self, msg):
        self.lock.acquire()
        self.jnt_name     = msg.name
        self.jnt_position = msg.position
        self.jnt_velocity = msg.velocity
        self.jnt_effort   = msg.effort
        self.counter += 1
        self.lock.release()


    #returns (found, position, velocity, effort) for the joint joint_name 
    #(found is 1 if found, 0 otherwise)
    def return_joint_state(self):

        #no messages yet
        if self.jnt_name == []:
            rospy.logerr("No robot_state messages received!\n")
            return (0, 0., 0., 0.)

        positions = []
        velocities = []
        efforts = []
            
        #return info for this joint
        self.lock.acquire()
        
        for joint_name in self.joint_names:
        
            if joint_name in self.jnt_name:
                index    = self.jnt_name.index(joint_name)
                position = self.jnt_position[index]
                velocity = self.jnt_velocity[index]
                effort   = self.jnt_effort[index]

                #unless it's not found
            else:
                rospy.logerr("Joint %s not found!", (joint_name,))
                self.lock.release()
                return (0, 0., 0., 0.)

            positions.append(position)
            velocities.append(velocity)
            efforts.append(effort)
            
        self.lock.release()

        self.jnt_positions  = np.array([positions]).T
        self.jnt_velocities = np.array([velocities]).T
        self.jnt_efforts    = np.array([efforts]).T
        
        return (self.jnt_positions, self.jnt_velocities, self.jnt_efforts)

                                                                                                            
    def getEEFrame(self):
        try:
            self.tf_lstnr.waitForTransform(self.torso_frame, self.ee_frame, rospy.Time(0), \
                                           rospy.Duration(1.0))
        except:
            self.tf_lstnr.waitForTransform(self.torso_frame, self.ee_frame, rospy.Time(0), \
                                           rospy.Duration(1.0))
                                           
        [ee_pos, ee_quat] = self.tf_lstnr.lookupTransform(self.torso_frame, self.ee_frame, rospy.Time(0))  
        self.ee_pos = np.array([[ee_pos[0], ee_pos[1], ee_pos[2]]]).T
        self.ee_quat = np.array([[ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3]]]).T
                    
        return (self.ee_pos, self.ee_quat)


    def getTargetFrame(self):

        try:
            self.tf_lstnr.waitForTransform(self.torso_frame, self.target_frame, rospy.Time(0), \
                                           rospy.Duration(1.0))
        except:
            self.tf_lstnr.waitForTransform(self.torso_frame, self.target_frame, rospy.Time(0), \
                                           rospy.Duration(1.0))

        [pos, quat] = self.tf_lstnr.lookupTransform(self.torso_frame, self.target_frame, rospy.Time(0))  

        p = PyKDL.Vector(pos[0], pos[1], pos[2])
        M = PyKDL.Rotation.Quaternion(quat[0],quat[1],quat[2],quat[3])

        p = p + M*PyKDL.Vector(self.target_pos_offset['x'], self.target_pos_offset['y'], \
                               self.target_pos_offset['z'])
        M.DoRotX(self.target_orient_offset['rx'])
        M.DoRotY(self.target_orient_offset['ry'])
        M.DoRotZ(self.target_orient_offset['rz'])        
        
        target_pos  = np.array( [[p[0], p[1], p[2]]] ).T
        target_quat = np.array( [[ M.GetQuaternion()[0], M.GetQuaternion()[1],\
                                   M.GetQuaternion()[2], M.GetQuaternion()[3] ]] ).T

        return target_pos, target_quat
        
        
    def run(self):
        """Overloaded Thread.run, runs the update
        method once per every xx milliseconds."""
        while not self.cancelled:
            if self.isReset:
                ee_pos, ee_quat           = self.getEEFrame()
                jnt_pos, jnt_vel, jnt_eff = self.return_joint_state()
                target_pos, target_quat   = self.getTargetFrame()
                
                if self.counter > self.counter_prev:
                    self.counter_prev = self.counter
                else:
                    continue
                
                self.time_data.append(rospy.get_time() - self.init_time)
                self.lock.acquire()
                
                if self.kinematics_ee_pos is None:
                    self.kinematics_ee_pos  = ee_pos
                    self.kinematics_ee_quat = ee_quat
                    self.kinematics_jnt_pos = jnt_pos 
                    self.kinematics_jnt_vel = np.zeros((len(jnt_vel),1))
                    self.kinematics_jnt_eff = jnt_eff

                    self.Kinematics_target_pos  = target_pos
                    self.Kinematics_target_quat = target_quat
                    
                else:
                    self.kinematics_ee_pos  = np.hstack([self.kinematics_ee_pos, ee_pos])
                    self.kinematics_ee_quat = np.hstack([self.kinematics_ee_quat, ee_quat])
                    self.kinematics_jnt_vel = np.hstack([self.kinematics_jnt_vel, jnt_pos - self.kinematics_jnt_pos[:,-1:] ]) #delta
                    self.kinematics_jnt_pos = np.hstack([self.kinematics_jnt_pos, jnt_pos])
                    self.kinematics_jnt_eff = np.hstack([self.kinematics_jnt_eff, jnt_eff])

                    self.Kinematics_target_pos = np.hstack([self.Kinematics_target_pos, target_pos])
                    self.Kinematics_target_quat= np.hstack([self.Kinematics_target_quat, target_quat])
                    
                self.lock.release()
                
                
    def cancel(self):
        """End this timer thread"""
        self.cancelled = True
        self.isReset = False


    def reset(self, init_time):
        self.init_time = init_time
        self.isReset = True

        # Reset containers
        self.time_data = []
        self.kinematics_ee_pos  = None
        self.kinematics_ee_quat = None
        self.kinematics_jnt_pos = None
        self.kinematics_jnt_vel = None
        self.kinematics_jnt_eff = None
        self.Kinematics_target_pos = None
        self.Kinematics_target_quat = None

        self.counter = 0
        self.counter_prev = 0
        

    def isReady(self):

        self.getEEFrame()
        self.return_joint_state()

        if self.ee_pos is not None and self.ee_quat is not None and \
          self.jnt_position is not None:
          return True
        else:
          return False
        
