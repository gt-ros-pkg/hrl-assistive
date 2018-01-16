#!/usr/bin/env python
#test client for joint_states_listener

import numpy as np
import matplotlib.pyplot as plt
import roslib
import rospy
import tf
import tf.transformations as tft
import threading
import time
import sys
import hrl_lib.circular_buffer as cb
from geometry_msgs.msg import Point32, Quaternion, PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String

FT_INI_THRESH = -6.0
FT_MAX_THRESH = -1.0

class JointStateListener(object):
    def __init__(self, joint_names):
        #Joint listener variables
        self.joint_names = joint_names
        self.j_lock = threading.RLock()
        self.header = None
        self.name = None
        self.position = None
        self.velocity = None
        self.effort = None

        self.ft_lock = threading.RLock()
        self.force = None
        self.torque = None

        self.f = {}
        self.f['x'] = []
        self.f['y'] = []
        self.f['z'] = []        
        self.f['tx'] = []
        self.f['ty'] = []
        self.f['tz'] = []
        self.f_buf = cb.CircularBuffer(10, (1,))
        self.wiping = False
        self.detect_stop = False
        self.plotf = False

        #Initialize offsets
        self.initialize_offsets()
        self.tool_offset = [.2156, 0., 0.]

        #Subscriber
        self.js_sub     = rospy.Subscriber('/joint_states', JointState, self.js_callback)
        self.ft_sub     = rospy.Subscriber('/ft/l_gripper_motor', WrenchStamped, self.ft_callback)
        self.status_sub = rospy.Subscriber('/manipulation_task/proceed', String, self.status_callback) 

        #Publisher
        self.pose_pub = rospy.Publisher('/hkim_random_test', PoseStamped, queue_size=10)
        self.stop_motion_pub = rospy.Publisher('/manipulation_task/InterruptAction', String, queue_size = 10)
        self.emergency_pub = rospy.Publisher('/manipulation_task/emergency', String, queue_size = 10)
        
        self.run2()

    def ft_callback(self, data):
        with self.ft_lock:
            self.force = data.wrench.force
            self.torque = data.wrench.torque

    def status_callback(self, data):
        if data.data == "Set: Wiping 2, Wiping 3, Wipe":
            self.detect_stop = True
        else:
            self.detect_stop = False
        if data.data == "Set: Wiping 3, Wipe, Retract":
            self.wiping = True
        else:
            if self.wiping:
                self.plotf = True
            self.wiping = False

    def run2(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            with self.ft_lock:
                force = self.force
                torque = self.torque
            if force is None:
                rate.sleep()
                continue
            #print force.y
            if force.y >= FT_INI_THRESH and self.detect_stop:
                self.stop_motion_pub.publish("found good thresh")
            if force.y >= FT_MAX_THRESH and self.wiping:
                self.emergency_pub.publish("STOP")
                self.wiping = False
                self.plotf = True
            elif self.wiping or self.detect_stop:
                self.f['x'].append(force.x)
                self.f['y'].append(force.y)
                self.f['z'].append(force.z)
                self.f['tx'].append(torque.x)
                self.f['ty'].append(torque.y)
                self.f['tz'].append(torque.z)
                if self.wiping:
                    self.f_buf.append(force.y)
                if len(self.f_buf) >= 10:
                    par = np.polyfit(xrange(10), self.f_buf, 1)
                    if par[0][0] >= .4:
                        if self.wiping:
                            self.f_buf = cb.CircularBuffer(10, (1,))
                            print "force spike detected for stop"
                            self.emergency_pub.publish("STOP")
                            self.wiping = False
                            self.plotf = True
            elif self.plotf:
                plt.subplot(321)
                plt.plot(self.f['x'], 'r')
                plt.subplot(322)
                plt.plot(self.f['y'], 'b')
                plt.subplot(323)
                plt.plot(self.f['z'], 'y')
                plt.subplot(324)
                plt.plot(self.f['tx'], 'r--')
                plt.subplot(325)
                plt.plot(self.f['ty'], 'b--')
                plt.subplot(326)
                plt.plot(self.f['tz'], 'y--')
                plt.show()
                self.plotf = False
                self.f['x'] = []
                self.f['y'] = []
                self.f['z'] = []        
                self.f['tx'] = []
                self.f['ty'] = []
                self.f['tz'] = []

            rate.sleep()

        


    #Manually load offset from joint links
    def initialize_offsets(self):
        self.links = []
        #fixed translation from joint to joint
        self.links.append((0., 0.188, 0.))
        self.links.append((0.1, 0., 0.))
        self.links.append((0., 0., 0.))        
        self.links.append((0.4, 0., 0.))
        self.links.append((0., 0., 0.))
        self.links.append((0.321, 0., 0.))
        self.links.append((0., 0., 0.))

        #angle that it rotates about
        self.angle_loc = [2, 1, 0, 1, 0, 1, 0]
        
    def js_callback(self, data):
        with self.j_lock:
            self.header = data.header
            self.name = data.name
            self.position = data.position
            self.velocity = data.velocity
            self.effort = data.effort

    def run(self):
        rate = rospy.Rate(3)
        while not rospy.is_shutdown():
            with self.j_lock:
                name = self.name
                position = self.position
                velocity = self.velocity
                effort = self.effort
            if name is None:
                rate.sleep()
                continue
            p = []; v = []; e = []
            for joint_name in self.joint_names:
                for (i, name) in enumerate(self.name):
                    if name == joint_name:
                        p.append(position[i])
                        v.append(velocity[i])
                        e.append(effort[i])
            #print self.joint_names
            #print "position", p
            #print "velocity", v
            #print "effort", e
            #pose = self.calc_pose(p, self.joint_names[-1])
            #self.pose_pub.publish(pose)
            force_pose = self.calc_pose(p, self.joint_names[-1], self.tool_offset) #calculate tool pose
            self.pose_pub.publish(force_pose)
            self.calc_force(force_pose, p, v, e)#, [self.joint_names[0]])#self.joint_names)
            rate.sleep()

    #assume force is applied toward +z axis of force_pose.
    def calc_force(self, force_pose, p, v, e):#, joint_names):
        """
        if not np.allclose(v, np.zeros(len(v))):
            return
        """
        total_force = 0.
        for i, joint_name in enumerate(self.joint_names):
            #calculate joint pose
            pose = self.calc_pose(p, joint_name)

            #calculate relative position from joint to force pose
            position = np.asarray(self.pose_to_tuple(pose)[0])
            force_position = np.asarray(self.pose_to_tuple(force_pose)[0])
            delta = force_position - position
            orientation = self.pose_to_tuple(pose)[1]
            matrix = np.matrix(tft.inverse_matrix(tft.quaternion_matrix(orientation))) *\
                     np.matrix(tft.translation_matrix(delta))
            vector1 = tft.translation_from_matrix(matrix)
            z_offset = tft.quaternion_matrix(self.pose_to_tuple(force_pose)[1])
            z_offset = np.matrix(z_offset) * np.matrix(tft.translation_matrix((0., 0., 1.)))
            delta2 = tft.translation_from_matrix(z_offset)
            orientation = self.pose_to_tuple(pose)[1]
            matrix = np.matrix(tft.inverse_matrix(tft.quaternion_matrix(orientation))) *\
                     np.matrix(tft.translation_matrix(delta2))
            vector2 = tft.translation_from_matrix(matrix)
            angle_loc = self.angle_loc[i]
            vector2[angle_loc] = 0
            vector1[angle_loc] = 0
            proj_vector1 = (np.dot(vector1, vector2) / np.dot(vector2, vector2)) * vector2
            vector1 = np.array(vector1) - np.array(proj_vector1)
            #vector2[angle_loc] = 0
            #vector1[angle_loc] = 0
            normal_vector = tft.unit_vector(np.cross(vector1, vector2))
            dist = np.sqrt(np.dot(vector1, vector1))
            if dist < 0.1 or 'wrist' in joint_name:
                continue
            force = (np.sum(normal_vector) * e[i]) / dist #np.sum gives u direction of force
            #print joint_name,  position, force_position, force, dist
            #print force
            total_force += force
        print "total ", total_force

    def pose_to_tuple(self, pose):
        position = (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
        orientation = (pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w)
        return (position, orientation)

    def calc_pose(self, angular_positions, joint_name, translation = None):
        matrices = []
        angle_loc = self.angle_loc
        for j, name in enumerate(self.joint_names):
            angles = [0., 0., 0.]
            angles[angle_loc[j]] = angular_positions[j]
            matrix = tft.euler_matrix(angles[0], angles[1], angles[2])
            for i in xrange(3):
                matrix[i][3] = self.links[j][i]
            matrices.append(matrix)
            if name == joint_name:
                break
        if translation is None:
            temp = tft.translation_matrix((0.0, 0.0, 0.0))
        else:
            temp = tft.translation_matrix(translation)
        for matrix in reversed(matrices):
            temp = np.matrix(matrix) * temp
        pose = PoseStamped()
        x, y, z, w = tft.quaternion_from_matrix(temp)
        pose.pose.orientation = Quaternion(x, y, z, w)
        x, y, z = tft.translation_from_matrix(temp)
        pose.pose.position = Point32(x, y, z)
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = '/torso_lift_link'
        return pose
            

if __name__ == "__main__":
    joint_names = ["l_shoulder_pan_joint",
                   "l_shoulder_lift_joint",
                   "l_upper_arm_roll_joint",
                   "l_elbow_flex_joint",
                   "l_forearm_roll_joint",
                   "l_wrist_flex_joint",
                   "l_wrist_roll_joint"]
    rospy.init_node('joint_state_listern')
    JointStateListener(joint_names)
        
            
