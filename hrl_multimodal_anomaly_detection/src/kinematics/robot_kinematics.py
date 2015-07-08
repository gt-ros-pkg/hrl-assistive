#!/usr/bin/env python

import rospy
import threading
from sensor_msgs.msg import JointState

class robot_kinematics(threading.Thread):
    def __init__(self, tfListener):
        super(robot_kinematics, self).__init__()
        self.daemon = True
        self.cancelled = False
        self.arm = 'r'
        self.init_time = 0.
        self.jstate_lock = threading.RLock() ## joint state lock
        self.l_end_effector = 'l_gripper_spoon_frame'
        self.r_end_effector = 'r_gripper_tool_frame'

        self.joint_angles = []
        self.joint_velocities = []

        self.time_data  = []
        self.joint_data = []
        self.l_end_effector_pos = []
        self.l_end_effector_quat = []
        self.r_end_effector_pos = []
        self.r_end_effector_quat = []

        self.tf_listener = tfListener

        groups = rospy.get_param('/right/haptic_mpc/groups' )
        for group in groups:
            if group['name'] == 'left_arm_joints' and self.arm == 'l':
                self.joint_names_list = group['joints']
            elif group['name'] == 'right_arm_joints' and self.arm == 'r':
                self.joint_names_list = group['joints']

        self.jointSub = rospy.Subscriber("/joint_states", JointState, self.jointStatesCallback)

    def jointStatesCallback(self, data):
        joint_angles = []
        ## joint_efforts = []
        joint_vel = []
        jt_idx_list = [0]*len(self.joint_names_list)
        for i, jt_nm in enumerate(self.joint_names_list):
            jt_idx_list[i] = data.name.index(jt_nm)

        for i, idx in enumerate(jt_idx_list):
            if data.name[idx] != self.joint_names_list[i]:
                raise RuntimeError('joint angle name does not match.')
            joint_angles.append(data.position[idx])
            ## joint_efforts.append(data.effort[idx])
            joint_vel.append(data.velocity[idx])

        with self.jstate_lock:
            self.joint_angles  = joint_angles
            ## self.joint_efforts = joint_efforts
            self.joint_velocities = joint_vel


    def run(self):
        """Overloaded Thread.run, runs the update
        method once per every xx milliseconds."""

        rate = rospy.Rate(1000) # 25Hz, nominally.
        while not self.cancelled:
            self.log()
            rate.sleep()

    def log(self):

        self.time_data.append(rospy.get_time()-self.init_time)
        self.joint_data.append(self.joint_angles)
        (pos_l, quat_l) = self.tf_listener.lookupTransform('/torso_lift_link', self.l_end_effector, rospy.Time(0))
        (pos_r, quat_r) = self.tf_listener.lookupTransform('/torso_lift_link', self.r_end_effector, rospy.Time(0))
        self.l_end_effector_pos.append(pos_l)
        self.l_end_effector_quat.append(quat_l)
        self.r_end_effector_pos.append(pos_r)
        self.r_end_effector_quat.append(quat_r)

    def cancel(self):
        """End this timer thread"""
        self.cancelled = True
        self.jointSub.unregister()
        rospy.sleep(1.0)

