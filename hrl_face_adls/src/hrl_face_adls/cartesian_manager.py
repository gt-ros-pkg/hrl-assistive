#!/usr/bin/python

import numpy as np
import sys

import roslib
roslib.load_manifest("hrl_face_adls")

import rospy
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import TwistStamped, PoseStamped

from hrl_ellipsoidal_control.controller_base import CartesianStepController
from pykdl_utils.pr2_kin import kin_from_param
from hrl_generic_arms.pose_converter import PoseConverter
from hrl_pr2_arms.pr2_arm import create_pr2_arm, PR2ArmCartesianBase, PR2ArmJTransposeTask
from hrl_pr2_arms.pr2_controller_switcher import ControllerSwitcher
from face_adls_manager import async_call
from hrl_face_adls.srv import EnableCartController, EnableCartControllerResponse

VELOCITY = 0.03
_, FLIP_PERSPECTIVE_ROT = PoseConverter.to_pos_rot([0]*3, [0, 0, np.pi])

class CartesianControllerManager(object):
    def __init__(self, arm_char):
        self.arm_char = arm_char
        self.arm = None
        self.kin = None
        self.cart_ctrl = CartesianStepController()
        self.ctrl_switcher = ControllerSwitcher()
        self.command_move_sub = rospy.Subscriber("/face_adls/%s_cart_move" % arm_char, TwistStamped, 
                                                 async_call(self.command_move_cb))
        self.command_absolute_sub = rospy.Subscriber("/face_adls/%s_cart_absolute" % arm_char, 
                                                     PoseStamped, async_call(self.command_absolute_cb))
        self.adjust_posture_sub = rospy.Subscriber("/face_adls/%s_cart_posture" % arm_char, 
                                                   Float32, self.adjust_posture_cb)
        def enable_controller_cb(req):
            if req.enable:
                _, frame_rot = PoseConverter.to_pos_rot([0]*3, 
                                            [req.frame_rot.x, req.frame_rot.y, req.frame_rot.z])
                if req.velocity == 0:
                    req.velocity = 0.03
                success = self.enable_controller(req.end_link, req.ctrl_params, req.ctrl_name,
                                                 frame_rot, velocity=req.velocity)
            else:
                success = self.disable_controller()
            return EnableCartControllerResponse(success)
        self.controller_enabled_pub = rospy.Publisher('/face_adls/%s_cart_ctrl_enabled' % arm_char, 
                                                      Bool, latch=True)
        self.enable_controller_srv = rospy.Service("/face_adls/%s_enable_cart_ctrl" % arm_char, 
                                                   EnableCartController, enable_controller_cb)

    def enable_controller(self, end_link="%s_gripper_tool_frame",
                          ctrl_params="$(find hrl_face_adls)/params/%s_jt_task_tool.yaml",
                          ctrl_name="%s_cart_jt_task_tool",
                          frame_rot=FLIP_PERSPECTIVE_ROT, velocity=0.03):
#frame_rot=np.mat(np.eye(3))):
        rospy.loginfo("[cartesian_manager] Enabling %s controller with end link %s" %
                      (ctrl_name, end_link))

        try:
            if '%s' in end_link:
                end_link = end_link % self.arm_char
            if '%s' in ctrl_params:
                ctrl_params = ctrl_params % self.arm_char
            if '%s' in ctrl_name:
                ctrl_name = ctrl_name % self.arm_char
            self.ctrl_switcher.carefree_switch(self.arm_char, ctrl_name, ctrl_params, reset=False)
            rospy.sleep(0.2)
            cart_arm = create_pr2_arm(self.arm_char, PR2ArmJTransposeTask, 
                                      controller_name=ctrl_name, 
                                      end_link=end_link, timeout=5)
            self.cart_ctrl.set_arm(cart_arm)
            self.arm = cart_arm
            rospy.sleep(0.1)
            self.cur_posture_angle = self.arm.get_joint_angles()[2]
            self.adjust_posture_cb(Float32(0.))
            self.frame_rot = frame_rot
            self.velocity = velocity
            self.controller_enabled_pub.publish(True)
        except Exception as e:
            print e
            return False
        rospy.loginfo("[cartesian_manager] Successfully enabled %s controller with end link %s" %
                      (ctrl_name, end_link))
        return True

    def disable_controller(self):
        rospy.loginfo("[cartesian_manager] Disabling cartesian controller.")
        self.cart_ctrl.set_arm(None)
        self.arm = None
        self.controller_enabled_pub.publish(False)
        return True

    def command_move_cb(self, msg):
        if self.arm is None:
            rospy.logwarn("[cartesian_manager] Cartesian controller not enabled.")
            return
        self.cart_ctrl.stop_moving(wait=True)
        if msg.header.frame_id == "":
            msg.header.frame_id = "torso_lift_link"
        if self.kin is None or msg.header.frame_id not in self.kin.get_segment_names():
            self.kin = kin_from_param("torso_lift_link", msg.header.frame_id)
        torso_pos_ep, torso_rot_ep = self.arm.get_ep()
        torso_B_ref = self.kin.forward_filled(base_segment="torso_lift_link", 
                                              target_segment=msg.header.frame_id)
        _, torso_rot_ref = PoseConverter.to_pos_rot(torso_B_ref)
        torso_rot_ref *= self.frame_rot
        ref_pos_off, ref_rot_off = PoseConverter.to_pos_rot(msg)
        change_pos = torso_rot_ep.T * torso_rot_ref * ref_pos_off
        change_pos_xyz = change_pos.T.A[0]
        ep_rot_ref = torso_rot_ep.T * torso_rot_ref
        change_rot = ep_rot_ref * ref_rot_off * ep_rot_ref.T
        _, change_rot_rpy = PoseConverter.to_pos_euler(np.mat([0]*3).T, change_rot)
        rospy.loginfo("[cartesian_manager] Command relative movement." + 
                      str((change_pos_xyz, change_rot_rpy)))
        self.cart_ctrl.execute_cart_move((change_pos_xyz, change_rot_rpy), ((0, 0, 0), 0), 
                                         velocity=self.velocity, blocking=True)

    def command_absolute_cb(self, msg):
        if self.arm is None:
            rospy.logwarn("[cartesian_manager] Cartesian controller not enabled.")
            return
        self.cart_ctrl.stop_moving(wait=True)
        if msg.header.frame_id == "":
            msg.header.frame_id = "torso_lift_link"
        if self.kin is None or msg.header.frame_id not in self.kin.get_segment_names():
            self.kin = kin_from_param("torso_lift_link", msg.header.frame_id)
        torso_pos_ep, torso_rot_ep = self.arm.get_ep()
        torso_B_ref = self.kin.forward_filled(base_segment="torso_lift_link", 
                                              target_segment=msg.header.frame_id)
        ref_B_goal = PoseConverter.to_homo_mat(msg)
        torso_B_goal = torso_B_ref * ref_B_goal

        change_pos, change_rot = PoseConverter.to_pos_rot(torso_B_goal)
        change_pos_xyz = change_pos.T.A[0]
        rospy.loginfo("[cartesian_manager] Command absolute movement." + 
                      str((change_pos_xyz, change_rot)))
        self.cart_ctrl.execute_cart_move((change_pos_xyz, change_rot), ((1, 1, 1), 1), 
                                         velocity=self.velocity, blocking=True)

    def adjust_posture_cb(self, msg):
        rospy.loginfo("[cartesian_manager] Adjusting posture")
        new_posture = [None] * 7
        self.cur_posture_angle += msg.data
        new_posture[2] = self.cur_posture_angle
        self.arm.set_posture(new_posture)

def main():
    rospy.init_node("cartesian_manager")
    arm_char = sys.argv[1]
    assert arm_char in ['r', 'l']
    cart_man = CartesianControllerManager(arm_char)
    if False:
        cart_man.enable_controller(
            end_link="%s_gripper_tool_frame" % arm_char, 
            ctrl_name="%s_cart_jt_task_tool" % arm_char,
            ctrl_params="$(find hrl_face_adls)/params/%s_jt_task_tool.yaml" % arm_char,
            frame_rot=FLIP_PERSPECTIVE_ROT)
        rospy.sleep(1)
        #t = PoseConverter.to_twist_stamped_msg("l_gripper_tool_frame", (-0.15, 0, 0), (0, 0, 0))
        #t = PoseConverter.to_twist_stamped_msg("torso_lift_link", (0, 0, 0), (-np.pi/6, 0, 0))
        t = PoseConverter.to_twist_stamped_msg("torso_lift_link", (0, 0.1, 0), (0, 0, 0))
        cart_man.command_move_cb(t)
        return

    rospy.spin()

if __name__ == "__main__":
    main()
