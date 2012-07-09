#! /usr/bin/python

import numpy as np
import cPickle as pkl

import roslib
roslib.load_manifest("hrl_ellipsoidal_control")
import rospy
import tf.transformations as tf_trans
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse

from hrl_pr2_arms.pr2_arm import PR2ArmCartesianPostureBase, PR2ArmJointTrajectory
from hrl_pr2_arms.pr2_arm import create_pr2_arm, PR2ArmJTransposeTask
from hrl_pr2_arms.pr2_controller_switcher import ControllerSwitcher
from hrl_ellipsoidal_control.ellipsoid_controller import EllipsoidController
from hrl_ellipsoidal_control.interface_backend import ControllerInterfaceBackend
from hrl_ellipsoidal_control.ellipsoidal_parameters import *

class EllipsoidalInterfaceBackend(ControllerInterfaceBackend):
    def __init__(self, use_service=True):
        super(EllipsoidalInterfaceBackend, self).__init__("Ellipsoid Controller", 
                                                          use_service=use_service)
        self.controller = EllipsoidController()
        self.button_distances = {}
        self.button_times = {}
        for button in ell_trans_params.keys() + ell_rot_params.keys() + ["reset_rotation"]:
            self.button_distances[button] = []
            self.button_times[button] = []

    def set_arm(self, cart_arm):
        self.cart_arm = cart_arm
        self.controller.set_arm(self.cart_arm)

    def run_controller(self, button_press):
        start_pos, _ = self.cart_arm.get_ep()
        start_time = rospy.get_time()
        quat_gripper_rot = tf_trans.quaternion_from_euler(np.pi, 0, 0)
        if button_press in ell_trans_params:
            change_trans_ep = ell_trans_params[button_press]
            self.controller.execute_ell_move((change_trans_ep, (0, 0, 0)), ((0, 0, 0), 0), 
                                                 quat_gripper_rot, ELL_LOCAL_VEL)
        elif button_press in ell_rot_params:
            change_rot_ep = ell_rot_params[button_press]
            self.controller.execute_ell_move(((0, 0, 0), change_rot_ep), ((0, 0, 0), 0), 
                                                 quat_gripper_rot, ELL_ROT_VEL)
        elif button_press == "reset_rotation":
            self.controller.execute_ell_move(((0, 0, 0), np.mat(np.eye(3))), ((0, 0, 0), 1), 
                                                 quat_gripper_rot, ELL_ROT_VEL)
        end_pos, _ = self.cart_arm.get_ep()
        end_time = rospy.get_time()
        dist = np.linalg.norm(end_pos - start_pos)
        time_diff = end_time - start_time
        print "%s, dist: %.3f, time_diff: %.2f" % (button_press, dist, time_diff)
        self.button_distances[button_press].append(dist)
        self.button_times[button_press].append(time_diff)

    def print_statistics(self):
        print self.button_distances
        print self.button_times
        print "MEANS:"
        for button in ell_trans_params.keys() + ell_rot_params.keys() + ["reset_rotation"]:
            print "%s, avg dist: %.3f, avg_time: %.3f, num_presses: %d" % (
                    button, 
                    np.mean(self.button_distances[button]), 
                    np.mean(self.button_times[button]), 
                    len(self.button_distances[button]))

    def save_statistics(self, filename):
        button_mean_distances = {}
        button_mean_times = {}
        button_num_presses = {}
        for button in ell_trans_params.keys() + ell_rot_params.keys() + ["reset_rotation"]:
            button_mean_distances[button] = np.mean(self.button_distances[button])
            button_mean_times[button] = np.mean(self.button_times[button])
            button_num_presses[button] = len(self.button_distances[button])
        stats = {
            "button_distances"      : self.button_distances,
            "button_times"          : self.button_times,
            "button_mean_distances" : button_mean_distances,
            "button_mean_times"     : button_mean_times,
            "button_num_presses"    : button_num_presses
        }
        f = file(filename, "w")
        pkl.dump(stats, f)
        f.close()

def main():
    rospy.init_node("ellipsoidal_controller_backend")


    cart_arm = create_pr2_arm('l', PR2ArmJTransposeTask, 
                              controller_name='%s_cart_jt_task', 
                              end_link="%s_gripper_shaver45_frame")

    ell_backend = EllipsoidalInterfaceBackend(cart_arm)
    ell_backend.disable_interface("Setting up arm.")

    if True:
        ctrl_switcher = ControllerSwitcher()
        ctrl_switcher.carefree_switch('l', '%s_arm_controller', None)
        rospy.sleep(1)
        joint_arm = create_pr2_arm('l', PR2ArmJointTrajectory)

        setup_angles = [0, 0, np.pi/2, -np.pi/2, -np.pi, -np.pi/2, -np.pi/2]
        joint_arm.set_ep(setup_angles, 5)
        rospy.sleep(5)

        ctrl_switcher.carefree_switch('l', '%s_cart_jt_task', 
                                      "$(find hrl_rfh_fall_2011)/params/l_jt_task_shaver45.yaml") 
        rospy.sleep(1)

    if True:
        rospy.sleep(1)
        setup_angles = [0, 0, np.pi/2, -np.pi/2, -np.pi, -np.pi/2, -np.pi/2]
        cart_arm.set_posture(setup_angles)
        cart_arm.set_gains([200, 800, 800, 80, 80, 80], [15, 15, 15, 1.2, 1.2, 1.2])
        ell_backend.controller.reset_arm_orientation(8)

    ell_backend.enable_interface("Controller ready.")
    rospy.spin()
    ell_backend.print_statistics()

if __name__ == "__main__":
    main()
