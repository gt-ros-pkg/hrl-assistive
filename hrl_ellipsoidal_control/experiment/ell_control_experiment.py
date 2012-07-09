#! /usr/bin/python

import sys
import numpy as np
import roslib
roslib.load_manifest('hrl_ellipsoidal_control')
roslib.load_manifest("pr2_controllers_msgs")

import rospy
import actionlib

from geometry_msgs.msg import PoseStamped
from pr2_controllers_msgs.msg import SingleJointPositionAction, SingleJointPositionGoal
from pr2_controllers_msgs.msg import Pr2GripperCommandAction, Pr2GripperCommandGoal

from hrl_generic_arms.ep_trajectory_controller import EPArmController
from hrl_generic_arms.pose_converter import PoseConverter
from hrl_pr2_arms.pr2_arm import PR2ArmCartesianPostureBase, PR2ArmJointTrajectory
from hrl_pr2_arms.pr2_arm import create_pr2_arm, PR2ArmJTransposeTask
from hrl_pr2_arms.pr2_controller_switcher import ControllerSwitcher
from hrl_ellipsoidal_control.ellipsoidal_interface_backend import EllipsoidalInterfaceBackend
from hrl_ellipsoidal_control.cartesian_interface_backend import CartesianInterfaceBackend

class EllipsoidControlExperiment(object):
    def __init__(self, interface_backend):
        self.backend = interface_backend()
        self.cart_arm = create_pr2_arm('l', PR2ArmJTransposeTask, 
                                       controller_name='%s_cart_jt_task', 
                                       end_link="%s_gripper_shaver45_frame")
        self.backend.set_arm(self.cart_arm)
        rospy.loginfo("[ell_control_experiment] EllipsoidControlExperiment ready.")

    def run_experiment(self):
        self.backend.disable_interface("Setting up arm.")
        pos, rot = PoseConverter.to_pos_rot(self.backend.controller.get_ell_frame())
        pos += rot * np.mat([0.3, -0.2, -0.05]).T
        _, tool_rot = PoseConverter.to_pos_rot([0, 0, 0], [np.pi, 0, 3./4. * np.pi])
        rot *= tool_rot
        ep_ac = EPArmController(self.cart_arm)
        ep_ac.execute_interpolated_ep((pos, rot), 6.)
        self.backend.set_arm(self.cart_arm)
        self.backend.enable_interface("Controller ready.")

def main():
    if sys.argv[1] == "ell":
        interface_backend = EllipsoidalInterfaceBackend
    elif sys.argv[1] == "cart":
        interface_backend = CartesianInterfaceBackend
    else:
        print "Argument must be either 'ell' or 'cart'"
    rospy.init_node("ell_control_experiment")
    ece = EllipsoidControlExperiment(interface_backend)
    ece.run_experiment()
    rospy.spin()
    ece.backend.print_statistics()
    if len(sys.argv) >= 3:
        ece.backend.save_statistics(sys.argv[2])

if __name__ == "__main__":
    main()
