#!/usr/bin/env python
import sys
import numpy as np

import rospy

from hrl_ellipsoidal_control.ellipsoid_controller import EllipsoidController
from hrl_pr2_arms.ep_arm_base import create_ep_arm
from hrl_pr2_arms.pr2_arm_jt import PR2ArmJTranspose


class CartesianNormalizedController(EllipsoidController):
    # TODO overwrite execute_cart_move

    def _get_ell_equiv_dist(self, ell_change_ep, ell_abs_sel, orient_quat, velocity):
        ell_f, rot_mat_f = self._parse_ell_move(ell_change_ep, ell_abs_sel, orient_quat)
        traj = self._create_ell_trajectory(ell_f, rot_mat_f, orient_quat, velocity)
        dist = np.linalg.norm(traj[0][0] - traj[-1][0])
        return dist


def main():
    rospy.init_node("cartesian_controller", sys.argv)
    cart_arm = create_ep_arm('l', PR2ArmJTranspose,
                             controller_name='%s_cart_jt_task',
                             end_link="%s_gripper_shaver45_frame",
                             timeout=0)

    rospy.sleep(1)
    cart_controller = CartesianController(cart_arm)
    rospy.spin()
