#! /usr/bin/python
        
import sys

import roslib
roslib.load_manifest('hrl_ellipsoidal_control')
from hrl_ellipsoidal_control.ellipsoid_controller import EllipsoidController

class CartesianNormalizedController(EllipsoidController):
    # TODO overwrite execute_cart_move

    def _get_ell_equiv_dist(self, ell_change_ep, ell_abs_sel, orient_quat, velocity):
        ell_f, rot_mat_f = self._parse_ell_move(ell_change_ep, ell_abs_sel, orient_quat)
        traj = self._create_ell_trajectory(ell_f, rot_mat_f, orient_quat, velocity)
        dist = np.linalg.norm(traj[0][0] - traj[-1][0])
        return dist


def main():
    rospy.init_node("cartesian_controller", sys.argv)
    cart_arm = create_pr2_arm('l', PR2ArmJTransposeTask, 
                              controller_name='%s_cart_jt_task', 
                              end_link="%s_gripper_shaver45_frame", timeout=0)

    rospy.sleep(1)
    cart_controller = CartesianController(cart_arm)
    rospy.spin()
    

if __name__ == "__main__":
    main()
