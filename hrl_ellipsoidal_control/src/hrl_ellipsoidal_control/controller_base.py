
import numpy as np
import sys
from threading import Lock

import roslib
roslib.load_manifest('hrl_ellipsoidal_control')

import rospy
from geometry_msgs.msg import PoseStamped
import tf.transformations as tf_trans

from hrl_pr2_arms.pr2_arm import create_pr2_arm, PR2ArmCartesianBase, PR2ArmJTransposeTask
from hrl_generic_arms.pose_converter import PoseConverter

def min_jerk_traj(n):
    return [(10 * t**3 - 15 * t**4 + 6 * t**5)
            for t in np.linspace(0, 1, n)]

class CartTrajController(object):
    def __init__(self):
        self._moving_lock = Lock()
        self._timer = None

    def stop_moving(self, wait=False):
        self._stop_moving = True
        if wait:
            self.wait_until_stopped()

    def is_moving(self):
        return self._timer is not None

    def wait_until_stopped(self):
        if self._timer is not None:
            self._timer.join()

    def execute_cart_traj(self, cart_arm, traj, time_step, blocking=True):
        if self._moving_lock.acquire(False):
            self._stop_moving = False
            self._is_blocking = blocking
            def execute_cart_traj_cb(event):
                self._cur_result = self._execute_cart_traj(cart_arm, traj, time_step)
                self._timer = None
                if not self._is_blocking:
                    self._moving_lock.release()
            self._timer = rospy.Timer(rospy.Duration(0.00000001), execute_cart_traj_cb, oneshot=True)
            if self._is_blocking:
                self.wait_until_stopped()
                retval = self._cur_result
                self._moving_lock.release()
                return retval
            else:
                return True
        else:
            return False

    def _execute_cart_traj(self, cart_arm, traj, time_step):
        rate = rospy.Rate(1.0/time_step)
        for ep in traj:
            if rospy.is_shutdown() or self._stop_moving:
                return False
            cart_arm.set_ep(ep, time_step)
            rate.sleep()
        return True

class CartesianStepController(CartTrajController):
    def __init__(self):
        super(CartesianStepController, self).__init__()

        self.time_step = 1. / 20.
        self.arm = None
        self.cmd_lock = Lock()
        self.start_pub = rospy.Publisher("/start_pose", PoseStamped)
        self.end_pub = rospy.Publisher("/end_pose", PoseStamped)

    def set_arm(self, arm):
        self.arm = arm

    def _run_traj(self, traj, blocking=True):
        self.start_pub.publish(
                PoseConverter.to_pose_stamped_msg("/torso_lift_link", traj[0]))
        self.end_pub.publish(
                PoseConverter.to_pose_stamped_msg("/torso_lift_link", traj[-1]))
        # make sure traj beginning is close to current end effector position
        init_pos_tolerance = rospy.get_param("~init_pos_tolerance", 0.05)
        init_rot_tolerance = rospy.get_param("~init_rot_tolerance", np.pi/12)
        ee_pos, ee_rot = self.arm.get_end_effector_pose()
        _, rot_diff = PoseConverter.to_pos_euler((ee_pos, ee_rot * traj[0][1].T))
        pos_diff = np.linalg.norm(ee_pos - traj[0][0])
        if pos_diff > init_pos_tolerance:
            rospy.logwarn("[controller_base] End effector too far from current position. " + 
                          "Pos diff: %.3f, Tolerance: %.3f" % (pos_diff, init_pos_tolerance))
            return False
        if np.linalg.norm(rot_diff) > init_rot_tolerance:
            rospy.logwarn("[controller_base] End effector too far from current rotation. " + 
                          "Rot diff: %.3f, Tolerance: %.3f" % (np.linalg.norm(rot_diff), 
                                                               init_rot_tolerance))
            return False
        return self.execute_cart_traj(self.arm, traj, self.time_step, blocking=blocking)

    def execute_cart_move(self, change_ep, abs_sel, velocity=0.001,
                          num_samps=None, blocking=True):
        if not self.cmd_lock.acquire(False):
            return False
        cur_pos, cur_rot = self.arm.get_ep()
        change_pos_ep, change_rot_ep = change_ep
        abs_cart_ep_sel, is_abs_rot = abs_sel
        pos_f = np.where(abs_cart_ep_sel, change_pos_ep, 
                         np.array(cur_pos + cur_rot * np.mat(change_pos_ep).T).T[0])
        if is_abs_rot:
            rot_mat_f = change_rot_ep
        else:
            rpy = change_rot_ep
            _, cur_rot = self.arm.get_ep()
            rot_mat = np.mat(tf_trans.euler_matrix(*rpy))[:3,:3]
            rot_mat_f = cur_rot * rot_mat
        traj = self._create_cart_trajectory(pos_f, rot_mat_f, velocity, num_samps)
        retval = self._run_traj(traj, blocking=blocking)
        self.cmd_lock.release()
        return retval

    def _create_cart_trajectory(self, pos_f, rot_mat_f, velocity=0.001, num_samps=None):
        cur_pos, cur_rot = self.arm.get_ep()

        rpy = tf_trans.euler_from_matrix(cur_rot.T * rot_mat_f) # get roll, pitch, yaw of angle diff

        if num_samps is None:
            num_samps = np.max([2, int(np.linalg.norm(pos_f - cur_pos) / velocity), 
                                   int(np.linalg.norm(rpy) / velocity)])

        traj = self.arm.interpolate_ep([cur_pos, cur_rot], 
                                       [np.mat(pos_f).T, rot_mat_f], 
                                       min_jerk_traj(num_samps))
        return traj
