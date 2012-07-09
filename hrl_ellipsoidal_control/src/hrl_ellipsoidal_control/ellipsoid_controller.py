#! /usr/bin/python
        
import sys
import numpy as np

import roslib
roslib.load_manifest('hrl_ellipsoidal_control')

import rospy
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseStamped

from ellipsoid_space import EllipsoidSpace
from msg import EllipsoidMoveAction, EllipsoidMoveResult
from msg import EllipsoidParams
from hrl_ellipsoidal_control.controller_base import CartesianStepController, min_jerk_traj
from pykdl_utils.pr2_kin import kin_from_param
from hrl_generic_arms.pose_converter import PoseConverter

class EllipsoidParamServer(object):

    def __init__(self):
        self.ell_param_sub = rospy.Subscriber("/ellipsoid_params", EllipsoidParams, 
                                              self.load_params)
        self.kin_head = kin_from_param("base_link", "openni_rgb_optical_frame")
        self.ell_space = None
        self.head_center = None
        self.head_center_pub = rospy.Publisher("/head_center", PoseStamped)
        def pub_head_center(te):
            if self.head_center is not None:
                self.head_center_pub.publish(self.head_center)
        rospy.Timer(rospy.Duration(0.2), pub_head_center)

    def load_params(self, params):
        kinect_B_head = PoseConverter.to_homo_mat(params.e_frame)
        base_B_kinect = self.kin_head.forward_filled(base_segment="base_link",
                                                     target_segment="openni_rgb_optical_frame")
        base_B_head = base_B_kinect * kinect_B_head
        self.head_center = PoseConverter.to_pose_stamped_msg("/base_link",base_B_head)
        self.ell_space = EllipsoidSpace()
        self.ell_space.load_ell_params(params.E, params.is_oblate, params.height)
        rospy.loginfo("Loaded ellispoidal parameters.")

    def params_loaded(self):
        return self.ell_space is not None
    
    def get_ell_pose(self, pose):
        torso_B_kinect = self.kin_head.forward_filled(base_segment="/torso_lift_link")
        torso_B_ee = PoseConverter.to_homo_mat(pose)
        kinect_B_ee = torso_B_kinect**-1 * torso_B_ee
        ell_B_pose = self.get_ell_frame("/openni_rgb_optical_frame")**-1 * kinect_B_ee
        return self.ell_space.pose_to_ellipsoidal(ell_B_pose)

    ##
    # Get pose in robot's frame of ellipsoidal coordinates
    def robot_ellipsoidal_pose(self, lat, lon, height, orient_quat, kinect_frame_mat=None):
        if kinect_frame_mat is None:
            kinect_frame_mat = self.get_ell_frame()
        pos, quat = self.ell_space.ellipsoidal_to_pose(lat, lon, height)
        quat_rotated = tf_trans.quaternion_multiply(quat, orient_quat)
        ell_pose_mat = PoseConverter.to_homo_mat(pos, quat_rotated)
        return PoseConverter.to_pos_rot(kinect_frame_mat * ell_pose_mat)

    def get_ell_frame(self, frame="/torso_lift_link"):
        # find the current ellipsoid frame location in this frame
        base_B_head = PoseConverter.to_homo_mat(self.head_center)
        target_B_base = self.kin_head.forward_filled(target_segment=frame)
        return target_B_base**-1 * base_B_head

class EllipsoidController(CartesianStepController):

    def __init__(self):
        super(EllipsoidController, self).__init__()
        self.ell_server = EllipsoidParamServer()
        self._lat_bounds = None
        self._lon_bounds = None
        self._height_bounds = None
        self._no_bounds = True

    def get_ell_ep(self):
        ell_ep, ell_rot = self.ell_server.get_ell_pose(self.arm.get_ep())
        return ell_ep

    def execute_ell_move(self, change_ep, abs_sel, orient_quat=[0., 0., 0., 1.], 
                         velocity=0.001, blocking=True):
        if not self.cmd_lock.acquire(False):
            return False
        ell_f, rot_mat_f = self._parse_ell_move(change_ep, abs_sel, orient_quat)
        traj = self._create_ell_trajectory(ell_f, rot_mat_f, orient_quat, velocity)
        if traj is None:
            rospy.logerr("[ellipsoid_controller] Bad trajectory.")
            self.cmd_lock.release()
            return False
        retval = self._run_traj(traj, blocking=blocking)
        self.cmd_lock.release()
        return retval

    def set_bounds(self, lat_bounds=None, lon_bounds=None, height_bounds=None):
        if lat_bounds is None and lon_bounds is None and height_bounds is None:
            self._no_bounds = True
        self._no_bounds = False
        assert lon_bounds[1] >= 0
        self._lat_bounds = lat_bounds
        self._lon_bounds = lon_bounds
        self._height_bounds = height_bounds

    def _clip_ell_ep(self, ell_ep):
        if self._no_bounds:
            return ell_ep
        lat = np.clip(ell_ep[0], self._lat_bounds[0], self._lat_bounds[1])
        if self._lon_bounds[0] >= 0:
            lon = np.clip(ell_ep[1], self._lon_bounds[0], self._lon_bounds[1])
        else:
            ell_ep_1 = np.mod(ell_ep[1], 2 * np.pi)
            min_lon = np.mod(self._lon_bounds[0], 2 * np.pi)
            if ell_ep_1 >= min_lon or ell_ep_1 <= self._lon_bounds[1]:
                lon = ell_ep[1]
            else:
                if min_lon - ell_ep_1 < ell_ep_1 - self._lon_bounds[1]:
                    lon = min_lon
                else:
                    lon = self._lon_bounds[1]
        height = np.clip(ell_ep[2], self._height_bounds[0], self._height_bounds[1])
        return np.array([lat, lon, height])

    def arm_in_bounds(self):
        if self._no_bounds:
            return True
        ell_ep = self.get_ell_ep()
        equals = ell_ep == self._clip_ell_ep(ell_ep)
        print ell_ep, equals
        if self._lon_bounds[0] >= 0 and ell_ep[1] >= 0:
            return np.all(equals)
        else:
            ell_ep_1 = np.mod(ell_ep[1], 2 * np.pi)
            min_lon = np.mod(self._lon_bounds[0], 2 * np.pi)
            return (equals[0] and equals[2] and 
                    (ell_ep_1 >= min_lon or ell_ep_1 <= self._lon_bounds[1]))

    def _parse_ell_move(self, change_ep, abs_sel, orient_quat):
        change_ell_ep, change_rot_ep = change_ep
        abs_ell_ep_sel, is_abs_rot = abs_sel
        ell_f = np.where(abs_ell_ep_sel, change_ell_ep, 
                                     np.array(self.get_ell_ep()) + np.array(change_ell_ep))
        print "old", ell_f
        if ell_f[0] > np.pi:
            ell_f[0] = 2 * np.pi - ell_f[0]
            ell_f[1] *= -1
        if ell_f[0] < 0.:
            ell_f[0] *= -1
            ell_f[1] *= -1
        ell_f[1] = np.mod(ell_f[1], 2 * np.pi)
        ell_f = self._clip_ell_ep(ell_f)
        print "new", ell_f
        if is_abs_rot:
            rot_change_mat = change_rot_ep
            _, ell_final_rot = self.ell_server.robot_ellipsoidal_pose(ell_f[0], ell_f[1], ell_f[2],
                                                                      orient_quat)
            rot_mat_f = ell_final_rot * rot_change_mat
        else:
            quat = change_rot_ep
            _, cur_rot = self.arm.get_ep()
            rot_mat = np.mat(tf_trans.quaternion_matrix(quat))[:3,:3]
            rot_mat_f = cur_rot * rot_mat
        return ell_f, rot_mat_f

    def _create_ell_trajectory(self, ell_f, rot_mat_f, orient_quat=[0., 0., 0., 1.], velocity=0.001):
        _, cur_rot = self.arm.get_ep()

        rpy = tf_trans.euler_from_matrix(cur_rot.T * rot_mat_f) # get roll, pitch, yaw of angle diff

        ell_f[1] = np.mod(ell_f[1], 2 * np.pi) # wrap longitude value

        # get the current ellipsoidal location of the end effector
        ell_init = np.mat(self.get_ell_ep()).T 
        ell_final = np.mat(ell_f).T

        # find the closest longitude angle to interpolate to
        if np.fabs(2 * np.pi + ell_final[1,0] - ell_init[1,0]) < np.pi:
            ell_final[1,0] += 2 * np.pi
        elif np.fabs(-2 * np.pi + ell_final[1,0] - ell_init[1,0]) < np.pi:
            ell_final[1,0] -= 2 * np.pi
        
        if np.any(np.isnan(ell_init)) or np.any(np.isnan(ell_final)):
            rospy.logerr("[ellipsoid_controller] Nan values in ellipsoid EPs. " +
                         "ell_init: %f, %f, %f; " % (ell_init[0,0], ell_init[1,0], ell_init[2,0]) +
                         "ell_final: %f, %f, %f; " % (ell_final[0,0], ell_final[1,0], ell_final[2,0]))
            return None
        
        num_samps = np.max([2, int(np.linalg.norm(ell_final - ell_init) / velocity), 
                               int(np.linalg.norm(rpy) / velocity)])
        t_vals = min_jerk_traj(num_samps) # makes movement smooth
            
        # smoothly interpolate from init to final
        ell_traj = np.array(ell_init) + np.array(np.tile(ell_final - ell_init, 
                                                         (1, num_samps))) * np.array(t_vals)

        ell_frame_mat = self.ell_server.get_ell_frame()

        ell_pose_traj = [self.ell_server.robot_ellipsoidal_pose(
                            ell_traj[0,i], ell_traj[1,i], ell_traj[2,i], orient_quat, ell_frame_mat) 
                         for i in range(ell_traj.shape[1])]

        # modify rotation of trajectory
        _, ell_init_rot = self.ell_server.robot_ellipsoidal_pose(
                ell_init[0,0], ell_init[1,0], ell_init[2,0], orient_quat, ell_frame_mat)
        rot_adjust_traj = self.arm.interpolate_ep([np.mat([0]*3).T, cur_rot], 
                                                  [np.mat([0]*3).T, rot_mat_f], 
                                                  min_jerk_traj(num_samps))
        ell_pose_traj = [(ell_pose_traj[i][0], 
                          ell_pose_traj[i][1] * ell_init_rot.T * rot_adjust_traj[i][1]) 
                         for i in range(num_samps)]

        return ell_pose_traj

def main():
    rospy.init_node("ellipsoid_controller", sys.argv)
    cart_arm = create_pr2_arm('l', PR2ArmJTransposeTask, 
                              controller_name='%s_cart_jt_task', 
                              end_link="%s_gripper_shaver45_frame", timeout=0)

    rospy.sleep(1)
    ell_controller = EllipsoidController()
    ell_controller.set_arm(cart_arm)
    rospy.spin()
    

if __name__ == "__main__":
    main()
