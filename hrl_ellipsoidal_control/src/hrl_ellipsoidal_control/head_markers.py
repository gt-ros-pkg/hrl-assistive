#! /usr/bin/python

import numpy as np
import copy

import roslib
roslib.load_manifest('hrl_ellipsoidal_control')
import rospy
import tf.transformations as tf_trans

from hrl_ellipsoidal_control.msg import EllipsoidParams
from geometry_msgs.msg import PoseStamped, PoseArray, Vector3
from hrl_generic_arms.pose_converter import PoseConverter
from hrl_ellipsoidal_control.ellipsoid_space import EllipsoidSpace
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

eye_scale = Vector3(0.02, 0.01, 0.010)
l_eye_loc = [(3.5 * np.pi/8, -0.9 * np.pi/8,     1.20), (    np.pi/2,    np.pi/2,     0)]
r_eye_loc = [(3.5 * np.pi/8,  0.9 * np.pi/8,     1.20), (    np.pi/2,    np.pi/2,     0)]

mouth_scale = Vector3(0.05, 0.01, 0.010)
mouth_loc = [(4.7 * np.pi/8,    0 * np.pi/8,     1.25), (    np.pi/2,    np.pi/2,     0)]

ear_scale = Vector3(0.06, 0.03, 0.030)
l_ear_loc = [(  4 * np.pi/8,   -3.9 * np.pi/8,     1.10), (  0,    np.pi/2,   0)]
r_ear_loc = [(  4 * np.pi/8,    3.9 * np.pi/8,     1.10), (  0,    np.pi/2,   0)]

class HeadMarkers(object):
    def __init__(self):
        self.ell_space = EllipsoidSpace(1)
        self.ell_sub = rospy.Subscriber("/ellipsoid_params", EllipsoidParams, self.read_params)
        self.found_params = False

    def read_params(self, e_params):
        self.ell_space.load_ell_params(e_params.E, e_params.is_oblate, e_params.height)
        if not self.found_params:
            rospy.loginfo("[head_markers] Found params from /ellipsoid_params")
        self.found_params = True

    def create_eye_marker(self, pose, m_id, color=ColorRGBA(1., 1., 1., 1.)):
        m = Marker()
#m.header.frame_id = "/base_link"
        m.header.frame_id = "/ellipse_frame"
        m.header.stamp = rospy.Time.now()
        m.ns = "head_markers"
        m.id = m_id
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.scale = eye_scale
        m.color = color
        m.pose = PoseConverter.to_pose_msg(pose)
        return m

    def create_mouth_marker(self, pose, m_id, color=ColorRGBA(1., 0., 0., 1.)):
        m = Marker()
#m.header.frame_id = "/base_link"
        m.header.frame_id = "/ellipse_frame"
        m.header.stamp = rospy.Time.now()
        m.ns = "head_markers"
        m.id = m_id
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.scale = mouth_scale
        m.color = color
        m.pose = PoseConverter.to_pose_msg(pose)
        return m

    def create_ear_marker(self, pose, m_id, color=ColorRGBA(0., 1., 1., 1.)):
        m = Marker()
#m.header.frame_id = "/base_link"
        m.header.frame_id = "/ellipse_frame"
        m.header.stamp = rospy.Time.now()
        m.ns = "head_markers"
        m.id = m_id
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.scale = ear_scale
        m.color = color
        m.pose = PoseConverter.to_pose_msg(pose)
        return m

    def get_head(self):
        if not self.found_params:
            return
        head_array = MarkerArray()
        head_array.markers.append(
                self.create_eye_marker(self.get_head_pose(l_eye_loc), 0))
        head_array.markers.append(
                self.create_eye_marker(self.get_head_pose(r_eye_loc), 1))
        head_array.markers.append(
                self.create_mouth_marker(self.get_head_pose(mouth_loc), 2))
        head_array.markers.append(
                self.create_ear_marker(self.get_head_pose(l_ear_loc), 3))
        head_array.markers.append(
                self.create_ear_marker(self.get_head_pose(r_ear_loc), 4))
        return head_array

    def get_head_pose(self, ell_coords_rot, gripper_rot=0.):
        lat, lon, height = ell_coords_rot[0]
        roll, pitch, yaw = ell_coords_rot[1]
        pos, rot = PoseConverter.to_pos_rot(self.ell_space.ellipsoidal_to_pose(lat, lon, height))
        rot = rot * tf_trans.euler_matrix(yaw, pitch, roll + gripper_rot, 'szyx')[:3, :3] 
        return pos, rot

def main():
    rospy.init_node("head_markers")
    hm = HeadMarkers()
    pub_head = rospy.Publisher("visualization_markers_array", MarkerArray)
    while not rospy.is_shutdown():
        head = hm.get_head()
        pub_head.publish(head)
        rospy.sleep(0.1)
    

if __name__ == "__main__":
    main()
