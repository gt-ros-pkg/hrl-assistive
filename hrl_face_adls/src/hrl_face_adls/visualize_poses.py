#!/usr/bin/python

import sys
import yaml
import numpy as np

import roslib
roslib.load_manifest("hrl_ellipsoidal_control")
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3, PoseStamped
import rosbag

from hrl_ellipsoidal_control.ellipsoid_space import EllipsoidSpace
from hrl_geom.pose_converter import PoseConv
from std_msgs.msg import ColorRGBA

def create_arrow_marker(pose, m_id, color=ColorRGBA(1., 0., 0., 1.)):
    m = Marker()
    m.header.frame_id = "/base_link"
    m.header.stamp = rospy.Time.now()
    m.ns = "ell_pose_viz"
    m.id = m_id
    m.type = Marker.ARROW
    m.action = Marker.ADD
    m.scale = Vector3(0.19, 0.09, 0.02)
    m.color = color
    m.pose = PoseConv.to_pose_msg(pose)
    return m

def main():
    rospy.init_node("visualize_poses")
    pose_file = file(sys.argv[1], 'r')
    params = yaml.load(pose_file)
    ell_reg_bag = rosbag.Bag(sys.argv[2], 'r')
    for topic, ell_reg, ts in ell_reg_bag.read_messages():
        pass
    ell_space = EllipsoidSpace(E=ell_reg.E)

    pub_head_pose = rospy.Publisher("/head_center_test", PoseStamped)
    pub_arrows = rospy.Publisher("visualization_markers_array", MarkerArray)
    def create_tool_arrow():
        arrows = MarkerArray()
        color = ColorRGBA(0., 0., 1., 1.)
        for i, param in enumerate(params):
            ell_pos, ell_rot = params[param]
            _, ell_rot_mat = PoseConv.to_pos_rot([0]*3, ell_rot)
            cart_pose = PoseConv.to_homo_mat(ell_space.ellipsoidal_to_pose(*ell_pos))
            cart_pose[:3,:3] = cart_pose[:3,:3] * ell_rot_mat
            arrow = create_arrow_marker(cart_pose, i, color)
            arrow.header.stamp = rospy.Time.now()
            arrows.markers.append(arrow)
        return arrows
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        pub_head_pose.publish(PoseConv.to_pose_stamped_msg("/base_link", [0]*3, [0]*3))
        arrows = create_tool_arrow()
        pub_arrows.publish(arrows)
        r.sleep()

if __name__ == "__main__":
    main()
