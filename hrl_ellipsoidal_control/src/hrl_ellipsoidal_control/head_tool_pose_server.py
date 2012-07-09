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
#from hrl_rfh_fall_2011.srv import GetHeadPose 

head_poses = {
    #             lat   lon    height    roll   pitch   yaw
    "near_ear" : [(4 * np.pi/8,   -3 * np.pi/8,     1),      (0,     0,      0)],
    "upper_cheek" : [(4 * np.pi/8,   -1.5 * np.pi/8,     1),      (0,     0,      0)],
    "middle_cheek" : [(4.5 * np.pi/8,   -2 * np.pi/8,     1),      (0,     0,      0)],
    "jaw_bone" : [(5.1 * np.pi/8,   -2 * np.pi/8,     1),      (0,     0,      0)],
    "back_neck" : [(5.1 * np.pi/8,   -3 * np.pi/8,     1),      (0,     0,      0)],
    "nose" : [(4 * np.pi/8,    0 * np.pi/8,     1),      (0,     0,      0)],
    "chin" : [(5.4 * np.pi/8,    0 * np.pi/8,     1),      (0 * np.pi / 2,    np.pi/8,      0)],
    "mouth_corner" : [(4.5 * np.pi/8,   -0.9 * np.pi/8,     1),      (0,     0,      0)]
}

def create_arrow_marker(pose, m_id, color=ColorRGBA(1., 0., 0., 1.)):
    m = Marker()
    m.header.frame_id = "/ellipse_frame"
    m.header.stamp = rospy.Time.now()
    m.ns = "ell_pose"
    m.id = m_id
    m.type = Marker.ARROW
    m.action = Marker.ADD
    m.scale = Vector3(0.19, 0.09, 0.02)
    m.color = color
    m.pose = PoseConverter.to_pose_msg(pose)
    return m

class HeadToolPoseServer(object):
    def __init__(self):
        self.ell_space = EllipsoidSpace()
        self.ell_sub = rospy.Subscriber("/ellipsoid_params", EllipsoidParams, self.read_params)
        #self.head_pose_srv = rospy.Service("/get_head_pose", GetHeadPose, self.get_head_pose_srv)
        self.lock_ell = False
        self.found_params = False
#self.tmp_pub = rospy.Publisher("/toolpose", PoseStamped)

    def lock_ell_model(self, lock_model):
        self.lock_ell = lock_model

    def read_params(self, e_params):
        if not self.lock_ell:
            self.ell_space.load_ell_params(e_params.E, e_params.is_oblate, e_params.height)
            self.found_params = True

    def get_many_vectors(self):
        arrows = MarkerArray()
        coords = []
        i = 0
        color = ColorRGBA(0., 1., 0., 1.)
        for lat in np.linspace(0, np.pi, 10):
            color.g += 0.1
            color.b = 0
            for lon in np.linspace(0, 2 * np.pi , 10):
                color.b += 0.1
                coords.append((lat, lon, 1, i, copy.copy(color)))
                i += 1
        arrows.markers = [create_arrow_marker(self.ell_space.ellipsoidal_to_pose(lat, lon, height), i, clr)
                          for lat, lon, height, i, clr in coords] 
        return arrows

    def get_pose_markers(self):
        arrows = MarkerArray()
        coords = []
        i = 0
        color = ColorRGBA(0., 1., 0., 1.)
        for name in head_poses:
            arrows.markers.append(create_arrow_marker(self.get_head_pose(name), i, color))
            i += 1
        return arrows

    def get_head_pose(self, name, gripper_rot=0.):
        lat, lon, height = head_poses[name][0]
        roll, pitch, yaw = head_poses[name][1]
        pos, rot = PoseConverter.to_pos_rot(self.ell_space.ellipsoidal_to_pose(lat, lon, height))
        rot = rot * tf_trans.euler_matrix(yaw, pitch, roll + gripper_rot, 'rzyx')[:3, :3] 
        return pos, rot

    def get_head_pose_srv(self, req):
        if req.name not in head_poses:
            pose = (np.mat([-9999, -9999, -9999]).T, np.mat(np.zeros((3, 3))))
        else:
            pose = self.get_head_pose(req.name, req.gripper_rot)
        frame = "/ellipse_frame"
        pose_stamped = PoseConverter.to_pose_stamped_msg(frame, pose)
#self.tmp_pub.publish(pose_stamped)
        return pose_stamped

def main():
    rospy.init_node("head_tool_pose_server")
    htps = HeadToolPoseServer()
    pub_arrows = rospy.Publisher("visualization_markers_array", MarkerArray)
    if False:
        while not rospy.is_shutdown():
            arrows = htps.get_pose_markers()
            for arrow in arrows.markers:
                arrow.header.stamp = rospy.Time.now()
            pub_arrows.publish(arrows)
            rospy.sleep(0.1)
    else:
        while not rospy.is_shutdown():
            arrows = htps.get_many_vectors()
            for arrow in arrows.markers:
                arrow.header.stamp = rospy.Time.now()
            pub_arrows.publish(arrows)
            rospy.sleep(1)

if __name__ == "__main__":
    main()
