#!/usr/bin/env python
import roslib
import numpy as np
import time
import numpy as np
import math as m
import openravepy as op
import copy

roslib.load_manifest('hrl_head_tracking')
import rospy
import rospkg
from threading import RLock
from sensor_msgs.msg import PointCloud2, CompressedImage, Image, CameraInfo

from geometry_msgs.msg import PoseStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
import tf

roslib.load_manifest('hrl_base_selection')
from helper_functions import createBMatrix, Bmat_to_pos_quat

roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle
from joblib import Parallel, delayed


class PoseRecorder(object):
    def __init__(self, file_number, subject_number, video=False, model='autobed'):

        self.model = model

        self.count = 0
        self.lock = RLock()
        self.head_pose = []
        self.depth_img = []
        self.camera_depth_info = None
        self.camera_rgb_info = None
        self.rgb_img = []
        self.file_number = file_number
        self.subject_number = subject_number
        self.head_pose_sub = rospy.Subscriber('/haptic_mpc/head_pose', PoseStamped, self.head_pose_cb)
        self.listener = tf.TransformListener()
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_head_tracking')
        # self.pkg_path = '/home/ari/git/gt-ros-pkg.hrl_assistive/hrl_head_tracking/'
        self.setup_openrave()
        self.rviz_model_publisher()
        # rospy.sleep(3)
        print 'Ready to record Ground Truth head pose!'

    def rviz_model_publisher(self):
        sub_pos, sub_ori = Bmat_to_pos_quat(self.originsubject_B_originworld)
        self.publish_sub_marker(sub_pos, sub_ori)
        if self.model == 'chair':
            headmodel = self.subject.GetLink('head_center')
            origin_B_head = np.matrix(headmodel.GetTransform())
            head_pos, head_ori = Bmat_to_pos_quat(origin_B_head)
            self.publish_reference_marker(head_pos, head_ori, 'head_center')
        elif self.model == 'autobed':
            headmodel = self.autobed.GetLink('head_link')
            origin_B_head = np.matrix(headmodel.GetTransform())
            head_pos, head_ori = Bmat_to_pos_quat(origin_B_head)
            self.publish_reference_marker(head_pos, head_ori, 'head_center')


    def publish_reference_marker(self, pos, ori, name):
        # vis_pub = rospy.Publisher(''.join(['~', name]), Marker, latch=True)
        # marker = Marker()
        # #marker.header.frame_id = "/base_footprint"
        # marker.header.frame_id = "/base_link"
        # marker.header.stamp = rospy.Time()
        # marker.ns = name
        # marker.id = 0
        # marker.type = Marker.ARROW
        # marker.action = Marker.ADD
        # marker.pose.position.x = pos[0]
        # marker.pose.position.y = pos[1]
        # marker.pose.position.z = pos[2]
        # marker.pose.orientation.x = ori[0]
        # marker.pose.orientation.y = ori[1]
        # marker.pose.orientation.z = ori[2]
        # marker.pose.orientation.w = ori[3]
        # marker.scale.x = .2
        # marker.scale.y = .2
        # marker.scale.z = .2
        # marker.color.a = 1.
        # marker.color.r = 1.0
        # marker.color.g = 0.0
        # marker.color.b = 0.0
        # vis_pub.publish(marker)
        vis_pub = rospy.Publisher(''.join(['~', name]), PoseStamped, latch=True)
        marker = PoseStamped()
        #marker.header.frame_id = "/base_footprint"
        marker.header.frame_id = "/base_link"
        marker.header.stamp = rospy.Time()
        # marker.ns = name
        # marker.id = 0
        # marker.type = Marker.ARROW
        # marker.action = Marker.ADD
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.x = ori[0]
        marker.pose.orientation.y = ori[1]
        marker.pose.orientation.z = ori[2]
        marker.pose.orientation.w = ori[3]
        # marker.scale.x = .2
        # marker.scale.y = .2
        # marker.scale.z = .2
        # marker.color.a = 1.
        # marker.color.r = 1.0
        # marker.color.g = 0.0
        # marker.color.b = 0.0
        vis_pub.publish(marker)
        print 'Published a goal marker to rviz'

    def publish_sub_marker(self, pos, ori):
        marker = Marker()
        #marker.header.frame_id = "/base_footprint"
        marker.header.frame_id = "/base_link"
        marker.header.stamp = rospy.Time()
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.x = ori[0]
        marker.pose.orientation.y = ori[1]
        marker.pose.orientation.z = ori[2]
        marker.pose.orientation.w = ori[3]
        marker.color.a = 1.
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        if self.model == 'chair':
            name = 'wc_model'
            marker.mesh_resource = "package://hrl_base_selection/models/wheelchair_and_body_assembly_rviz.dae"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        elif self.model == 'bed':
            name = 'bed_model'
            marker.mesh_resource = "package://hrl_base_selection/models/head_bed.dae"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        elif self.model == 'autobed':
            name = 'autobed_model'
            marker.mesh_resource = "package://hrl_base_selection/models/bed_and_body_v3_rviz.dae"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        else:
            print 'I got a bad model. What is going on???'
            return None
        vis_pub = rospy.Publisher(''.join(['~',name]), Marker, latch=True)
        marker.ns = ''.join(['base_service_',name])
        vis_pub.publish(marker)
        print 'Published a model of the subject to rviz'

    def head_pose_cb(self, msg):
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        ori = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        if self.head_pose == [pos, ori]:
            now = rospy.Time.now()
            self.listener.waitForTransform('/torso_lift_link', '/head_mount_kinect_depth_optical_frame', now,
                                           rospy.Duration(20))
            (trans_d, rot_d) = self.listener.lookupTransform('/torso_lift_link',
                                                             '/head_mount_kinect_depth_optical_frame', now)
            # now = rospy.Time.now()
            # self.listener.waitForTransform('/torso_lift_link', '/head_mount_kinect_rgb_optical_frame', now,
            #                                rospy.Duration(20))
            # (trans_r, rot_r) = self.listener.lookupTransform('/torso_lift_link',
            #                                                  '/head_mount_kinect_rgb_optical_frame', now)

            # pos_out_d, ori_out_d = Bmat_to_pos_quat(createBMatrix(trans_d, rot_d).I*createBMatrix(pos, ori))
            # pos_out_d, ori_out_d = Bmat_to_pos_quat(createBMatrix(pos, ori))
            pos_out_d, ori_out_d = pos, ori

            record_file_gt_d = open(''.join([self.pkg_path, '/data/', 'subj_', str(self.subject_number),
                                             '_img_', str(self.file_number), '_gt_depth', '.txt']), 'w')
            record_file_gt_d.write(''.join([' %f %f %f %f %f %f %f \n' % (pos_out_d[0], pos_out_d[1], pos_out_d[2],
                                                                          ori_out_d[0], ori_out_d[1], ori_out_d[2],
                                                                          ori_out_d[3])]))
            record_file_gt_d.close()
            # pos_out_r, ori_out_r = Bmat_to_pos_quat(createBMatrix(trans_r, rot_r).I*createBMatrix(pos, ori))
            # record_file_gt_r = open(''.join([self.pkg_path, '/data/', 'subj_', str(self.subject_number),
            #                                  '_img_', str(self.file_number), '_gt_rgb', '.txt']), 'w')
            # record_file_gt_r.write(''.join([' %f %f %f %f %f %f %f \n' % (pos_out_r[0], pos_out_r[1], pos_out_r[2],
            #                                                               ori_out_r[0], ori_out_r[1], ori_out_r[2],
            #                                                               ori_out_r[3])]))
            # record_file_gt_r.close()
            print 'Just saved file # ', self.file_number, 'for subject ', self.subject_number

            self.file_number += 1
        self.head_pose = [pos, ori]

    def setup_openrave(self):
        self.env = op.Environment()
        if self.model == 'chair':
            self.env.Load(''.join([pkg_path, '/collada/wheelchair_and_body_assembly.dae']))
            self.wheelchair = self.env.GetRobots()[0]
            headmodel = self.wheelchair.GetLink('head_center')
            head_T = np.matrix(headmodel.GetTransform())
            self.originsubject_B_headfloor = np.matrix([[1., 0.,  0., head_T[0, 3]],  # .442603 #.45 #.438
                                                        [0., 1.,  0., head_T[1, 3]],  # 0.34 #.42
                                                        [0., 0.,  1.,           0.],
                                                        [0., 0.,  0.,           1.]])
            self.originsubject_B_originworld = copy.copy(self.originsubject_B_headfloor)
        elif self.model == 'autobed':
            self.env.Load(''.join([pkg_path, '/collada/bed_and_body_v3_rounded.dae']))
            self.autobed = self.env.GetRobots()[1]
            v = self.autobed.GetActiveDOFValues()

            #0 degrees, 0 height
            v[self.autobed.GetJoint('head_rest_hinge').GetDOFIndex()] = 0.0
            v[self.autobed.GetJoint('tele_legs_joint').GetDOFIndex()] = -0.
            v[self.autobed.GetJoint('neck_body_joint').GetDOFIndex()] = -.1
            v[self.autobed.GetJoint('upper_mid_body_joint').GetDOFIndex()] = .4
            v[self.autobed.GetJoint('mid_lower_body_joint').GetDOFIndex()] = -.72
            v[self.autobed.GetJoint('body_quad_left_joint').GetDOFIndex()] = -0.4
            v[self.autobed.GetJoint('body_quad_right_joint').GetDOFIndex()] = -0.4
            v[self.autobed.GetJoint('quad_calf_left_joint').GetDOFIndex()] = 0.1
            v[self.autobed.GetJoint('quad_calf_right_joint').GetDOFIndex()] = 0.1
            v[self.autobed.GetJoint('calf_foot_left_joint').GetDOFIndex()] = .02
            v[self.autobed.GetJoint('calf_foot_right_joint').GetDOFIndex()] = .02
            v[self.autobed.GetJoint('body_arm_left_joint').GetDOFIndex()] = -.12
            v[self.autobed.GetJoint('body_arm_right_joint').GetDOFIndex()] = -.12
            v[self.autobed.GetJoint('arm_forearm_left_joint').GetDOFIndex()] = 0.05
            v[self.autobed.GetJoint('arm_forearm_right_joint').GetDOFIndex()] = 0.05
            v[self.autobed.GetJoint('forearm_hand_left_joint').GetDOFIndex()] = -0.1
            v[self.autobed.GetJoint('forearm_hand_right_joint').GetDOFIndex()] = -0.1
            #v[self.autobed.GetJoint('leg_rest_upper_joint').GetDOFIndex()]= -0.1
            self.autobed.SetActiveDOFValues(v)
            self.env.UpdatePublishedBodies()
            headmodel = self.autobed.GetLink('head_link')
            head_T = np.matrix(headmodel.GetTransform())

            self.originsubject_B_headfloor = np.matrix([[1.,  0., 0.,  head_T[0, 3]],  #.45 #.438
                                                        [0.,  1., 0.,  head_T[1, 3]],  # 0.34 #.42
                                                        [0.,  0., 1.,           0.],
                                                        [0.,  0., 0.,           1.]])
            self.originsubject_B_originworld = np.matrix(np.eye(4))

        else:
            print 'I got a bad model. What is going on???'
            return None
        self.subject = self.env.GetBodies()[0]
        self.subject.SetTransform(np.array(self.originsubject_B_originworld))
        print 'OpenRave has succesfully been initialized. \n'

if __name__ == "__main__":
    rospy.init_node('pose_recorder')

    file_number = 0
    subject_number = 0
    recorder = PoseRecorder(file_number, subject_number)
    rospy.spin()






















