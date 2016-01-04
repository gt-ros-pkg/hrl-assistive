#!/usr/bin/env python
import roslib
import numpy as np
import time
import numpy as np
import math as m
import openravepy as op
import copy

# roslib.load_manifest('hrl_head_tracking')
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
import os.path



class PoseRecorder(object):
    def __init__(self, file_number, task, reference, model='chair'):
        self.model = model
        self.task = task
        self.reference = reference
        self.count = 0
        self.lock = RLock()
        self.head_pose = []
        self.depth_img = []
        self.camera_depth_info = None
        self.camera_rgb_info = None
        self.rgb_img = []
        self.file_number = file_number
        self.head_pose_sub = rospy.Subscriber('/haptic_mpc/head_pose', PoseStamped, self.head_pose_cb)
        self.listener = tf.TransformListener()
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_base_selection')
        self.setup_openrave()
        self.rviz_model_publisher()
        # rospy.sleep(3)
        print 'Ready to record Ground Truth head pose!'

    def head_pose_cb(self, msg):
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        ori = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        if self.head_pose == [pos, ori]:
            now = rospy.Time.now()
            self.listener.waitForTransform('/base_link', '/torso_lift_link', now,
                                           rospy.Duration(20))
            (trans, rot) = self.listener.lookupTransform('/base_link',
                                                             '/torso_lift_link', now)
            pos_out, ori_out = Bmat_to_pos_quat(createBMatrix(trans, rot)*createBMatrix(pos, ori))
            # pos_out, ori_out = pos, ori
            record_file_gt = open(''.join([self.pkg_path, '/data/goals/', self.task, '/', self.task, '_', self.reference,
                                           '_pose_', str(self.file_number), '.txt']), 'w')
            record_file_gt.write(''.join(['[%f, %f, %f, %f, %f, %f, %f]\n' % (pos_out[0], pos_out[1], pos_out[2],
                                                                              ori_out[0], ori_out[1], ori_out[2],
                                                                              ori_out[3])]))
            record_file_gt.close()
            print 'Just saved file # ', self.file_number, 'for task ', self.task, ' using reference ', self.reference
            self.file_number += 1
        self.head_pose = [pos, ori]


    def rviz_model_publisher(self):
        sub_pos, sub_ori = Bmat_to_pos_quat(self.originsubject_B_originworld)
        self.publish_sub_marker(sub_pos, sub_ori)
        if self.model == 'chair':
            if self.reference == 'head':
                ref_model = self.subject.GetLink('head_center')
        elif self.model == 'autobed':
            if self.reference == 'head':
                ref_model = self.autobed.GetLink('head_link')
            elif self.reference == 'left_arm':
                ref_model = self.autobed.GetLink('arm_left_link')
            elif self.reference == 'right_arm':
                ref_model = self.autobed.GetLink('arm_right_link')
            elif self.reference == 'left_thigh':
                ref_model = self.autobed.GetLink('quad_left_link')
            elif self.reference == 'right_thigh':
                ref_model = self.autobed.GetLink('quad_right_link')
            elif self.reference == 'left_knee':
                ref_model = self.autobed.GetLink('calf_left_link')
            elif self.reference == 'right_knee':
                ref_model = self.autobed.GetLink('calf_right_link')
            elif self.reference == 'left_forearm':
                ref_model = self.autobed.GetLink('forearm_left_link')
            elif self.reference == 'right_forearm':
                ref_model = self.autobed.GetLink('forearm_right_link')
            elif self.reference == 'chest':
                ref_model = self.autobed.GetLink('upper_body_link')
        origin_B_ref = np.matrix(ref_model.GetTransform())
        ref_pos, ref_ori = Bmat_to_pos_quat(origin_B_ref)
        # self.publish_reference_marker(head_pos, head_ori, 'head_center')
        self.publish_reference_pose(ref_pos, ref_ori, 'reference_pose')
        if os.path.isfile(''.join([self.pkg_path, '/data/goals/', self.task, '/', self.task, '_', self.reference,
                                              '_pose_reference', '.txt'])):
            print '\n\n:: WARNING ::\n:: WARNING :: There was a previously existing reference file in that location. I ' \
                  'will rename it with _previous and proceed. Stop now if you do not want to continue deleting ' \
                  'files!\n:: WARNING ::\n\n'
            os.rename(''.join([self.pkg_path, '/data/goals/', self.task, '/', self.task, '_', self.reference,
                                              '_pose_reference', '.txt']),
                      ''.join([self.pkg_path, '/data/goals/', self.task, '/', self.task, '_', self.reference,
                                              '_pose_reference_previous', '.txt']))
        record_file_reference = open(''.join([self.pkg_path, '/data/goals/', self.task, '/', self.task, '_', self.reference,
                                              '_pose_reference', '.txt']), 'w')
        record_file_reference.write(''.join(['[%f, %f, %f, %f, %f, %f, %f]\n' % (ref_pos[0], ref_pos[1], ref_pos[2],
                                                                                 ref_ori[0], ref_ori[1], ref_ori[2],
                                                                                 ref_ori[3])]))

    def publish_reference_pose(self, pos, ori, name):
        vis_pub = rospy.Publisher(''.join(['~', name]), PoseStamped, queue_size=1, latch=True)
        marker = PoseStamped()
        marker.header.frame_id = "/base_link"
        marker.header.stamp = rospy.Time()
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.x = ori[0]
        marker.pose.orientation.y = ori[1]
        marker.pose.orientation.z = ori[2]
        marker.pose.orientation.w = ori[3]
        vis_pub.publish(marker)
        print 'Published a goal marker to rviz'

    def publish_reference_marker(self, pos, ori, name):
        vis_pub = rospy.Publisher(''.join(['~', name]), Marker, queue_size=1, latch=True)
        marker = Marker()
        marker.header.frame_id = "/base_link"
        marker.header.stamp = rospy.Time()
        marker.ns = name
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.x = ori[0]
        marker.pose.orientation.y = ori[1]
        marker.pose.orientation.z = ori[2]
        marker.pose.orientation.w = ori[3]
        marker.scale.x = .2
        marker.scale.y = .2
        marker.scale.z = .2
        marker.color.a = 1.
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
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
            name = 'subject_model'
            marker.mesh_resource = "package://hrl_base_selection/models/wheelchair_and_body_assembly_rviz.STL"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        elif self.model == 'bed':
            name = 'subject_model'
            marker.mesh_resource = "package://hrl_base_selection/models/head_bed.dae"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        elif self.model == 'autobed':
            name = 'subject_model'
            marker.mesh_resource = "package://hrl_base_selection/models/bed_and_body_v3_rviz.dae"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        else:
            print 'I got a bad model. What is going on???'
            return None
        vis_pub = rospy.Publisher(''.join(['~',name]), Marker, queue_size=1, latch=True)
        marker.ns = ''.join(['base_service_',name])
        vis_pub.publish(marker)
        print 'Published a model of the subject to rviz'

    def setup_openrave(self):
        self.env = op.Environment()
        if self.model == 'chair':
            self.env.Load(''.join([self.pkg_path, '/collada/wheelchair_and_body_assembly.dae']))
            self.wheelchair = self.env.GetRobots()[0]
            headmodel = self.wheelchair.GetLink('head_center')
            head_T = np.matrix(headmodel.GetTransform())
            self.originsubject_B_headfloor = np.matrix([[1., 0.,  0., head_T[0, 3]],  # .442603 #.45 #.438
                                                        [0., 1.,  0., head_T[1, 3]],  # 0.34 #.42
                                                        [0., 0.,  1.,           0.],
                                                        [0., 0.,  0.,           1.]])
            self.originsubject_B_originworld = copy.copy(self.originsubject_B_headfloor)
        elif self.model == 'autobed':
            self.env.Load(''.join([self.pkg_path, '/collada/bed_and_body_v3_rounded.dae']))
            self.autobed = self.env.GetRobots()[0]
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
            # headmodel = self.autobed.GetLink('head_link')
            # head_T = np.matrix(headmodel.GetTransform())

            # self.originsubject_B_headfloor = np.matrix([[1.,  0., 0.,  head_T[0, 3]],  #.45 #.438
            #                                             [0.,  1., 0.,  head_T[1, 3]],  # 0.34 #.42
            #                                             [0.,  0., 1.,           0.],
            #                                             [0.,  0., 0.,           1.]])
            self.originsubject_B_originworld = np.matrix(np.eye(4))

        else:
            print 'I got a bad model. What is going on???'
            return None
        self.subject = self.env.GetRobots()[0]
        self.subject.SetTransform(np.array(self.originsubject_B_originworld))
        print 'OpenRave has succesfully been initialized. \n'

if __name__ == "__main__":
    rospy.init_node('pose_recorder')
    model = 'autobed'
    file_number = 0
    task = 'scratching_knee_right'  # options are: bathing, brushing, feeding, shaving, scratching_upper_arm/forearm/thigh/chest/knee_left/right
    reference = 'right_knee'  # options are: head, left/right_arm, left/right_thigh, left/right_forearm, chest
    recorder = PoseRecorder(file_number, task, reference, model=model)
    rospy.spin()






















