#!/usr/bin/env python
import roslib
import numpy as np
import math as m
roslib.load_manifest('hrl_head_tracking')
import rospy
import rospkg
from threading import RLock
from sensor_msgs.msg import PointCloud2, CompressedImage, Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import PoseStamped, Point, Quaternion
import tf

roslib.load_manifest('hrl_base_selection')
from helper_functions import createBMatrix, Bmat_to_pos_quat


roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle


class HeadPoseRecorder(object):
    def __init__(self, file_number, subject_number):
        self.lock = RLock()
        self.head_pose = []
        self.depth_img = []
        self.rgb_img = []
        self.file_number = file_number
        self.subject_number = subject_number
        self.head_pose_sub = rospy.Subscriber('/haptic_mpc/head_pose', PoseStamped, self.head_pose_cb)
        self.listener = tf.TransformListener()
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_head_tracking')
        # self.pkg_path = '/home/ari/git/gt-ros-pkg.hrl_assistive/hrl_head_tracking/'
        print 'Ready to record Ground Truth head pose!'
        # depth_img_path = '/head_mount_kinect/depth/points'
        # self.depth_img_sub = rospy.Subscriber(depth_img_path, PointCloud2, self.depth_img_cb)
        depth_img_path = '/head_mount_kinect/depth/image'
        self.depth_img_sub = rospy.Subscriber(depth_img_path, Image, self.depth_img_cb)
        # depth_img_path = '/head_mount_kinect/depth_registered/image_raw'
        # self.depth_img_sub = rospy.Subscriber(depth_img_path, Image, self.depth_img_cb)
        # depth_img_path = '/head_mount_kinect/depth/image/compressed'
        # self.depth_img_sub = rospy.Subscriber(depth_img_path, CompressedImage, self.depth_img_cb)
        # rgb_img_path = '/head_mount_kinect/rgb/image_color/compressed'
        # self.rgb_img_sub = rospy.Subscriber(rgb_img_path, CompressedImage, self.rgb_img_cb)
        rgb_imgpath = '/head_mount_kinect/rgb/image_color'
        self.rgb_img_sub = rospy.Subscriber(rgb_imgpath, Image, self.rgb_img_cb)
        self.bridge = CvBridge()

    def head_pose_cb(self, msg):
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        ori = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        if self.head_pose == [pos, ori]:
            now = rospy.Time.now()
            self.listener.waitForTransform('/torso_lift_link', '/head_mount_kinect_depth_optical_frame', now,
                                           rospy.Duration(20))
            (trans_d, rot_d) = self.listener.lookupTransform('/torso_lift_link',
                                                             '/head_mount_kinect_depth_optical_frame', now)
            now = rospy.Time.now()
            self.listener.waitForTransform('/torso_lift_link', '/head_mount_kinect_rgb_optical_frame', now,
                                           rospy.Duration(20))
            (trans_r, rot_r) = self.listener.lookupTransform('/torso_lift_link',
                                                             '/head_mount_kinect_rgb_optical_frame', now)

            pos_out_d, ori_out_d = Bmat_to_pos_quat(createBMatrix(trans_d, rot_d).I*createBMatrix(pos, ori))

            record_file_gt_d = open(''.join([self.pkg_path, '/data/', 'subj_', str(self.subject_number),
                                             '_img_', str(self.file_number), '_gt_depth', '.txt']), 'w')
            record_file_gt_d.write(''.join([' %f %f %f %f %f %f %f \n' % (pos_out_d[0], pos_out_d[1], pos_out_d[2],
                                                                          ori_out_d[0], ori_out_d[1], ori_out_d[2],
                                                                          ori_out_d[3])]))
            record_file_gt_d.close()
            pos_out_r, ori_out_r = Bmat_to_pos_quat(createBMatrix(trans_r, rot_r).I*createBMatrix(pos, ori))
            record_file_gt_r = open(''.join([self.pkg_path, '/data/', 'subj_', str(self.subject_number),
                                             '_img_', str(self.file_number), '_gt_rgb', '.txt']), 'w')
            record_file_gt_r.write(''.join([' %f %f %f %f %f %f %f \n' % (pos_out_r[0], pos_out_r[1], pos_out_r[2],
                                                                          ori_out_r[0], ori_out_r[1], ori_out_r[2],
                                                                          ori_out_r[3])]))
            record_file_gt_r.close()
            print 'Did all the tf things successfully'
            self.img_save()
            print 'Just saved file # ', self.file_number, 'for subject ', self.subject_number
            self.file_number += 1
        self.head_pose = [pos, ori]

    def img_save(self):
        # depth_img_file = open(''.join([self.pkg_path, '/data/', 'depth', '_subj_', str(self.subject_number), '_img_',
        #                                str(self.file_number), '.jpeg']), 'w')
        # print self.depth_img
        with self.lock:
            save_pickle(self.depth_img, ''.join([self.pkg_path, '/data/', 'subj_', str(self.subject_number),
                                                 '_img_', str(self.file_number), '_depth', '.pkl']))
            try:
                path = ''.join([self.pkg_path, '/data/', 'subj_', str(self.subject_number), '_img_',
                                str(self.file_number), '_depth', '.png'])
                # The depth image is a single-channel float32 image
                # the values is the distance in mm in z axis
                depth_image = self.bridge.imgmsg_to_cv(self.depth_img, '32FC1')
                # Convert the depth image to a Numpy array since most cv2 functions
                # require Numpy arrays.
                depth_array = np.array(depth_image, dtype=np.float32)
                # Normalize the depth image to fall between 0 (black) and 1 (white)
                cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
                # At this point you can display the result properly:
                # cv2.imshow('Depth Image', depth_display_image)
                # If you write it as it si, the result will be a image with only 0 to 1 values.
                # To actually store in a this a image like the one we are showing its needed
                # to reescale the otuput to 255 gray scale.
                cv2.imwrite(path, depth_array*255)

                # cv2.imwrite(path, frame)
            except CvBridgeError, e:
                print e

            try:
                path = ''.join([self.pkg_path, '/data/', 'subj_', str(self.subject_number), '_img_',
                                str(self.file_number), '_rgb', '.png'])
                # The depth image is a single-channel float32 image
                # the values is the distance in mm in z axis
                rgb_image = self.bridge.imgmsg_to_cv(self.rgb_img, 'bgr8')
                # Convert the depth image to a Numpy array since most cv2 functions
                # require Numpy arrays.
                rgb_array = np.array(rgb_image, dtype=np.uint8)
                # Normalize the depth image to fall between 0 (black) and 1 (white)
                # cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
                # At this point you can display the result properly:
                # cv2.imshow('Depth Image', depth_display_image)
                # If you write it as it si, the result will be a image with only 0 to 1 values.
                # To actually store in a this a image like the one we are showing its needed
                # to reescale the otuput to 255 gray scale.
                cv2.imwrite(path, rgb_array)

                # cv2.imwrite(path, frame)
            except CvBridgeError, e:
                print e


            # record_depth = open(''.join([self.pkg_path, '/data/', 'depth', '_subj_', str(self.subject_number), '_img_',
            #                            str(self.file_number), '.png']), 'w')
            # record_depth.write(depth_img)
            # record_depth.close()
            # record_depth = open(''.join([self.pkg_path, '/data/', 'depth', '_subj_', str(self.subject_number), '_img_',
            #                              str(self.file_number), '.jpeg']), 'w')
            # print self.depth_img
            # record_depth.write(self.depth_img.data)
            # record_depth.close()

            # record_rgb = open(''.join([self.pkg_path, '/data/', 'rgb', '_subj_', str(self.subject_number), '_img_',
            #                            str(self.file_number), '.jpeg']), 'w')
            # record_rgb.write(self.rgb_img.data)
            # record_rgb.close()
            # depth_img_file.write(self.depth_img.data)
            # depth_img_file.close()

    def rgb_img_cb(self, msg):
        self.rgb_img = msg
        # print 'got a new rgb image'

    def depth_img_cb(self, msg):
        self.depth_img = msg

    # def rgb_img_save(self):
    #     record_rgb = open(''.join([self.pkg_path, '/data/', 'gt', '_subj_', str(self.subject_number), '_img_',
    #                                     str(self.file_number), '.jpeg']), 'w')
    #     record_rgb.write(self.rgb_img)
    #     record_rgb.close()

    # def publish_head_marker(self, pos, ori, name):
    #     vis_pub = rospy.Publisher(''.join(['~', name]), Marker, latch=True)
    #     marker = Marker()
    #     #marker.header.frame_id = "/base_footprint"
    #     marker.header.frame_id = "/base_link"
    #     marker.header.stamp = rospy.Time()
    #     marker.ns = name
    #     marker.id = 0
    #     marker.type = Marker.ARROW
    #     marker.action = Marker.ADD
    #     marker.pose.position.x = pos[0]
    #     marker.pose.position.y = pos[1]
    #     marker.pose.position.z = pos[2]
    #     marker.pose.orientation.x = ori[0]
    #     marker.pose.orientation.y = ori[1]
    #     marker.pose.orientation.z = ori[2]
    #     marker.pose.orientation.w = ori[3]
    #     marker.scale.x = .2
    #     marker.scale.y = .2
    #     marker.scale.z = .2
    #     marker.color.a = 1.
    #     marker.color.r = 1.0
    #     marker.color.g = 0.0
    #     marker.color.b = 0.0
    #     vis_pub.publish(marker)
    #     print 'Published a goal marker to rviz'

if __name__ == "__main__":
    rospy.init_node('head_pose_recorder')

    file_number = 0
    subject_number = 1

    recorder = HeadPoseRecorder(file_number, subject_number)
    rospy.spin()






















