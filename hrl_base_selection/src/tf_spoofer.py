#!/usr/bin/env python

from threading import Lock
import copy
import numpy as np

import roslib
roslib.load_manifest('hrl_face_adls')
import rospy
from hrl_msgs.msg import FloatArrayBare
from std_msgs.msg import String, Int32, Int8, Bool
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped
import tf
from tf import TransformListener
from tf import transformations as tft
roslib.load_manifest('hrl_base_selection')
from helper_functions import createBMatrix, Bmat_to_pos_quat
roslib.load_manifest('hrl_pr2_ar_servo')
from ar_pose.msg import ARMarker


class TF_Spoofer(object):
    # Object for spoofing tf frames and ar tags to allow use of servoing without AR tags. For manual base positioning

    def __init__(self):
        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.lock = Lock()
        self.robot_pose = None
        self.target_pose = None
        self.robot_B_world = np.matrix(np.eye(4))
        self.world_B_reference = None
        now = rospy.Time.now()
        self.tf_listener.waitForTransform('/base_link', '/r_forearm_cam_optical_frame', rospy.Time(), rospy.Duration(15.0))
        (trans, rot) = self.tf_listener.lookupTransform('/base_link', '/r_forearm_cam_optical_frame', rospy.Time())
        self.base_link_B_camera = createBMatrix(trans, rot)
        self.robot_center_pub = rospy.Publisher('/robot_frame', PoseStamped, latch=True)
        self.head_center_pub = rospy.Publisher('/head_frame', PoseStamped, latch=True)
        self.reference_pub = rospy.Publisher('/reference', PoseStamped, latch=True)
        self.ar_pub = rospy.Publisher("/l_pr2_ar_pose_marker", ARMarker, latch=True)
        self.robot_sub = rospy.Subscriber('/robot_back/pose', TransformStamped, self.robot_cb)
        self.head_sub = rospy.Subscriber('/head_back/pose', TransformStamped, self.head_cb)
        self.reference_sub = rospy.Subscriber('/reference_back/pose', TransformStamped, self.reference_cb)
        
        
        print 'The tf_spoofer has initialized without a problem, as far as I can tell!'

    def robot_cb(self, data):
        with self.lock:
            trans = [data.transform.translation.x,
                     data.transform.translation.y,
                     data.transform.translation.z]
            rot = [data.transform.rotation.x,
                   data.transform.rotation.y,
                   data.transform.rotation.z,
                   data.transform.rotation.w]
            world_B_robot_back = createBMatrix(trans, rot)
            robot_back_B_robot_center = np.matrix([[1., 0., 0., (.665/2+.02)],
                                                [0., 1., 0., 0.],
                                                [0., 0., 1., 0.],
                                                [0., 0., 0., 1.]])
            #world_B_robot = world_B_robot_back  # *robot_back_B_base_link

            z_origin = np.array([0, 0, 1])
            x_robot = np.array([world_B_robot_back[0, 0], world_B_robot_back[1, 0], world_B_robot_back[2, 0]])
            y_robot_project = np.cross(z_origin, x_robot)
            y_robot_project = y_robot_project/np.linalg.norm(y_robot_project)
            x_robot_project = np.cross(y_robot_project, z_origin)
            x_robot_project = x_robot_project/np.linalg.norm(x_robot_project)
            world_B_robot_back_project = np.eye(4)
            for i in xrange(3):
                world_B_robot_back_project[i, 0] = x_robot_project[i]
                world_B_robot_back_project[i, 1] = y_robot_project[i]
                world_B_robot_back_project[i, 3] = world_B_robot_back[i, 3]
            world_B_robot = np.matrix(world_B_robot_back_project)*robot_back_B_robot_center
            world_B_robot[2, 3] = 0.


            pos, ori = Bmat_to_pos_quat(world_B_robot.I)
            pos[2] = 0.
            self.robot_B_world = createBMatrix(pos, ori)
            psm = PoseStamped()
            psm.header.frame_id = '/base_link'
            psm.pose.position.x = pos[0]
            psm.pose.position.y = pos[1]
            psm.pose.position.z = pos[2]
            psm.pose.orientation.x = ori[0]
            psm.pose.orientation.y = ori[1]
            psm.pose.orientation.z = ori[2]
            psm.pose.orientation.w = ori[3]
            self.robot_center_pub.publish(psm)
            self.tf_broadcaster.sendTransform((pos[0], pos[1], 0.), (ori[0], ori[1], ori[2], ori[3]),
                                              rospy.Time.now(), '/optitrak', '/base_link')
            # world_B_pr2 = createBMatrix(trans, rot)
            # self.robot_pose = world_B_pr2
            # self.update_feedback()

    def head_cb(self, data):
        with self.lock:

            trans = [data.transform.translation.x,
                     data.transform.translation.y,
                     data.transform.translation.z]
            rot = [data.transform.rotation.x,
                   data.transform.rotation.y,
                   data.transform.rotation.z,
                   data.transform.rotation.w]
            world_B_head_back = createBMatrix(trans, rot)
            head_back_B_head_center = np.matrix([[1., 0., 0., -0.07],
                                                 [0., 1., 0., 0.],
                                                 [0., 0., 1., -0.075],
                                                 [0., 0., 0., 1.]])
            pos, ori = Bmat_to_pos_quat(world_B_head_back*head_back_B_head_center)
            psm = PoseStamped()
            psm.header.frame_id = data.header.frame_id
            psm.pose.position.x = pos[0]
            psm.pose.position.y = pos[1]
            psm.pose.position.z = pos[2]
            psm.pose.orientation.x = ori[0]
            psm.pose.orientation.y = ori[1]
            psm.pose.orientation.z = ori[2]
            psm.pose.orientation.w = ori[3]
            self.head_center_pub.publish(psm)
            # world_B_pr2 = createBMatrix(trans, rot)
            # self.robot_pose = world_B_pr2
            # self.update_feedback()
            self.tf_broadcaster.sendTransform((pos[0], pos[1], pos[2]), (ori[0], ori[1], ori[2], ori[3]),
                                              rospy.Time.now(), '/head_frame', "/optitrak")

    def reference_cb(self, data):
        with self.lock:

            trans = [data.transform.translation.x,
                     data.transform.translation.y,
                     data.transform.translation.z]
            rot = [data.transform.rotation.x,
                   data.transform.rotation.y,
                   data.transform.rotation.z,
                   data.transform.rotation.w]
            world_B_reference_back = createBMatrix(trans, rot)

            reference_back_B_reference_center = np.matrix([[1., 0., 0., 0.06],
                                                           [0., 1., 0., 0.],
                                                           [0., 0., 1., -0.74],
                                                           [0., 0., 0., 1.]])

            z_origin = np.array([0, 0, 1])
            x_ref = np.array([world_B_reference_back[0, 0], world_B_reference_back[1, 0], world_B_reference_back[2, 0]])
            y_ref_project = np.cross(z_origin, x_ref)
            y_ref_project = y_ref_project/np.linalg.norm(y_ref_project)
            x_ref_project = np.cross(y_ref_project, z_origin)
            x_ref_project = x_ref_project/np.linalg.norm(x_ref_project)
            world_B_ref_back_project = np.eye(4)
            for i in xrange(3):
                world_B_ref_back_project[i, 0] = x_ref_project[i]
                world_B_ref_back_project[i, 1] = y_ref_project[i]
                world_B_ref_back_project[i, 3] = world_B_reference_back[i, 3]
            world_B_ref = np.matrix(world_B_ref_back_project)*reference_back_B_reference_center
            world_B_ref[2, 3] = 0.

            
            # world_B_reference = world_B_reference_back*reference_back_B_reference
            # camera_B_reference = ((self.robot_B_world.I)*self.base_link_B_camera).I*world_B_reference
            pos, ori = Bmat_to_pos_quat(world_B_ref)
            psm = PoseStamped()
            psm.header.frame_id = '/optitrak'
            psm.pose.position.x = pos[0]
            psm.pose.position.y = pos[1]
            psm.pose.position.z = pos[2]
            psm.pose.orientation.x = ori[0]
            psm.pose.orientation.y = ori[1]
            psm.pose.orientation.z = ori[2]
            psm.pose.orientation.w = ori[3]
            self.reference_pub.publish(psm)
            self.tf_broadcaster.sendTransform((pos[0], pos[1], pos[2]), (ori[0], ori[1], ori[2], ori[3]),
                                              rospy.Time.now(), '/reference', "/optitrak")
            
            ar_m = ARMarker()
            ar_m.header.frame_id = '/r_forearm_cam_optical_frame'
            ar_m.pose.pose.position.x = pos[0]
            ar_m.pose.pose.position.y = pos[1]
            ar_m.pose.pose.position.z = pos[2]
            ar_m.pose.pose.orientation.x = ori[0]
            ar_m.pose.pose.orientation.y = ori[1]
            ar_m.pose.pose.orientation.z = ori[2]
            ar_m.pose.pose.orientation.w = ori[3]
            self.ar_pub.publish(ar_m)
            
            # self.tf_broadcaster.sendTransform((pos[0], pos[1], pos[2]), (ori[0], ori[1], ori[2], ori[3]),
            #                                   rospy.Time.now(), '/ar_marker', "/r_arm_camera")

if __name__ == '__main__':
    rospy.init_node('tf_spoofer')
    # myrobot = '/base_location'
    # mytarget = '/goal_location'
    tf_spoofer = TF_Spoofer()
    rospy.spin()

