#!/usr/bin/env python

from threading import Lock
import copy

import roslib
roslib.load_manifest('hrl_face_adls')
import rospy
from std_msgs.msg import String, Int32
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf import TransformListener

from hrl_pr2_ar_servo.msg import ARServoGoalData
from hrl_base_selection.srv import BaseMove, BaseMoveRequest
from hrl_ellipsoidal_control.msg import EllipsoidParams

POSES = {'knee': ([0.443, -0.032, -0.716], [0.162, 0.739, 0.625, 0.195]),
         'arm': ([0.337, -0.228, -0.317], [0.282, 0.850, 0.249, 0.370]),
         'shoulder': ([0.108, -0.236, -0.105], [0.346, 0.857, 0.238, 0.299]),
         'face': ([0.252, -0.067, -0.021], [0.102, 0.771, 0.628, -0.002])}

class ServoingManager(object):
    """ Manager for providing test goals to pr2 ar servoing. """

    def __init__(self):
        self.goal_data_pub = rospy.Publisher("ar_servo_goal_data", ARServoGoalData)
        self.ui_input_sub = rospy.Subscriber("action_location_goal", String, self.ui_cb)
        self.head_pose_sub = rospy.Subscriber("ellipsoid_params", EllipsoidParams, self.ell_cb)
        self.base_selection_client = rospy.ServiceProxy("select_base_position", BaseMove)
        self.feedback_pub = rospy.Publisher('wt_log_out', String)
        self.servo_goal_pub = rospy.Publisher('servo_goal_pub', PoseStamped)
        self.tfl = TransformListener()
        self.lock = Lock()
        self.head_pose = None
        self.goal_pose = None
        self.marker_topic = None

    def ui_cb(self, msg):
        if self.head_pose is None:
            log_msg = "Must register head before sending action command"
            self.feedback_pub.publish(String(log_msg))
            rospy.loginfo("[%s]" % rospy.get_name() + log_msg)
            return

        loc = msg.data
        if loc not in POSES:
            log_msg = "Invalid goal location. No Known Pose for %s" % loc
            self.feedback_pub.publish(String(log_msg))
            rospy.loginfo("[%s]" % rospy.get_name() + log_msg)
            return
        pos, quat = POSES[loc]
        goal_ps_ell = PoseStamped()
        goal_ps_ell.header.frame_id = 'ellipse_frame'
        goal_ps_ell.pose.position = Point(*pos)
        goal_ps_ell.pose.orientation = Quaternion(*quat)

        now = rospy.Time.now()
        self.tfl.waitForTransform('base_link', goal_ps_ell.header.frame_id, now, rospy.Duration(10))
        goal_ps_ell.header.stamp = now
        goal_ps = self.tfl.transformPose('base_link', goal_ps_ell)
        with self.lock:
            self.action = "touch"
            self.goal_pose = goal_ps
            self.marker_topic = "r_pr2_ar_pose_marker"  # based on location

        ar_data = ARServoGoalData()
        base_goal = self.call_base_selection()
        self.servo_goal_pub.publish(base_goal)

        with self.lock:
            ar_data.tag_id = -1
            ar_data.marker_topic = self.marker_topic
            ar_data.base_pose_goal = base_goal
            self.action = None
            self.location = None
        self.goal_data_pub.publish(ar_data)

    def call_base_selection(self):
        bm = BaseMoveRequest()
        bm.head = self.head_pose
        bm.goal = self.goal_pose
        return self.base_selection_client.call(bm).BaseGoal

    def ell_cb(self, ell_msg):
        head_pose = PoseStamped()
        head_pose.header = copy.copy(ell_msg.e_frame.header)
        head_pose.pose.position.x = ell_msg.e_frame.transform.translation.x
        head_pose.pose.position.y = ell_msg.e_frame.transform.translation.y
        head_pose.pose.position.z = ell_msg.e_frame.transform.translation.z
        head_pose.pose.orientation.x = ell_msg.e_frame.transform.rotation.x
        head_pose.pose.orientation.y = ell_msg.e_frame.transform.rotation.y
        head_pose.pose.orientation.z = ell_msg.e_frame.transform.rotation.z
        head_pose.pose.orientation.w = ell_msg.e_frame.transform.rotation.w
        now = rospy.Time.now()
        self.tfl.waitForTransform('base_link', head_pose.header.frame_id, now, rospy.Duration(10))
        head_pose.header.stamp = now
        self.head_pose = self.tfl.transformPose("base_link", head_pose)


if __name__ == '__main__':
    rospy.init_node('ar_servo_manager')
    manager = ServoingManager()
    rospy.spin()
