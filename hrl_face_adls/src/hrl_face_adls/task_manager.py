#!/usr/bin/env python

from threading import Lock
import copy
import numpy as np

import roslib
roslib.load_manifest('hrl_face_adls')
import rospy
from std_msgs.msg import String, Int32, Int8, Bool
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf import TransformListener, transformations as tft

from hrl_pr2_ar_servo.msg import ARServoGoalData
from hrl_base_selection.srv import BaseMove_multi  # , BaseMoveRequest
from hrl_ellipsoidal_control.msg import EllipsoidParams

POSES = {'Knee': ([0.443, -0.032, -0.716], [0.162, 0.739, 0.625, 0.195]),
         'Arm': ([0.337, -0.228, -0.317], [0.282, 0.850, 0.249, 0.370]),
         'Shoulder': ([0.108, -0.236, -0.105], [0.346, 0.857, 0.238, 0.299]),
         'Face': ([0.252, -0.067, -0.021], [0.102, 0.771, 0.628, -0.002])}

class ServoingManager(object):
    """ Manager for providing test goals to pr2 ar servoing. """

    def __init__(self):
        self.task = 'yogurt'
        self.model = 'chair'

        self.tfl = TransformListener()

        self.goal_data_pub = rospy.Publisher("ar_servo_goal_data", ARServoGoalData)
        self.servo_goal_pub = rospy.Publisher('servo_goal_pose', PoseStamped, latch=True)
        self.reach_goal_pub = rospy.Publisher("arm_reacher/goal_pose", PoseStamped)
        self.test_pub = rospy.Publisher("test_goal_pose", PoseStamped, latch=True)
        self.test_head_pub = rospy.Publisher("test_head_pose", PoseStamped, latch=True)
        self.feedback_pub = rospy.Publisher('wt_log_out', String)

        self.base_selection_client = rospy.ServiceProxy("select_base_position", BaseMove_multi)

        self.ui_input_sub = rospy.Subscriber("action_location_goal", String, self.ui_cb)
        self.servo_fdbk_sub = rospy.Subscriber("/pr2_ar_servo/state_feedback", Int8, self.servo_fdbk_cb)

        self.lock = Lock()
        self.head_pose = None
        self.goal_pose = None
        self.marker_topic = None
        rospy.loginfo("[%s] Ready" %rospy.get_name())

    def servo_fdbk_cb(self, msg):
        if not msg.data == 5:
            return
        #self.reach_goal_pub.publish(self.goal_pose)
        #self.feedback_pub.publish("Servoing succeeded. Reaching to location.")
        #rospy.loginfo("Servoing Succeeded. Sending goal to arm reacher.")
        msg = "Servoing Succeeded. Please proceed by bringing up the arms."
        self.feedback_pub.publish(msg)
        rospy.loginfo(msg)

    def ui_cb(self, msg):
        self.head_pose = self.get_head_pose()
        if self.head_pose is None:
            log_msg = "Please register your head before sending a task."
            self.feedback_pub.publish(String(log_msg))
            rospy.loginfo("[%s] %s" % (rospy.get_name(), log_msg))
            return
        rospy.loginfo("[%s] Found head frame" % rospy.get_name());
        self.test_head_pub.publish(self.head_pose)

        loc = msg.data
        if loc not in POSES:
            log_msg = "Invalid goal location. No Known Pose for %s" % loc
            self.feedback_pub.publish(String(log_msg))
            rospy.loginfo("[%s]" % rospy.get_name() + log_msg)
            return
        rospy.loginfo("[%s] Received valid goal location: %s" % (rospy.get_name(), loc))
        pos, quat = POSES[loc]
        goal_ps_ell = PoseStamped()
        goal_ps_ell.header.frame_id = 'head_frame'
        goal_ps_ell.pose.position = Point(*pos)
        goal_ps_ell.pose.orientation = Quaternion(*quat)

        now = rospy.Time.now()
        self.tfl.waitForTransform('base_link', goal_ps_ell.header.frame_id, now, rospy.Duration(3))
        goal_ps_ell.header.stamp = now
        goal_ps = self.tfl.transformPose('base_link', goal_ps_ell)
        self.test_pub.publish(goal_ps)
        with self.lock:
            self.action = "touch"
            self.goal_pose = goal_ps
            self.marker_topic = "r_pr2_ar_pose_marker"  # based on location

        ar_data = ARServoGoalData()
        base_goal, configuration_goal = self.call_base_selection()
        print "Base Goals returned:\r\n", base_goal
        if base_goal is None:
            rospy.loginfo("No base goal found")
            return
        base_goals_list = []
        for i in xrange(int(len(base_goal)/7)):
            psm = PoseStamped()
            psm.header.frame_id = '/base_link'
            psm.pose.position.x = base_goal[int(0+7*i)]
            psm.pose.position.y = base_goal[int(1+7*i)]
            psm.pose.position.z = base_goal[int(2+7*i)]
            psm.pose.orientation.x = base_goal[int(3+7*i)]
            psm.pose.orientation.y = base_goal[int(4+7*i)]
            psm.pose.orientation.z = base_goal[int(5+7*i)]
            psm.pose.orientation.w = base_goal[int(6+7*i)]
            base_goals_list.append(copy.copy(psm))
        # Here should publish configuration_goal items to robot Z axis and to Autobed.
        self.servo_goal_pub.publish(base_goals_list[0])

        with self.lock:
            ar_data.tag_id = -1
            ar_data.marker_topic = self.marker_topic
            ar_data.base_pose_goal = base_goal
            self.action = None
            self.location = None
        self.feedback_pub.publish("Base Position Found. Please use servoing tool.")
        rospy.loginfo("[%s] Base position found. Sending Servoing goals." % rospy.get_name())
        self.goal_data_pub.publish(ar_data)

    def call_base_selection(self):
        # Place holder return
        #bg = PoseStamped()
        #bg.header.stamp = rospy.Time.now()
        #bg.header.frame_id = 'ar_marker'
        #bg.pose.position = Point(0., 0., 0.5)
        #q = tft.quaternion_from_euler(0., np.pi/2, 0.)
        #bg.pose.orientation = Quaternion(*q)
        #return bg
        ## End Place Holder
        self.feedback_pub.publish("Finding a good base location, please wait.")
        rospy.loginfo("[%s] Calling base selection. Please wait." %rospy.get_name())

        # bm = BaseMoveRequest()
        # bm.model = self.model
        # bm.task = self.task
        try:
            resp = self.base_selection_client(self.task, self.model)
            # resp = self.base_selection_client.call(bm)
        except rospy.ServiceException as se:
            rospy.logerr(se)
            self.feedback_pub.publish("Failed to find good base position. Please try again.")
            return None
        return resp.base_goal, resp.configuration_goal

    def get_head_pose(self, head_frame="head_frame"):
        try:
            now = rospy.Time.now()
            self.tfl.waitForTransform("/base_link", head_frame, now, rospy.Duration(3))
            pos, quat = self.tfl.lookupTransform("/base_link", head_frame, now)
            head_pose = PoseStamped()
            head_pose.header.frame_id = "/base_link"
            head_pose.header.stamp = now
            head_pose.pose.position = Point(*pos)
            head_pose.pose.orientation = Quaternion(*quat)
            return head_pose
        except Exception as e:
            rospy.loginfo("TF Exception:\r\n%s" %e)
            return None


if __name__ == '__main__':
    rospy.init_node('ar_servo_manager')
    manager = ServoingManager()
    import sys
    rospy.spin()
