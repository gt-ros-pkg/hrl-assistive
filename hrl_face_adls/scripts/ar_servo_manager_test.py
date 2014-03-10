#!/usr/bin/env python

from threading import Lock

import roslib; roslib.load_manifest('hrl_face_adls')
import rospy
from std_msgs.msg import String, Int32
from geometry_msgs.msg import PoseStamped, Point, Quaternion

from hrl_pr2_ar_servo.msg import ARServoGoalData


class ServoingManager(object):
    """ Manager for providing test goals to pr2 ar servoing. """
    def __init__(self):
        self.goal_data_pub = rospy.Publisher("ar_servo_goal_data", ARServoGoalData)
        self.ui_input_sub = rospy.Subscriber("action_location_goal", String, self.ui_cb)
        self.lock = Lock()
        self.action = None
        self.location = None
        self.marker_topic = None

    def ui_cb(self, msg):
        print "Got callback"
        with self.lock:
            self.action = "scratch"
            self.location = "nose"
            self.goal_pose = self.goal_pose_from_location(self.location)
            self.marker_topic = "r_pr2_ar_pose_marker" #based on location
        print "callback done"

    def goal_pose_from_location(self, location):
        ps = PoseStamped()
        ps.header.frame_id = 'base_link'
        ps.pose.position = Point(1,1,0)
        ps.pose.orientation = Quaternion(1,0,0,0)
        return ps

    def get_servo_goal(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            print "CHecking for data"
            if (self.action is None) or (self.location is None):
                rate.sleep()
                continue
            break
        ar_data = ARServoGoalData()
        with self.lock:
            ar_data.tag_id = -1
            ar_data.marker_topic = self.marker_topic
            ar_data.base_pose_goal = self.goal_pose
            self.action = None
            self.location = None
        self.goal_data_pub.publish(ar_data)
        print "Published"


if __name__=='__main__':
    rospy.init_node('ar_servo_manager')
    manager = ServoingManager()
    while not rospy.is_shutdown():
        manager.get_servo_goal()
