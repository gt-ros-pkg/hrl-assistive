#!/usr/bin/env python

from threading import Lock

import roslib; roslib.load_manifest('hrl_face_adls')
import rospy
from geometry_msgs.msg import PoseStamped

from hrl_pr2_ar_servo.msg import ARServoGoalData


class ServoingManager(object):
    """ Manager for providing test goals to pr2 ar servoing. """
    def __init__(self):
        self.goal_data_pub = rospy.Publisher("ar_servo_goal_data", ARServoGoalData)
        self.ui_input_sub = rospy.Subscriber("action_location_goal", String, self.ui_cb)
        self.lock = Lock()
        self.action = None
        self.location = None
        self.camera_topic = None

    def ui_cb(self, msg):
        with self.lock:
            self.action = "scratch"
            self.location = "nose"
            self.goal_pose = self.goal_pose_from_location(self.location)
            self.camera_topic = "r_forearm_camera/image_color" #based on location

    def goal_pose_from_location(self, location):
        ps = PoseStamped()
        ps.header.frame_id = 'base_link'
        ps.pose.position = Point(1,1,0)
        ps.pose.orientation = Quaternion(1,0,0,0)
        return ps

    def get_servo_goal(self)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if (self.action is None) or (self.location is None):
                rate.sleep()
                continue
        ar_data = ARServoGoalData()
        with self.lock:
            ar_data.tag_id = -1
            ar_data.camera_topic = self.camera_topic
            ar_data.base_pose_goal = self.goal_pose
        self.publish(ar_data)



if __name__=='__main__':
    manager = ServoingManager()
