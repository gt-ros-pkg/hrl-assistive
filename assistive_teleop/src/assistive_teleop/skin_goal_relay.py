#!/usr/bin/env python

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
from geometry_msgs.msg import PoseStamped
from tf import TransformListener

from std_msgs.msg import String, Bool

import pose_utils as pu

class SkinGoalRelay(object):
    def __init__(self):
        rospy.Subscriber('wt_skin_goal', PoseStamped, self.skin_goal_cb)
        self.skin_goal_pub = rospy.Publisher('skin_goal_out', PoseStamped)
        self.go_goal_pub = rospy.Publisher('/epc_skin/command/behavior', String)
        self.stop_epc_pub = rospy.Publisher('/epc/stop', Bool)
        self.tfl = TransformListener()

    def skin_goal_cb(self, goal_ps):
        self.stop_epc_pub.publish(True)
        rospy.sleep(0.3)
        self.stop_epc_pub.publish(False)
        rospy.sleep(0.1)
        flipped_pose = pu.pose_relative_rot(goal_ps, p=180)
        backed_off_pose = pu.pose_relative_trans(flipped_pose, x=0.0)
        backed_off_pose.header.stamp = rospy.Time(0)
        torso_pose = self.tfl.transformPose('/torso_lift_link', backed_off_pose)
        torso_pose.header.stamp = rospy.Time.now()
        self.skin_goal_pub.publish(torso_pose)
        rospy.sleep(0.25)
        self.go_goal_pub.publish('go_to_way_point')


if __name__=='__main__':
    rospy.init_node('wt_skin_goal_relay')
    sgr = SkinGoalRelay()
    rospy.spin()
