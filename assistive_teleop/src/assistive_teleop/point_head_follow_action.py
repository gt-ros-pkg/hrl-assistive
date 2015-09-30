#!/usr/bin/env python

import rospy
import actionlib

from pr2_controllers_msgs.msg import PointHeadAction

from hrl_undo.srv import SetActive


class PointHeadFollower(object):
    def __init__(self):
        self.lookatClient = actionlib.SimpleActionClient('/head_traj_controller/point_head_action', PointHeadAction)
        self.lookatClient.wait_for_server()
        self.follow_server = actionlib.SimpleActionServer('point_head_follow_action', PointHeadAction, execute_cb=self.execute_cb, auto_start=False)
        self.follow_server.start()
        self.undo_disable_service_client = rospy.ServiceProxy('undo/move_head/set_active', SetActive)
        rospy.loginfo("[%s] Point Head Follow Action started.", rospy.get_name())

    def execute_cb(self, action_goal_msg):
        goal = action_goal_msg
        done = False
        try:
            self.undo_disable_service_client.call(False)
        except rospy.ServiceException:
            pass
#            rospy.loginfo("[%s] Undo service unavailable, ignoring." % (rospy.get_name()))
        rospy.loginfo("[%s] Tracking (%.3f, %.3f, %.3f) in frame: %s",
                      rospy.get_name(),
                      goal.target.point.x,
                      goal.target.point.y,
                      goal.target.point.z,
                      goal.target.header.frame_id)
        rate = rospy.Rate(5)
        while not rospy.is_shutdown() and not done:
            if self.follow_server.is_new_goal_available():
                msg = "New Goal Received"
                break
            if self.follow_server.is_preempt_requested():
                msg = "Done tracking %s" % goal.target.header.frame_id
                try:
                    self.undo_disable_service_client.call(True)
                except rospy.ServiceException:
                    pass
                break
            goal.target.header.stamp = rospy.Time.now()
            self.lookatClient.send_goal(goal)
            rate.sleep()
        self.follow_server.set_aborted()
        rospy.loginfo("[%s] %s", rospy.get_name(), msg)


def main():
    rospy.init_node('point_head_follow_action_node')
    point_follower = PointHeadFollower()
    rospy.spin()
