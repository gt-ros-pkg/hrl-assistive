#!/usr/bin/python

import math

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
import actionlib
from geometry_msgs.msg import PoseStamped, WrenchStamped
from assistive_teleop.msg import FtMoveAction, FtMoveFeedback, FtMoveResult


class FtMoveServer(object):
    def __init__(self):
        self.ft_move_server = actionlib.SimpleActionServer('ft_move_action', FtMoveAction, self.ft_move, False) 
        self.ft_move_server.start()
        rospy.Subscriber('/netft_gravity_zeroing/wrench_zeroed', WrenchStamped, self.get_netft_state)
        self.pose_out = rospy.Publisher('/l_cart/command_pose', PoseStamped)

    def get_netft_state(self, ws):
        self.netft_wrench = ws
        self.ft_mag = math.sqrt(ws.wrench.force.x**2 + ws.wrench.force.y**2 + ws.wrench.force.z**2)

    def ft_move(self, goal):
        if goal.ignore_ft:
            rospy.loginfo("Moving WITHOUT monitoring FT Sensor")
        else:
            rospy.loginfo("Moving while monitoring Force-Torque Sensor")
        update_rate=rospy.Rate(200)
        pub_period = rospy.Duration(0.05)
        pose_count = 0
        result = FtMoveResult()
        feedback = FtMoveFeedback()
        feedback.total = len(goal.poses)
        previous_pose_sent = rospy.Time.now() #+ pub_period
        while not rospy.is_shutdown():
            if not self.ft_move_server.is_active():
                rospy.loginfo("FtMoveAction Goal No Longer Active")
                break
            
            if self.ft_move_server.is_new_goal_available():
                self.ft_move_server.accept_new_goal()
            
            if self.ft_move_server.is_preempt_requested():
                self.ft_move_server.set_preempted()
                rospy.loginfo("Force-Torque Move Action Server Goal Preempted")
                break
            
            if not goal.ignore_ft:
                if self.ft_mag > goal.force_thresh:
                    result.contact = True
                    result.all_sent = False
                    self.ft_move_server.set_aborted(result, "Contact Detected")
                    rospy.loginfo("Force-Torque Move Action Aborted: Contact Detected")
                    break

            if rospy.Time.now()-previous_pose_sent >= pub_period:
                feedback.current = pose_count
                self.ft_move_server.publish_feedback(feedback)
                previous_pose_sent = rospy.Time.now()
                self.pose_out.publish(goal.poses[pose_count])
                pose_count += 1

                if pose_count >= feedback.total: #Total number of poses in list
                    result.contact = False
                    result.all_sent = True
                    self.ft_move_server.set_succeeded(result)
                    break
            update_rate.sleep()

if __name__ == '__main__':
    rospy.init_node('ft_move_action_server')
    ftms = FtMoveServer()
    while not rospy.is_shutdown():
        rospy.spin()
