#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import moveit_msgs.msg as mm
from geometry_msgs.msg import PoseStamped

from assistive_teleop.srv import MoveItPlan


class MoveItInterface(object):
    def __init__(self):
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.groups = {'right_arm': moveit_commander.MoveGroupCommander("right_arm"),
                       'left_arm': moveit_commander.MoveGroupCommander("left_arm")}
        for group in self.groups.itervalues():
            group.set_goal_tolerance(0.02)

        self.traj_display_pub = rospy.Publisher("/move_group/display_planned_path", mm.DisplayTrajectory)
        self.debug_pub = rospy.Publisher("plan_goal", PoseStamped)
        self.services = {'right_arm': rospy.Service("/moveit_plan/right_arm", MoveItPlan, self.r_service_cb),
                         'left_arm': rospy.Service("/moveit_plan/left_arm", MoveItPlan, self.l_service_cb)}

    def r_service_cb(self, req):
        return self.get_plan(req.pose_target, "right_arm")

    def l_service_cb(self, req):
        return self.get_plan(req.pose_target, "left_arm")

    def get_plan(self, ps_msg, arm):
        self.debug_pub.publish(ps_msg)
        self.groups[arm].set_pose_target(ps_msg.pose)
        plan = self.groups[arm].plan()
        # Publish display of trajectory plan
        disp = mm.DisplayTrajectory()
        disp.trajectory_start = self.robot.get_current_state()
        disp.trajectory.append(plan)
        self.traj_display_pub.publish(disp)
        # return joint trajectory from plan
        return plan


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface', anonymous=True)
    interface = MoveItInterface()
    rospy.spin()
