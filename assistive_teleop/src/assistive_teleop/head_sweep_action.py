#!/usr/bin/evn python

import rospy
import actionlib
from pr2_controllers_msgs.msg import JointTrajectoryAction, JointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import Trigger

from merge_pointclouds.srv import GetMergedPointcloud

from assistive_teleop.msg import HeadSweepAction, HeadSweepResult

from hrl_base_selection.srv import SetBaseModel


class HeadSweepActionServer(object):
    def __init__(self):
        self.head_joints = ['head_pan_joint', 'head_tilt_joint']
        self.start_merge_client = rospy.ServiceProxy("/merge_pointclouds/new_scan", Trigger)
        self.retrieve_merge_client = rospy.ServiceProxy("/merge_pointclouds/merged_pointcloud", GetMergedPointcloud)
        self.set_environment_client = rospy.ServiceProxy("/set_environment_model", SetBaseModel)
        self.head_joint_action = actionlib.SimpleActionClient("/head_traj_controller/joint_trajectory_action", JointTrajectoryAction)
        if not self.head_joint_action.wait_for_server(rospy.Duration(5)):
            rospy.logwarn("[%s] Cannot find head joint trajectory action server", rospy.get_name())
        self.action_server = actionlib.SimpleActionServer("head_sweep_action",
                                                          HeadSweepAction,
                                                          self.execute_cb,
                                                          auto_start=False)
        self.action_server.start()
        rospy.loginfo("[%s] Started Action Server", rospy.get_name())

    def execute_cb(self, goal_msg):
        # Move to start of sweep path
        jt = JointTrajectory()
        jt.joint_names = self.head_joints
        jtp = JointTrajectoryPoint()
        jtp.positions = [-0.9, -0.5]
        jtp.time_from_start = rospy.Duration(2)
        jt.points.append(jtp)
        head_scan_traj = JointTrajectoryGoal()
        head_scan_traj.trajectory = jt
        try:
            self.head_joint_action.send_goal(head_scan_traj)
            self.head_joint_action.wait_for_result(rospy.Duration(4))
        except Exception as e:
            self.report_failure(e)
        try:
            # Start kinect Scan + Merge
            self.start_merge_client.call()
        except Exception as e:
            self.report_failure(e)
        # Move along sweep path
        jtp.positions = [0.9, -0.5]
        jtp.time_from_start = rospy.Duration(7)
        jt.points = [jtp]
        head_scan_traj = JointTrajectoryGoal()
        head_scan_traj.trajectory = jt
        try:
            self.head_joint_action.send_goal(head_scan_traj)
            self.head_joint_action.wait_for_result(rospy.Duration(10))
        except Exception as e:
            self.report_failure(e)
        try:
            # Receive Pointcloud from Kinect Sweep
            merged_point_cloud = self.retrieve_merge_client()
        except Exception as e:
            self.report_failure(e)
        try:
            # Set pointcloud environment in Base Selection
            self.set_environment_client.call(merged_point_cloud)
        except Exception as e:
            self.report_failure(e)

        # Return result
        result = HeadSweepResult()
        self.action_server.set_succeeded(result)

        def report_failure(self, e):
            rospy.log_err("[%s] Head Sweep Action Failed: %s", rospy.get_name(), e)
            self.action_server.set_aborted()


def main():
    rospy.init_node("head_sweep_action_node")
    hsa = HeadSweepActionServer()
    rospy.spin()
