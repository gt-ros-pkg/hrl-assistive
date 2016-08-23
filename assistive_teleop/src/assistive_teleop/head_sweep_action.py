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
        self.start_merge_client = rospy.ServiceProxy("/merge_pointclouds/new_merge", Trigger)
        self.retrieve_merge_client = rospy.ServiceProxy("/merge_pointclouds/get_merged_pointcloud", GetMergedPointcloud)
        self.set_environment_client = rospy.ServiceProxy("/set_environment_model", SetBaseModel)
        self.head_joint_action = actionlib.SimpleActionClient("/head_traj_controller/joint_trajectory_action", JointTrajectoryAction)
        if not self.head_joint_action.wait_for_server(rospy.Duration(5)):
            rospy.logwarn("[%s] Cannot find head joint trajectory action server", rospy.get_name())
        self.action_server = actionlib.SimpleActionServer("head_sweep_action",
                                                          HeadSweepAction,
                                                          self.execute_cb,
                                                          auto_start=False)
        self.action_server.register_preempt_callback(self.preempt_cb)
        self.action_server.start()
        rospy.loginfo("[%s] Started Action Server", rospy.get_name())

    def preempt_cb(self):
        self.head_joint_action.cancel_goal()

    def execute_cb(self, goal_msg):
        # Move to start of sweep path
        jt = JointTrajectory()
        jt.joint_names = goal_msg.sweep_trajectory.joint_names
        jtp = JointTrajectoryPoint()
        jtp.positions = goal_msg.sweep_trajectory.points[0].positions
        jtp.time_from_start = rospy.Duration(2)
        jt.points.append(jtp)
        head_scan_traj = JointTrajectoryGoal()
        head_scan_traj.trajectory = jt
        try:
            self.head_joint_action.send_goal(head_scan_traj)
            self.head_joint_action.wait_for_result(rospy.Duration(4))
        except Exception as e:
            self.report_failure(e)
            return
        try:
            # Start kinect Scan + Merge
            self.start_merge_client.call()
        except Exception as e:
            self.report_failure(e)
            return
        rospy.sleep(0.5)
        # Move along sweep path
        try:
            jtp.positions = goal_msg.sweep_trajectory.points[-1].positions
            jtp.time_from_start = goal_msg.sweep_trajectory.points[-1].time_from_start
            if goal_msg.sweep_trajectory.points[-1].time_from_start < rospy.Duration(3):
                self.report_failure("Bad timing.")
                return
            self.head_joint_action.send_goal(head_scan_traj)
            timeout = goal_msg.sweep_trajectory.points[-1].time_from_start + rospy.Duration(3)
            self.head_joint_action.wait_for_result(timeout)
        except Exception as e:
            self.report_failure(e)
            return
        try:
            # Receive Pointcloud from Kinect Sweep
            merged_point_cloud = self.retrieve_merge_client().merged_pointcloud
        except Exception as e:
            self.report_failure(e)
            return
        try:
            # Set pointcloud environment in Base Selection
            self.set_environment_client.call(merged_point_cloud)
        except Exception as e:
            self.report_failure(e)
            return
        # Return result
        result = HeadSweepResult()
        self.action_server.set_succeeded(result)

    def report_failure(self, e):
        rospy.logerr("[%s] Head Sweep Action Failed: %s", rospy.get_name(), e)
        self.action_server.set_aborted()


def main():
    rospy.init_node("head_sweep_action_node")
    hsa = HeadSweepActionServer()
    rospy.spin()
