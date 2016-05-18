#!/usr/bin/env python

import rospy
import actionlib

from geometry_msgs.msg import Point, Vector3
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pr2_controllers_msgs.msg import PointHeadAction, PointHeadGoal, Pr2GripperCommandAction, Pr2GripperCommandGoal, Pr2GripperCommand


def reset_torso(torso_pub):
    trajPoint = JointTrajectoryPoint()
    trajPoint.positions = [0.325]
    trajPoint.velocities = [0]
    trajPoint.accelerations = [0]
    trajPoint.time_from_start = rospy.Duration(1)
    traj = JointTrajectory()
    traj.joint_names = ['torso_lift_joint']
    traj.points.append(trajPoint)
    torso_pub.publish(traj)


def reset_head(head_action_client):
    goal = PointHeadGoal()
    goal.target.header.frame_id = 'base_footprint'
    goal.target.header.stamp = rospy.Time.now()
    goal.target.point = Point(0.63, 0.0, 0.7)  # Center of the experimental table
    goal.pointing_axis = Vector3(0, 0, 1)
    goal.pointing_frame = 'head_mount_kinect_rgb_optical_frame'
    goal.max_velocity = 0.1
    head_action_client.send_goal(goal)


def reset_gripper(side, gripper_client):
    cmd = Pr2GripperCommand(0.09, -1.0)
    goal = Pr2GripperCommandGoal(cmd)
    gripper_client.send_goal(goal)


def reset_arm(side, arm_pub):
    traj_point = JointTrajectoryPoint()
    if side == 'right':
        traj_point.positions = [-0.88, 1.05, -1.47, -1.6, -4.14, -0.64, 4.61]
    else:
        traj_point.positions = [0.88, 1.05, 1.47, -1.6, 4.14, -0.64, 1.51]
    traj_point.velocities = [0.0]*7
    traj_point.accelerations = [0.0]*7
    traj_point.time_from_start = rospy.Duration(5)
    traj = JointTrajectory()
    traj.joint_names = ['%s_shoulder_pan_joint' % side[0],
                        '%s_shoulder_lift_joint' % side[0],
                        '%s_upper_arm_roll_joint' % side[0],
                        '%s_elbow_flex_joint' % side[0],
                        '%s_forearm_roll_joint' % side[0],
                        '%s_wrist_flex_joint' % side[0],
                        '%s_wrist_roll_joint' % side[0]]
    traj.points.append(traj_point)
    traj_point = JointTrajectoryPoint()
    traj_point.velocities = [0.0]*7
    traj_point.accelerations = [0.0]*7
    traj_point.time_from_start = rospy.Duration(5)
    arm_pub.publish(traj)


def init():
    """ Initialize publishers and clients """
    torso_pub = rospy.Publisher('/torso_controller/command', JointTrajectory, queue_size=1)
    arm_pubs = {'right': rospy.Publisher('/right_arm/haptic_mpc/joint_trajectory', JointTrajectory, queue_size=1),
                'left': rospy.Publisher('/left_arm/haptic_mpc/joint_trajectory', JointTrajectory, queue_size=1)}
    head_action_client = actionlib.SimpleActionClient('/head_traj_controller/point_head_action', PointHeadAction)
    gripper_clients = {'right': actionlib.SimpleActionClient('/r_gripper_sensor_controller/gripper_action', Pr2GripperCommandAction),
                       'left': actionlib.SimpleActionClient('/l_gripper_sensor_controller/gripper_action', Pr2GripperCommandAction)}
    rospy.sleep(1.5)  # Wait for connections to be made
    return (torso_pub, head_action_client, arm_pubs, gripper_clients)


def main():
    rospy.init_node('arat_reset')
    torso_pub, head_ac, arm_pubs, gripper_acs = init()

    reset_torso(torso_pub)
    reset_head(head_ac)
    reset_arm('right', arm_pubs['right'])
    reset_arm('left', arm_pubs['left'])
    reset_gripper('right', gripper_acs['right'])
    reset_gripper('left', gripper_acs['left'])


if __name__ == '__main__':
    main()
