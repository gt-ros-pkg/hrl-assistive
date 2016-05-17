#!/usr/bin/env python

import rospy

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def set_torso_up():
    torso_pub = rospy.Publisher('/torso_controller/command', JointTrajectory, queue_size=1)
    rospy.sleep(1)
    trajPoint = JointTrajectoryPoint()
    trajPoint.positions = [0.325]
    trajPoint.velocities = [0]
    trajPoint.accelerations = [0]
    trajPoint.time_from_start = rospy.Duration(1)
    traj = JointTrajectory()
    traj.joint_names = ['torso_lift_joint']
    traj.points.append(trajPoint)
    torso_pub.publish(traj)


def set_arm_joints():
    r_arm_pub = rospy.Publisher('/right_arm/haptic_mpc/joint_trajectory', JointTrajectory, queue_size=1)
    l_arm_pub = rospy.Publisher('/left_arm/haptic_mpc/joint_trajectory', JointTrajectory, queue_size=1)
    rospy.sleep(1)
    r_traj_point = JointTrajectoryPoint()
    r_traj_point.positions = [0.0, 1.25, 0.00, -1.50, -3.14, -0.26, 0.0]
    r_traj_point.velocities = [0.0]*7
    r_traj_point.accelerations = [0.0]*7
    r_traj_point.time_from_start = rospy.Duration(5)
    r_traj = JointTrajectory()
    r_traj.joint_names = ['r_shoulder_pan_joint',
                          'r_shoulder_lift_joint',
                          'r_upper_arm_roll_joint',
                          'r_elbow_flex_joint',
                          'r_forearm_roll_joint',
                          'r_wrist_flex_joint',
                          'r_wrist_roll_joint']
    r_traj.points.append(r_traj_point)
    l_traj_point = JointTrajectoryPoint()
    l_traj_point.positions = [0.0, 1.35, 0.00, -1.60, -3.14, -0.3, 0.0]
    l_traj_point.velocities = [0.0]*7
    l_traj_point.accelerations = [0.0]*7
    l_traj_point.time_from_start = rospy.Duration(5)
    l_traj = JointTrajectory()
    l_traj.joint_names = ['l_shoulder_pan_joint',
                          'l_shoulder_lift_joint',
                          'l_upper_arm_roll_joint',
                          'l_elbow_flex_joint',
                          'l_forearm_roll_joint',
                          'l_wrist_flex_joint',
                          'l_wrist_roll_joint']
    l_traj.points.append(l_traj_point)

    r_arm_pub.publish(r_traj)
    l_arm_pub.publish(l_traj)


def main():
    rospy.init_node('arat_reset')
    set_torso_up()
    set_arm_joints()


if __name__ == '__main__':
    main()
