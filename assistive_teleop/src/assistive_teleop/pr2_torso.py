#!/usr/bin/python

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class PR2Torso():
    def __init__(self):
	self.pub = rospy.Publisher("/torso_controller/command",JointTrajectory)

    def set_position(self,goal):
	traj = JointTrajectory()
	traj_p = JointTrajectoryPoint()
	traj.joint_names = ["torso_lift_joint"]
	traj_p.positions = [goal]
	traj_p.velocities = [0.]
	traj_p.accelerations = [0.]
	traj_p.time_from_start = rospy.Duration(2.)
	traj.points.append(traj_p)
	self.pub.publish(traj)

if __name__=='__main__':
    rospy.init_node('move_torso')
    mt = PR2Torso()
    mt.set_position(0.015)
    rospy.spin()
