#!/usr/bin/env python

import roslib
roslib.load_manifest("hrl_feeding_task")
roslib.load_manifest("hrl_haptic_mpc")
import rospy
from geometry_msgs.msg import PoseStamped
import std_msgs.msg


class bowlPublisher:
	def __init__(self):

		#MAY NEED TO REMAP ROOT TOPIC NAME GROUP!
		self.bowl_pub = rospy.Publisher('hrl_feeding_task/manual_bowl_location', PoseStamped, latch = False)
		self.head_pub = rospy.Publisher('hrl_feeding_task/manual_head_location', PoseStamped, latch = False)
		rospy.init_node('manual_bowl_head_pose_publisher', anonymous = True)
		self.rate = rospy.Rate(10)
		self.i = 0

		#Trying to simplify code...
		#Create PoseStamped() messages for bowl and head
		self.bowl_pose_manual = PoseStamped()
		self.head_pose_manual = PoseStamped()

		#Instantiate each PoseStamped Header()
		self.bowl_pose_manual.header = std_msgs.msg.Header()
		self.head_pose_manual.header = std_msgs.msg.Header()

		self.bowl_pose_manual.header.frame_id = '/torso_lift_link'
		self.head_pose_manual.header.frame_id = '/torso_lift_link'

		self.bowl_pose_manual.header.stamp = rospy.Time.now()
		self.head_pose_manual.header.stamp = rospy.Time.now()

		self.bowl_pose_manual.header.seq = self.i
		self.head_pose_manual.header.seq = self.i
		
		#The manually set positions and orientations!!!
		(self.bowl_pose_manual.pose.position.x, 
			self.bowl_pose_manual.pose.position.y, 
			self.bowl_pose_manual.pose.position.z) = (0.880, -0.007, -0.305) # 0.928, 0.263, -0.314 | 0.836, 0.496, -0.322 | 0.880, -0.007, -0.305

		(self.bowl_pose_manual.pose.orientation.x,
			self.bowl_pose_manual.pose.orientation.y,
			self.bowl_pose_manual.pose.orientation.z,
			self.bowl_pose_manual.pose.orientation.w) = (0, 0, 0, 1)

		(self.head_pose_manual.pose.position.x,
			self.head_pose_manual.pose.position.y,
			self.head_pose_manual.pose.position.z) = (0.797, -0.365, 0.097) # Mannequin: 0.742, -0.450, -0.049

		(self.head_pose_manual.pose.orientation.x,
			self.head_pose_manual.pose.orientation.y,
			self.head_pose_manual.pose.orientation.z,
			self.head_pose_manual.pose.orientation.w) = (0, 0, 0, 1)


	def publish(self):
		while not rospy.is_shutdown():

			self.bowl_pose_manual.header.stamp = rospy.Time.now()
			self.head_pose_manual.header.stamp = rospy.Time.now()
			self.bowl_pose_manual.header.seq = self.i
			self.head_pose_manual.header.seq = self.i

			self.head_pub.publish(self.head_pose_manual)

			self.bowl_pub.publish(self.bowl_pose_manual)


			rospy.loginfo(self.bowl_pose_manual)
			rospy.loginfo(self.head_pose_manual)

			self.i += 1 
			self.rate.sleep()

if __name__ == '__main__':
	publisher = bowlPublisher()
	try:
		publisher.publish()
	except rospy.ROSInterruptException:
		pass
