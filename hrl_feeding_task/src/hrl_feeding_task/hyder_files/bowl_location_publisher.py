#!/usr/bin/env python

import roslib
roslib.load_manifest("hrl_feeding_task")
roslib.load_manifest("hrl_haptic_mpc")
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
import std_msgs.msg


class bowlPublisher():
	def __init__(self):
		self.bowl_pub = rospy.Publisher('hrl_feeding_task/manual_bowl_location', PoseStamped, latch = True)
		rospy.init_node('bowl_location_publisher', anonymous = True)
		self.rate = rospy.Rate(10)
		self.i = 0

	def publish(self):
		while not rospy.is_shutdown():
			hdr = std_msgs.msg.Header()
			hdr.frame_id = '/torso_lift_link'
			hdr.stamp = rospy.Time.now()
			hdr.seq = self.i

			position = Point()
			orientation = Quaternion()

			#Actual bowl location, set manually...
			position.x, position.y, position.z = 0.732, 0.197, -0.376 #0.300, 0.912, -0.113 #
			#ORIGINAL HARD CODED BOWL POSITION: 0.763, 0.592, -0.301
			orientation.x, orientation.y, orientation.z, orientation.w = 0.528, 0.469, -0.006, 0.708
			#ORIGINAL HARD CODE BOWL ORIENTATION:  0.706, 0.068, -0.235, 0.664
			pose_msg = PoseStamped( header = hdr, pose =  Pose(position, orientation) )
			rospy.loginfo(pose_msg)

			#self.pose_msg = PoseStamped(header = self.hdr, pose = Pose(self.pos, self.quat))

			self.bowl_pub.publish(pose_msg)
			self.i = self.i + 1
			self.rate.sleep()


if __name__ == '__main__':
	publisher = bowlPublisher()
	try:
		publisher.publish()
	except rospy.ROSInterruptException:
		pass
