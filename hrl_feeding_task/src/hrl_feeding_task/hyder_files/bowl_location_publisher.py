#!/usr/bin/env python

import roslib
roslib.load_manifest("hrl_feeding_task")
roslib.load_manifest("hrl_haptic_mpc")
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion


class bowlPublisher():
	def __init__(self):
		self.bowl_pub = rospy.Publisher('bowl_location', Pose, latch = True)
		rospy.init_node('bowl_location_publisher', anonymous = True)
		self.rate = rospy.Rate(10)

	def publish(self):
		while not rospy.is_shutdown():
			position = Point()
			orientation = Quaternion()

			#Actual bowl location, set manually... 
			position.x, position.y, position.z = 0.766, 0.333, -0.287
			#ORIGINAL HARD CODED BOWL POSITION: 0.763, 0.592, -0.301
			orientation.x, orientation.y, orientation.z, orientation.w = 0.686, 0.177, -0.141, 0.691
			#ORIGINAL HARD CODE BOWL ORIENTATION:  0.706, 0.068, -0.235, 0.664
			pose_msg = Pose(position, orientation)
			rospy.loginfo(pose_msg)

			#self.pose_msg = PoseStamped(header = self.hdr, pose = Pose(self.pos, self.quat))

			self.bowl_pub.publish(pose_msg)
			self.rate.sleep()


if __name__ == '__main__':
	publisher = bowlPublisher()
	try:
		publisher.publish()
	except rospy.ROSInterruptException:
		pass


