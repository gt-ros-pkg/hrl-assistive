#bowl location publisher test code

import roslib
roslib.load_manifest("hrl_feeding_task")
roslib.load_manifest("hrl_haptic_mpc")
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion


class bowlPublisher(self):
	def __init__(self):
		self.bowl_pub = rospy.publisher('bowl_location', Pose, latch = True)
		rospy.init_node('bowl_location_publisher', anonymous = True)
		rate = rospy.Rate(10)

	def publish(self):
		position = Point()
		orientation = Quaternion()

		#Actual bowl location, set manually... 
		position.x, position.y, position.z = 0.763, 0.592, -0.301
		orientation.x, orientation.y, orientation.z, orientation.w = 0.706, 0.068, -0.235, 0.664

		pose_msg = Pose(position, orientation)
		rospy.loginfo(pose_msg)

		#self.pose_msg = PoseStamped(header = self.hdr, pose = Pose(self.pos, self.quat))

		self.bowl_pub(pose_msg)
		rate.sleep()


if __name__ == '__main__':
	publisher = bowlPublisher()
	try:
		publisher.publsh()
	except rospy.ROSInterruptException:
		pass


