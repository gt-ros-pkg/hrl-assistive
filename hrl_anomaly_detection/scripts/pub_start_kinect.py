import rospy
import numpy as np
from sensor_msgs.msg import *
from std_msgs.msg import String

def callback(data):
    print data
    print np.asarray(data.K).reshape((3,3))


def listener():
    rospy.init_node('kinect_starter_for_anomal')
    pub = rospy.Publisher("/head_mount_kinect/pause_kinect", String, queue_size=10)
    while not rospy.is_shutdown():
        raw_input("press any key to start kinect")
        pub.publish("start")

if __name__ == '__main__':
    listener()
