#!/usr/bin/env python

import roslib
roslib.load_manifest("hrl_feeding_task")
roslib.load_manifest("hrl_haptic_mpc")
import rospy
from std_msgs.msg import String, Bool


class emergencyArmStop():
    def __init__(self):
        self.stopPub = rospy.Publisher('emergency_arm_stop', String, latch = False)
        rospy.init_node('emergency_arm_stop_publisher', anonymous = False)
        self.rate = rospy.Rate(10)


    def checkStop(self):
        while not rospy.is_shutdown():
            stopMsg = String()
            keyIn = raw_input("Enter '!' to stop arm motion: ")
            if keyIn == '!':
                stopMsg = '!'
                self.stopPub.publish(stopMsg)
                self.rate.sleep()

if __name__ == '__main__':
    emergency = emergencyArmStop()
    try:
        emergency.checkStop()
    except rospy.ROSInterruptException:
		pass
