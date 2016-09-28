#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool

from serial import Serial


class ShaverToggle(object):
    def __init__(self, dev="/dev/ttyUSB0", baudrate=9600):
        self.device = dev
        self.baudrate = baudrate
        self.serial_dev = Serial(dev, baudrate)
        self.toggle_cmd_sub = rospy.Subscriber('toggle_shaver', Bool, self.toggle_shaver)
        rospy.loginfo("[%s] Shaver Toggle Ready.", rospy.get_name())

    def toggle_shaver(self, boolMsg):
        if boolMsg.data:
            self.serial_dev.write('toggle')


def main():
    rospy.init_node('shaver_node')
    toggler = ShaverToggle('/dev/robot/shaver')
    rospy.spin()
