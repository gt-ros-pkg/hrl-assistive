#!/usr/bin/env python

import os
import roslib; roslib.load_manifest('sound_play')
import rospy
from std_msgs.msg import String

class Speaker:
    def __init__(self):
        rospy.init_node('soundplay_node')
        rospy.Subscriber('/text_to_say', String, self.say)

    def say(self,msg):
        print "Speaking: %s" %msg.data
        os.system('echo '+str(msg.data)+'| festival --tts')

if __name__ == '__main__':
    Speaker = Speaker()
    while not rospy.is_shutdown():
        rospy.spin()
