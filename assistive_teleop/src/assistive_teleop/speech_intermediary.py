#!/usr/bin/env python

import roslib; roslib.load_manifest('sound_play')
import rospy
from std_msgs.msg import String
from sound_play.libsoundplay import SoundClient

class SoundIntermediary():
    def __init__(self):
        self.sound_client = SoundClient()
        self.voice = rospy.get_param("~voice", "kal_diphone")
        rospy.Subscriber("wt_speech", String, self.speak)

    def speak(self, msg):
        self.sound_client.say(msg.data, self.voice)

if __name__=='__main__':
    rospy.init_node('wt_sound_intermediary')
    SI = SoundIntermediary()
    while not rospy.is_shutdown():
        rospy.spin()
