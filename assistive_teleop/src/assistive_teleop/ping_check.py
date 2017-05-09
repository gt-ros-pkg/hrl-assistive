#!/usr/bin/env python

import rospy
from std_msgs.msg import Duration, Time


class CheckPingServer(object):
    def __init__(self, outTopic, inTopic, rtTopic):
        self.ping_pub = rospy.Publisher(outTopic, Time, queue_size=0)
        self.ping_sub = rospy.Subscriber(inTopic, Time, self.ping_cb)
        self.roundtrip_pub = rospy.Publisher(rtTopic, Duration, queue_size=0)
        rospy.loginfo("[%s] Ping Check Ready.", rospy.get_name())

    def ping_cb(self, msg):
        print msg
        roundtrip_time = rospy.Time.now() - msg.data
        rospy.loginfo("[%s] RoundTrip Time: %f", rospy.get_name(), roundtrip_time.to_sec())
        self.roundtrip_pub.publish(roundtrip_time)

    def send_ping(self):
        self.ping_pub.publish(rospy.Time.now())


def main():
    rospy.init_node('web_ping_check')
    cps = CheckPingServer('/ping_relay_out', '/ping_relay_return', '/ping_relay_roundtrip')
    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        cps.send_ping()
        rate.sleep()
