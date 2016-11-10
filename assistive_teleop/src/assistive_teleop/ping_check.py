#!/usr/bin/env python

import rospy
from std_msgs.msg import Duration
from assistive_teleop.msg import Ping


class CheckPingServer(object):
    def __init__(self, outTopic, inTopic, rtTopic):
        self.ping_pub = rospy.Publisher(outTopic, Ping, queue_size=0)
        self.ping_sub = rospy.Subscriber(inTopic, Ping, self.ping_cb)
        self.roundtrip_pub = rospy.Publisher(rtTopic, Duration, queue_size=0)

    def ping_cb(self, return_msg):
        print return_msg
        now = rospy.Time.now()
        sent_time = return_msg.send_time
        client_time = return_msg.recv_time
        s_to_c = (client_time - sent_time).to_sec()
        c_to_s = (now - client_time).to_sec()
        roundtrip_time = now - sent_time
        rt = roundtrip_time.to_sec()
        rospy.loginfo("[%s] S->C: %f, C->S: %f, RT: %f", rospy.get_name(), s_to_c, c_to_s, rt)
        self.roundtrip_pub.publish(roundtrip_time)

    def send_ping(self):
        msg = Ping()
        msg.send_time = rospy.Time.now()
        self.ping_pub.publish(msg)


def main():
    rospy.init_node('web_ping_check')
    cps = CheckPingServer('/ping_relay_out', '/ping_relay_return', '/ping_relay_roundtrip')
    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        cps.send_ping()
        rate.sleep()
