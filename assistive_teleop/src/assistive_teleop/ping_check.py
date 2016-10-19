#!/usr/bin/env python

import rospy
from assistive_telep.msg import Ping


class CheckPingServer(object):
    def __init__(self, outTopic, inTopic):
        self.ping_pub = rospy.Publisher(outTopic, Ping, queue_size=0)
        self.ping_sub = rospy.Subscriber(inTopic, Ping, self.ping_cb)

    def ping_cb(self, return_msg):
        now = rospy.Time.now()
        sent_time = return_msg.sent_time
        client_time = return_msg.recv_time
        s_to_c = (client_time - sent_time).to_sec()
        c_to_s = (now - client_time).to_sec()
        rt = (now - sent_time).to_sec()
        rospy.loginfo("[%s] S->C: %f, C->S: %f, RT: %f", rospy.get_name(), s_to_c, c_to_s, rt)

    def send_ping(self):
        msg = Ping()
        msg.send_time = rospy.Time.now()
        self.ping_pub.publish(msg)


def main():
    rospy.init_node('web_ping_check')
    cps = CheckPingServer('/ping_relay_out', '/ping_relay_return')
    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        cps.send_ping()
        rate.sleep()
