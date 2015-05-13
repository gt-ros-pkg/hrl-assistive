#!/usr/bin/env python

from collections import deque
from math import sqrt

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
from pr2_msgs.msg import AccelerometerState
from std_msgs.msg import Bool, Float64


class Contact(object):
    def __init__(self, queue_len=300, dt=0.00033, RC=0.0033, threshold=2.5):
        self.raw_queue = deque([0]*queue_len, queue_len)
        self.filt_queue = deque([0]*queue_len, queue_len)
        self.dt = dt
        self.RC = RC
        self.threshold = threshold
        self.alpha = self.RC / (self.RC + self.dt)

    def add_datum(self, datum):
        new_filt = self.update_filter(self.filt_queue[-1], self.raw_queue[-1], datum)
        self.raw_queue.append(datum)
        self.filt_queue.append(new_filt)

    def update_filter(self, last_filt, last_raw, new_raw):
        return self.alpha * last_filt + self.alpha * (new_raw - last_raw)


class ContactNode(Contact):
    def __init__(self, *args, **kwargs):
        super(ContactNode, self).__init__(*args, **kwargs)
        self.acc_sub = rospy.Subscriber('/accelerometer/r_gripper_motor', AccelerometerState, self.acc_cb)
        self.contact_pub = rospy.Publisher('/accelerometer/r_gripper_motor/contact', Bool)
        self.contact_mag_pub = rospy.Publisher('/acm', Float64)
        self.frame = ''
        self.latest_data_time = rospy.Time(0)

        self.count = 0

    def acc_cb(self, msg):
        self.frame = msg.header.frame_id
        self.latest_data_time = msg.header.stamp
        for i,vecs in enumerate(msg.samples):
            self.add_datum(sqrt( vecs.x**2 + vecs.y**2 + vecs.z**2 ))
            self.contact_mag_pub.publish(self.filt_queue[-1])
            if self.filt_queue[-1] > self.threshold:
                self.count +=1
                print "Contact! %s" %(self.count)
                self.contact_pub.publish(True)


if __name__=='__main__':
    rospy.init_node('r_accelerometer_contact')
    contact_node = ContactNode();
    rospy.spin()
