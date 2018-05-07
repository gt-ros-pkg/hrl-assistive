#!/usr/local/bin/python

import threading, subprocess 
import numpy as np
import rospy
from std_msgs.msg import Header, String

import hrl_lib.circular_buffer as cb

class start_detector:
    def __init__(self, window_size=25, duration=[2.0, 3.0]):
        '''
        initialize start detector. It sends start command if it finds stable face
        window_size: size of windows to check for "stable" start face
        duration: range of acceptable duration (exclusive) for the window
        '''
        self.duration=duration
        self.window_size = window_size

        self.feature_buffer = cb.CircularBuffer(window_size, (3,))
        self.f_lock = threading.RLock()

        self.features_sub = rospy.Subscriber("/gesture_control/features", String, self.cb, queue_size=10)
        self.start_pub    = rospy.Publisher("/gesture_control/start", String, queue_size=10)
        self.run()

    def cb(self, data):
        with self.f_lock:
            features = eval(data.data)
            self.feature_buffer.append(features)

    def run(self):
        rate =rospy.Rate(10)
        while not rospy.is_shutdown():
            with self.f_lock:
                feature_buffer = self.feature_buffer
            if self.stable(feature_buffer):
                self.start_pub.publish("Start")
                print "start"
                with self.f_lock:
                    self.feature_buffer.clear()
            rate.sleep()

    def stable(self, feature_buffer):
        if len(feature_buffer) == self.window_size:
            dur = abs(feature_buffer[0][0] - feature_buffer[-1][0])
            if dur > self.duration[0] and dur < self.duration[1]:
                feature_1 = np.asarray(feature_buffer[0]).copy()
                feature_1[1] = 0.0
                ret = True
                
                for feature in feature_buffer:
                    diff = np.asarray(feature) - feature_1
                    diff2 = diff < [100.0, 0.2, 4.0] 
                    diff3 = diff > [-100.0, -0.2, -4.0]
                    diff = []
                    for i in xrange(len(diff2)):
                        diff.append(diff2[i] and diff3[i])
                    for val in diff:
                        ret = ret and val
                return ret
        return False
def main():
    rospy.init_node('start_detector')
    detector = start_detector()

if __name__ == '__main__':
    main()
