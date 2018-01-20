#!/usr/bin/env python
import time
import rospy
import threading
import numpy as np
import hrl_lib.circular_buffer as cb
import cv2
import dlib
from sensor_msgs.msg import Image
from std_msgs.msg import String
from steady_state_linear_reg import SteadyStateDetector
from cv_bridge import CvBridge, CvBridgeError

class depthFaceSteady:
    def __init__(self, depth_image):
        self.min_dist = 50
        self.max_dist = 110
        self.bridge = CvBridge()
        self.cb = cb.CircularBuffer(1, (3,))
        self.win = dlib.image_window()
        self.time1 = None
        
        self.depth_img = None
        self.depth_lock = threading.RLock()
        self.steady_detector = SteadyStateDetector(20, (3,), 4, mode='std monitor', overlap =-1)

        self.depth_sub = rospy.Subscriber(depth_image, Image, self.depth_callback, queue_size=1)

        self.steady_pub = rospy.Publisher("/manipulation_task/steady_face", String, queue_size=10)

        self.run()

    def depth_callback(self, data):
        with self.depth_lock:
            self.depth_img = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.time1 is None:
                self.time1= time.time()
            with self.depth_lock:
                depth = self.depth_img
                self.depth_img = None
            if depth is not None:
                print "prep time", time.time() -self.time1
                self.time1 = None
                rate.sleep()
                continue
                depth = cv2.resize(depth, (depth.shape[0]/2, depth.shape[1]/2))
                #only take values that are min_dist ~ max_dist (cm)
                #print np.asarray(depth[100][100:200] * 100).astype('uint8')
                filtered = depth.astype('uint8')#(depth*100).astype('uint8')
                filtered2 = filtered#(depth*100).astype('uint8')
                filtered3 = filtered#(depth*100).astype('uint8')
                """
                filtered = (depth*100).astype('uint8')
                filtered2 = (depth*100).astype('uint8')
                filtered3 = (depth*100).astype('uint8')
                filtered[filtered < self.max_dist] = 255
                filtered[filtered != 255] = 0
                filtered2[filtered2 > self.min_dist] = 255
                filtered2[filtered2 != 255] = 0
                filtered = filtered & filtered2
                """
                print 'filtertime', time.time()-self.time1
                #print filtered[100][100:200]
                #print filtered[100][100:200]
                #cv2.imshow(np.asarray(filtered))
                #filtered = cv2.blur(filtered, (10, 10))
                #filtered[filtered != 0] = 255
                depth_float = filtered3 & filtered
                #self.win.set_image(depth_float)
                depth_float = depth_float.astype('float')
                #take avg of distances
                total = np.sum(depth_float)
                #print "nonzeros ", np.count_nonzero(depth_float)
                avg = total / float(np.count_nonzero(depth_float))
                x_arr, y_arr = np.sum(depth_float, axis=0), np.sum(depth_float, axis=1)
                x_mid = self.find_mid(x_arr, total)
                y_mid = self.find_mid(y_arr, total)
                self.cb.append([avg, x_mid, y_mid])
                if len(self.cb) == self.cb.size:
                    self.steady_detector.append(np.mean(self.cb, axis=0), 0)
                if self.steady_detector.stable([1.5, 1.5, 1.5]):
                    #print "steady"
                    self.steady_pub.publish("STEADY")
                else:
                    #print "not steady"
                    self.steady_pub.publish("NOT STEADY")
                print time.time() - self.time1
                self.time1 = None
            else:
                self.steady_pub.publish("NOT STEADY")
            rate.sleep()

    def find_mid(self, arr, total):
        cnt = 0.0
        for i, val in enumerate(arr):
            cnt = cnt + val
            if cnt >= total/2.0:
                return i
        return len(arr)

if __name__ == "__main__":
    rospy.init_node("depth_steady")
    depthFaceSteady("/SR300/depth_registered/sw_registered/image_rect")
