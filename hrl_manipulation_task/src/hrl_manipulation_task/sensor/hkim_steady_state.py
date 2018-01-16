import rospy
import threading
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from steady_state_linear_reg import SteadyStateDetector
from cv_bridge import CvBridge, CvBridgeError

class depthFaceSteady:
    def __init__(self, depth_image):
        self.min_dist = 30
        self.max_dist = 90
        self.bridge = CvBridge()
        
        self.depth_img = None
        self.depth_lock = threading.RLock()
        self.steady_detector = SteadyStateDetector(15, (3,), 1, mode='std monitor', overlap =-1)

        self.depth_sub = rospy.Subscriber(depth_image, Image, self.depth_callback)

        self.steady_pub = rospy.Publisher("/manipulation_task/steady_face", String, queue_size=10)

        self.run()

    def depth_callback(self, data):
        with self.depth_lock:
            self.depth_img = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            with self.depth_lock:
                depth = self.depth_img
            if depth is not None:
                #only take values that are min_dist ~ max_dist (cm) 
                filtered = (depth*100).astype('uint8')
                filtered2 = (depth*100).astype('uint8')
                filtered3 = (depth*100).astype('uint8')
                filtered[filtered < self.max_dist] = 255
                filtered2[filtered2 > self.min_dist] = 255
                filtered = filtered & filtered2
                #filtered = cv2.blur(filtered, (10, 10))
                #filtered[filtered != 0] = 255
                depth_float = filtered3 & filtered
                depth_float = depth_float.astype('float')
                #take avg of distances
                total = np.sum(depth_float)
                avg = total / float(np.count_nonzero(depth_float))
                x_arr, y_arr = np.sum(depth_float, axis=0), np.sum(depth_float, axis=1)
                x_mid = self.find_mid(x_arr, total)
                y_mid = self.find_mid(y_arr, total)
                print "raw states ", avg, x_mid, y_mid
                self.steady_detector.append([avg, x_mid, y_mid], 0)
                if self.steady_detector.stable([2., 2., 2.]):
                    self.steady_pub.publish("STEADY")
                else:
                    self.steady_pub.publish("NOT STEADY")
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
