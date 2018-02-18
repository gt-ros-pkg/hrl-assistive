#!/usr/bin/env python

#By Henry M. Clever
#This code records data from the thermal camera and from the kinect to create a multimodal bagfile of pose data


import sys
import operator
import numpy as np

import roslib; roslib.load_manifest('hrl_pose_estimation')
import rospy
import cPickle as pkl
from hrl_msgs.msg import FloatArrayBare
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
import threading
import rosbag
from cv_bridge import CvBridge, CvBridgeError



NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30 
TOTAL_TAXELS = NUMOFTAXELS_X*NUMOFTAXELS_Y

class PoseRecord():
    '''Subscribes to topics that are potentially useful for estimating pose'''
    def __init__(self, filename):
        #the filename is where it dumps the recorded rosbag

        self.filename = filename
        self.callback_lock = threading.Lock()
        self.bag = rosbag.Bag('test.bag','w')
        self.bridge = CvBridge()

        rospy.init_node('pose_record', anonymous=False)
        rospy.Subscriber("/kinect2/hd/image_depth_rect", Image, 
                self.depth_hd_callback)
        rospy.Subscriber("/kinect2/qhd/image_depth_rect", Image, 
                self.depth_qhd_callback)        
        rospy.Subscriber("/kinect2/sd/image_depth_rect", Image, 
                self.depth_sd_callback)
        rospy.Subscriber("thermal_camera/thermal_camera_driver/image_mono", Image, self.therm_mono_callback)
       # rospy.Subscriber("thermal_camera/thermal_camera_driver/image_color", Image, self.therm_color_callback)
        rospy.Subscriber("thermal_camera/thermal_camera_driver/image_rect", Image, self.therm_rect_callback)
        
        



    def depth_hd_callback(self, data):
        '''This callback will sample data until its asked to stop'''
        self.callback_lock.acquire()
        #print 'depth hd: ', data.height, data.width
        self.bag.write("/kinect2/hd/image_depth_rect",data)
    #    data.encoding = "mono16"
    #    try:
    #        cv_depth_hd = self.bridge.imgmsg_to_cv2(data, "mono16")
            #print 'length', len(cv_depth_hd), len(cv_depth_hd[0])
            #for row in range(500, 511):
               # print cv_depth_hd[row][500:510]
    #    except CvBridgeError as e:
    #        print e, 'error'
        self.callback_lock.release()


    def depth_qhd_callback(self, data):
        '''This callback will sample data until its asked to stop'''
        self.callback_lock.acquire()
        #print 'depth qhd: ', data.height, data.width
        self.bag.write("/kinect2/qhd/image_depth_rect",data)
        self.callback_lock.release()

    def depth_sd_callback(self, data):
        '''This callback will sample data until its asked to stop'''
        self.callback_lock.acquire()
        #print 'depth sd: ', data.height,  data.width
        self.bag.write("/kinect2/sd/image_depth_rect",data)

       # data.encoding = "mono16"
       # try:
       #     cv_depth_hd = self.bridge.imgmsg_to_cv2(data, "mono16")
       #     print 'length', len(cv_depth_hd), len(cv_depth_hd[0])
       #     for row in range(300, 311):
       #         print cv_depth_hd[row][300:310]
       # except CvBridgeError as e:
       #     print e, 'error'
        self.callback_lock.release()

    def therm_mono_callback(self, data):
        '''This callback will sample data until its asked to stop'''
        self.callback_lock.acquire()
        #print 'therm mono: ', data.height, data.width
        self.bag.write("thermal_camera/thermal_camera_driver/image_mono",data)
        self.callback_lock.release()

    #def therm_color_callback(self, data):
    #    '''This callback will sample data until its asked to stop'''
    #    self.callback_lock.acquire()
    #    #print 'therm color: ', data.height,  data.width
    #    self.bag.write("thermal_camera/thermal_camera_driver/image_color",data)
    #    self.callback_lock.release()

    def therm_rect_callback(self, data):
        '''This callback will sample data until its asked to stop'''
        self.callback_lock.acquire()
        #print 'therm rect: ', data.height,  data.width
        self.bag.write("thermal_camera/thermal_camera_driver/image_rect",data)
        self.callback_lock.release()




    def run(self):
        '''This code keeps callbacks running'''
        count = 0
        while not rospy.is_shutdown():
            count = count + 1
        self.bag.close()

        #pkl.dump(self.training_database, open(self.filename, "wb"))
                 

if __name__ == "__main__":
    #Argument 1: Filename of the pkl file to be stored.
    filename = '/home/henryclever/bagfiles'
    convertor = PoseRecord(filename)
    convertor.run()
