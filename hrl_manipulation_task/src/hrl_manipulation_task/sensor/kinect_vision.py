#!/usr/bin/env python
#
# Copyright (c) 2014, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

# system
import rospy, roslib
import os, threading, copy

# util
import numpy as np
import PyKDL
from matplotlib import pyplot as plt

# vision library
import cv2
from cv_bridge import CvBridge, CvBridgeError
import image_geometry

# ROS message
import tf
from pr2_controllers_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import Image, CameraInfo

class kinect_vision(threading.Thread):
    def __init__(self, verbose=False):
        super(kinect_vision, self).__init__()        
        self.daemon = True
        self.cancelled = False
        self.isReset = False
        self.verbose = verbose
       
        self.enable_log = False
        self.init_time = 0.0
        
        # instant data
        
        # Declare containers
        self.time_data = []

        self.image_lock = threading.RLock()
        
        self.initParams()
        self.initComms()

        rate = rospy.Rate(10) # 25Hz, nominally.            
        while not rospy.is_shutdown():
            if self.imageGray is not None: break
            rate.sleep()

        if self.verbose: print "Kinect Vision>> initialization complete"
        
    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''
        if self.verbose: print "Kinect Vision>> Initialized pusblishers and subscribers"
        rospy.Subscriber('/head_mount_kinect/rgb_lowres/image', Image, self.imageCallback)        
        rospy.Subscriber('/head_mount_kinect/rgb_lowres/camera_info', CameraInfo, self.cameraRGBInfoCallback)

    def initParams(self):
        '''
        Get parameters
        '''
        self.torso_frame = 'torso_lift_link'
        self.bridge = CvBridge()

        self.cameraWidth = None
        self.imageGray = None

    def imageCallback(self, data):
        # Grab image from Kinect sensor
        try:
            image = self.bridge.imgmsg_to_cv(data)
            image = np.asarray(image[:,:])
        except CvBridgeError, e:
            print e
            return

        with self.image_lock:
            # Convert to grayscale (if needed)
            self.imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, 1)

    def cameraRGBInfoCallback(self, data):
        if self.cameraWidth is None:
            self.cameraWidth = data.width
            self.cameraHeight = data.height
            self.pinholeCamera = image_geometry.PinholeCameraModel()
            self.pinholeCamera.fromCameraInfo(data)
            self.rgbCameraFrame = data.header.frame_id
            
    def transpose3DToCamera(self, frame3D, pos3D):
        # Transpose 3D position to camera frame
        self.transformer.waitForTransform(self.rgbCameraFrame, frame3D, rospy.Time(0), rospy.Duration(5))
        try :
            transPos, transRot = self.transformer.lookupTransform(self.rgbCameraFrame, frame3D, rospy.Time(0))
            transMatrix = np.dot(tf.transformations.translation_matrix(transPos), tf.transformations.quaternion_matrix(transRot))
        except tf.ExtrapolationException:
            print 'Transpose failed!'
            return

        transposedPos = np.dot(transMatrix, np.array([pos3D[0], pos3D[1], pos3D[2], 1.0]))[:3]
        x2D, y2D = self.pinholeCamera.project3dToPixel(transposedPos)
        return x2D, y2D
    
    
    def test(self, save_pdf=False):
        ## img = cv2.imread('/home/dpark/Dropbox/HRL/IMG_3499.JPG',0)
        sift = cv2.SIFT()
        fig = plt.figure()
        plt.ion()
        plt.show()        
        
        rate = rospy.Rate(10) # 25Hz, nominally.    
        while not rospy.is_shutdown():
            print "running test"
            with self.image_lock:
                imageGray = copy.copy(self.imageGray)

            kp = sift.detect(imageGray,None)
            img=cv2.drawKeypoints(imageGray,kp)
            ## cv2.imwrite('test.jpg',img)
            ## ## os.system('cp test.jpg ~/Dropbox/HRL/')
            plt.imshow(img)
            plt.draw()
            ## cv2.imshow('MyWindow', img)
            rate.sleep()

        ## # Initiate STAR detector
        ## orb = cv2.ORB_create()

        ## # find the keypoints with ORB
        ## kp = orb.detect(img,None)

        ## # compute the descriptors with ORB
        ## kp, des = orb.compute(img, kp)

        ## # draw only keypoints location,not size and orientation
        ## img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
        ## plt.imshow(img2)

        ## if save_pdf == True:
        ##     fig.savefig('test.pdf')
        ##     fig.savefig('test.png')
        ##     os.system('cp test.p* ~/Dropbox/HRL/')
        ## else:
        ##     if show_plot: plt.show()        



        
        
        
    def reset(self, init_time):
        self.init_time = init_time

        # Reset containers
        self.time_data = []
        
        self.isReset = True

        
    ## def isReady(self):
    ##     if self.azimuth is not None and self.power is not None and \
    ##       self.head_joints is not None:
    ##       return True
    ##     else:
    ##       return False




if __name__ == '__main__':
    rospy.init_node('kinect_vision')

    kv = kinect_vision()
    kv.test(True)


        
