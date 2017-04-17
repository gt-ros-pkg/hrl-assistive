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

# system library
import sys, time, random, threading
import numpy as np

# ROS library
import rospy, roslib
from roslib import message
import PyKDL

# HRL library
from hrl_srvs.srv import String_String, String_StringRequest
from pixel_2_3d.srv import Pixel23d, Pixel23dRequest

# vision library
import cv2
from cv_bridge import CvBridge, CvBridgeError
import image_geometry
from sensor_msgs.msg import Image, CameraInfo

from matplotlib import pyplot as plt


import tf
#from tf import TransformListener
from sensor_msgs.msg import PointCloud2, CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped, TransformStamped, PointStamped
from std_msgs.msg import Empty

try :
    import sensor_msgs.point_cloud2 as pc2
except:
    import point_cloud2 as pc2

class ArmReacherClient:
    def __init__(self, verbose=True, debug=False):
        rospy.init_node('visual_scooping')
        self.tf = tf.TransformListener()

        self.verbose = verbose
        self.debug   = debug
        ## self.started = False
        self.highestPointPublished = False
        self.bowlRawPos = None
        self.bowlCenter = None
        self.pinholeCamera = None
        self.cameraWidth = None
        self.cameraHeight = None
        self.image   = None
        self.img_frame = 'head_mount_kinect_rgb_optical_frame'

        self.image_lock = threading.RLock()

        self.initParams()
        self.initComms()

        self.initialized()


    def initParams(self):
        ''' Get parameters '''
        ## self.torso_frame = 'torso_lift_link'
        self.bridge = CvBridge()
    

    def initComms(self):

        # ROS publisher for data points
        if self.verbose: self.pointPublisher = rospy.Publisher('visualization_marker',
                                                               Marker, queue_size=100,
                                                               latch=True)
        self.highestBowlPointPublisher = rospy.Publisher('/hrl_manipulation_task/bowl_highest_point',
                                                         Point, queue_size=10, latch=True)

        # Connect to point cloud from Kinect
        self.camera_info_sub = rospy.Subscriber('/head_mount_kinect/qhd/camera_info', CameraInfo,
                                             self.cameraRGBInfoCallback)
        self.image_sub = rospy.Subscriber('/head_mount_kinect/qhd/image_color_rect',
                                          Image, self.imageCallback)        
        ## self.cloudSub = rospy.Subscriber('/head_mount_kinect/sd/points', PointCloud2,
        ##                                  self.cloudCallback)

        # Connect to arm reacher
        self.initSub = rospy.Subscriber('/hrl_manipulation_task/arm_reacher/init_bowl_height',
                                        Empty, self.initCallback)

        # Connect to bowl center location
        self.bowlSub = rospy.Subscriber('/hrl_manipulation_task/arm_reacher/bowl_cen_pose',
                                        PoseStamped, self.bowlCallback)

        # Service
        rospy.wait_for_service("/pixel_2_3d")
        self.pixel_2_3d  = rospy.ServiceProxy("/pixel_2_3d", Pixel23d)
        

    def initialized(self):
        rospy.sleep(5.0)
        rate = rospy.Rate(10) # 25Hz, nominally.
        while not rospy.is_shutdown():
            if self.pinholeCamera is not None: break
            rate.sleep()
        print "Initialized!!!"
        self.camera_info_sub.unregister()

    def bowlCallback(self, data):
        bowlPosePos = data.pose.position
        # Account for the fact  that the bowl center position is not directly in the center
        self.bowlRawPos = [bowlPosePos.x + 0.005, bowlPosePos.y + 0.01, bowlPosePos.z]
        ## if self.verbose: print 'Bowl position:', self.bowlRawPos

    def cameraRGBInfoCallback(self, data):
        if self.pinholeCamera is None:
            self.cameraWidth = data.width
            self.cameraHeight = data.height
            self.pinholeCamera = image_geometry.PinholeCameraModel()
            self.pinholeCamera.fromCameraInfo(data)

    def imageCallback(self, data):
        # Grab image from Kinect sensor
        try:
            image = self.bridge.imgmsg_to_cv2(data)
            image = np.asarray(image[:,:])
        except CvBridgeError, e:
            print e
            return

        with self.image_lock:
            # Convert to grayscale (if needed)
            ## self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, 1)
            self.image = image


    def initCallback(self, req):
        ## if not self.started:
        ##     # For some reason the initialize callback is trigger upon startup. This fixes that issue.
        ##     self.started = True
        ##     return
        # Call this right after 'lookAtBowl' and right before 'initScooping2'
        self.highestPointPublished = False
        self.bowlCenter = None
        print 'Initialize Callback'

        
    def publishHighestBowlPoint(self, highestBowlPoint):
        p = Point()
        p.x, p.y, p.z = highestBowlPoint
        self.highestBowlPointPublisher.publish(p)

    def publishPoints(self, name, points, size=0.02, r=1.0, g=0.0, b=0.0, a=1.0,
                      frame='head_mount_kinect_ir_optical_frame'):
        marker = Marker()
        marker.header.frame_id = frame
        marker.ns = name
        marker.type = marker.POINTS
        marker.action = marker.ADD
        marker.scale.x = size
        marker.scale.y = size
        marker.color.a = a
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        for point in points:
            p = Point()
            p.x, p.y, p.z = point
            marker.points.append(p)
        self.pointPublisher.publish(marker)

    def foodDetection(self, img):

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        #Conversion to gray scale
        n,m = np.shape(gray)
        ## gray = cv2.resize(gray, (m*3,n*3))

        # white bowl, colorful food
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)   #thresholding

        # noise removal
        kernel = np.ones((2,2),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
        ## opening = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 1)
        
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)
        # Finding sure foreground area
        ## dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
        dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,3)
        ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        ## cv2.imshow('image', np.hstack([sure_bg, sure_fg, thresh]))

        contours, hierarchy = cv2.findContours(sure_fg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        return contours, hierarchy

    def run(self):

        rate = rospy.Rate(10) # 25Hz, nominally.
        while not rospy.is_shutdown():

            # Wait to obtain cloud data until after arms have been initialized
            if self.highestPointPublished:
                rate.sleep()                
                continue

            # Transform the raw bowl center to the Kinect frame
            if self.bowlRawPos is not None:
                ## if self.verbose: print 'Using self.bowlRawPos'
                point = PointStamped()
                point.header.frame_id = 'torso_lift_link'
                point.point.x = self.bowlRawPos[0]
                point.point.y = self.bowlRawPos[1]
                point.point.z = self.bowlRawPos[2]
                # self.publishPoints('bowlCenterPre', [], g=1.0, frame='torso_lift_link')
                ## point = self.tf.transformPoint('head_mount_kinect_ir_optical_frame', point)
                point = self.tf.transformPoint(self.img_frame, point)
                self.bowlCenter = np.array([point.point.x, point.point.y, point.point.z])
            else:
                print 'No bowl center location has been published by the arm reacher server yet.'
                continue
            ## if self.verbose: self.publishPoints('bowlCenterPost', [self.bowlCenter], r=1.0)

            # Project bowl position to 2D pixel location to narrow search for bowl points
            # (use point projected to kinect frame)
            bowlProjX, bowlProjY = [int(x) for x in
                                    self.pinholeCamera.project3dToPixel(self.bowlCenter)]
            
            # Extract the bowl area
            x_offset = [30,50]
            y_offset = [40,30]
            with self.image_lock:
                if np.shape(self.image)[0] == 0: continue
                image = self.image[bowlProjY-y_offset[0]:bowlProjY+y_offset[1],
                                   bowlProjX-x_offset[0]:bowlProjX+x_offset[1]].copy()
                #debug
                ## image = self.image.copy()
            #debug
            ## cv2.rectangle(image,
            ##               (bowlProjX-x_offset[0],bowlProjY-y_offset[0]),   # upper left corner
            ##               (bowlProjX+x_offset[1],bowlProjY+y_offset[1]),   # lower right corner
            ##               (0,0,255),                  # red
            ##               2)                            # thickness              

            contours, hierarchy = self.foodDetection(image)
            # Project contours to the original image (only for debugging)
            if self.debug: image = self.image.copy()            


            # select the largest center?
            cen_list = []
            max_size = 0
            max_id   = 0
            points   = []
            for id in range(len(contours)):
                
                [intX, intY, intW, intH] = cv2.boundingRect(contours[id]+
                                                            np.array([[[bowlProjX-x_offset[0],
                                                                        bowlProjY-y_offset[0]]]])
                                                            )
                if intW*intH>max_size:
                    max_id   = id
                    max_size = intW*intH            
                    cen = [intX+intW/2, intY+intH/2 ]

                # debug
                if self.debug:
                    cv2.rectangle(image,
                                  (intX, intY),                 # upper left corner
                                  (intX+intW,intY+intH),        # lower right corner
                                  (0, 0, 255),                  # red
                                  2)                            # thickness              

                    
            ray = self.pinholeCamera.projectPixelTo3dRay(cen)
            req = Pixel23dRequest()
            req.pixel_u = cen[0]
            req.pixel_v = cen[1]                
            ret = self.pixel_2_3d(req)

            points.append([ret.pixel3d.pose.position.x,
                           ret.pixel3d.pose.position.y,
                           ret.pixel3d.pose.position.z])


            point = PointStamped()
            point.header.frame_id = self.img_frame
            point.point.x = points[0][0]
            point.point.y = points[0][1]
            point.point.z = points[0][2]

            point = self.tf.transformPoint('torso_lift_link', point)
            point = np.array([point.point.x, point.point.y, point.point.z])
            #point[0] += 0.02 

            print point
            self.publishHighestBowlPoint(point)
            self.highestPointPublished = True

            self.publishPoints('food', [point], frame='torso_lift_link')
            if self.debug:
                cv2.imshow('image', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            rate.sleep()

        if self.debug:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    client = ArmReacherClient(verbose=True, debug=False)
    client.run()
