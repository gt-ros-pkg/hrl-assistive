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
import os, threading, copy, random

# util
import numpy as np
import PyKDL
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# vision library
import cv2
from cv_bridge import CvBridge, CvBridgeError
import image_geometry

# ROS message
## import tf
## from pr2_controllers_msgs.msg import JointTrajectoryControllerState


class kinect_vision_image(threading.Thread):
    def __init__(self, verbose=False):
        super(kinect_vision, self).__init__()        
        self.daemon = True
        self.cancelled = False
        self.isReset = False
        self.verbose = verbose
       
        self.enable_log = False
        self.init_time = 0.0

        # instant data
        self.time    = None
        self.images  = None
        
        # Declare containers
        self.time_data = []

        self.lock = threading.RLock()
        
        self.initParams()
        self.initComms()
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
        self.image   = None
        ## self.currImg = None
        ## self.prevImg = None

        self.init_center = None
        self.last_center = None
        self.last_label  = None
        self.track_len = 10
        self.tracks_cen   = []
        self.tracks_label = []
        self.cluster_centers = None
        self.n_clusters = 5
        
        self.color_list = [[0,0,255],
                           [255,0,0],
                           [0,255,0],
                           [255,255,255]]
        for i in xrange(10):
            self.color_list.append([random.randint(0,255),
                                    random.randint(0,255),
                                    random.randint(0,255) ])

        
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
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, 1)

    def cameraRGBInfoCallback(self, data):
        if self.cameraWidth is None:
            self.cameraWidth = data.width
            self.cameraHeight = data.height
            self.pinholeCamera = image_geometry.PinholeCameraModel()
            self.pinholeCamera.fromCameraInfo(data)
            self.rgbCameraFrame = data.header.frame_id
            
            
    def test(self, save_pdf=False):

        fig = plt.figure()
        plt.ion()
        plt.show()        

        prevImg = None
        currImg = None
        counter = 0
        
        rate = rospy.Rate(10) # 25Hz, nominally.    
        while not rospy.is_shutdown():
            print "running test"
            with self.image_lock:
                if prevImg is None: 
                    currImg = copy.copy(self.image)
                    prevImg = copy.copy(currImg)
                    continue
                else:
                    prevImg = copy.copy(currImg)
                    currImg = copy.copy(self.image)
                    

            flow = cv2.calcOpticalFlowFarneback(prevImg, currImg, 0.5, 3, 15, 3, 5, 1.2, 0)
            cluster_img = self.draw_clustered_flow(currImg, flow, counter)

                    
            ## kp = sift.detect(imageGray,None)
            ## img=cv2.drawKeypoints(imageGray,kp)
            ## cv2.imwrite('test.jpg',img)
            ## ## os.system('cp test.jpg ~/Dropbox/HRL/')
            plt.imshow(img)
            plt.draw()
            ## cv2.imshow('MyWindow', img)

            counter += 1
            rate.sleep()
            


    def draw_clustered_flow(self, img, flow, counter):

        h, w = flow.shape[:2]

        ## start = time.clock()
        yy, xx = np.meshgrid(range(w), range(h))
        flow_array = flow.reshape((h*w,2))
        mag_array  = np.linalg.norm(flow_array, axis=1)

        data = np.vstack([xx.ravel(), yy.ravel(), mag_array]).T
        flow_filt = data[data[:,2]>3.0]
        ## end = time.clock()
        ## print "%.2gs" % (end-start)

        if len(flow_filt) < self.n_clusters: return img

        if self.last_center is not None:
            clt = KMeans(n_clusters = self.n_clusters, init=self.init_center)
        else:
            clt = KMeans(n_clusters = self.n_clusters)
        clt.fit(flow_filt)
        self.init_center = clt.cluster_centers_

        #----------------------------------------------------------
        # Spatio-temporal clustering
        #----------------------------------------------------------
        time_array = np.ones((self.n_clusters, 1))*counter
        if self.cluster_centers is None:
            self.cluster_centers = clt.cluster_centers_
            ## cluster_centers = np.hstack([time_array, clt.cluster_centers_])
        else:
            self.cluster_centers = np.vstack([ self.cluster_centers, clt.cluster_centers_ ])
            ## cluster_centers = np.vstack([ cluster_centers, np.hstack([time_array, clt.cluster_centers_]) ])

        if len(self.cluster_centers) > self.n_clusters*20:
            self.cluster_centers = self.cluster_centers[-self.n_clusters*20:]

        clt2 = KMeans(n_clusters = self.n_clusters)
        clt2.fit(self.cluster_centers)

        if self.last_label is None: 
            self.last_center = clt.cluster_centers_[-self.n_clusters:].tolist()
            self.last_label  = clt.labels_[-self.n_clusters:].tolist()
            for ii in xrange(len(self.last_label)):
                if self.last_label[ii] in self.tracks_label:
                    idx = self.tracks_label.index(self.last_label[ii])
                    self.tracks_cen[idx][-1] = [ (self.tracks_cen[idx][-1][0] + self.last_center[ii][1])/2.0,\
                                                 (self.tracks_cen[idx][-1][1] + self.last_center[ii][0])/2.0 ]
                else:
                    self.tracks_cen.append([ [self.last_center[ii][1],self.last_center[ii][0]] ])
                    self.tracks_label.append(self.last_label[ii])
            return img
        cur_centers = clt.cluster_centers_[-self.n_clusters:]
        cur_labels  = clt.labels_[-self.n_clusters:]

        # label matching
        max_label = max(self.last_label)
        new_tracks_cen   = []
        new_tracks_label = []
        for ii, (center, label) in enumerate(zip(cur_centers, cur_labels)):
            min_dist = 1000
            min_label= 0
            min_idx  = 0
            for jj, (c, l) in enumerate(zip(self.last_center, self.last_label)):
                dist = np.linalg.norm(center-c)
                if dist < min_dist:
                    min_dist = dist
                    min_label= l
                    min_idx  = jj

            # new label
            if min_dist > 50:
                cur_labels[ii] = max_label+1
                max_label += 1
                new_tracks_cen.append([[center[1],center[0]]])
                new_tracks_label.append(max_label)
                ## print tracks_cen            
                ## for jj in xrange(len(tracks_label)):
                ##     if tracks_label[jj] == min_label:
                ##         del tracks_label[jj]
                ##         break
            else:
                del self.last_center[min_idx]
                del self.last_label[min_idx]
                cur_labels[ii] = min_label

                idx = tracks_label.index(min_label)
                self.tracks_cen[idx].append([center[1],center[0]])

                if len( self.tracks_cen[idx] ) > self.track_len:
                    del self.tracks_cen[idx][0]

                new_tracks_cen.append( self.tracks_cen[idx] )
                new_tracks_label.append( self.tracks_label[idx] )

        self.tracks_label = new_tracks_label
        self.tracks_cen   = new_tracks_cen

        ######################### Update last centers and labels             
        self.last_center = cur_centers.tolist()
        self.last_label  = cur_labels.tolist()

        # cluster center
        overlay = img.copy()
        for ii, (center, label) in enumerate(zip(cur_centers, cur_labels)):
            x = int(center[1])
            y = int(center[0])
            color_idx = label if label < len(self.color_list) else label%len(self.color_list)
            c = self.color_list[color_idx]
            cv2.circle(overlay, (x, y), 8, (c[0], c[1], c[2]), -1)
            ## cv2.circle(overlay, (x, y), 8, (c[0], c[1], int(c[2]*center[2])), -1)

        opacity = 0.5
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

        ## # moving direction
        ## vis = img.copy()
        ## for ii, center in enumerate(cur_centers):
        ##     x = int(center[1])
        ##     y = int(center[0])
        ##     flow = np.sum(flow_array[data[:,2]>3.0][clt.labels_ == ii], axis=0)
        ##     lines = np.vstack([x, y, x+flow[0], y+flow[1]]).T.reshape(-1, 2, 2)
        ##     lines = np.int32(lines + 0.5)        
        ##     cv2.polylines(vis, lines, 0, (0, 255, 0))

        ## opacity = 1.0
        ## cv2.addWeighted(vis, opacity, img, 1 - opacity, 0, img)

        ## tracking
        vis = img.copy()

        new_tracks = []
        for tr, label in zip(self.tracks_cen, self.tracks_label):
            if len(tr)<3: continue
            new_tracks.append( np.int32(tr) )
        ##     print np.int32(tr), np.shape(np.int32(tr))
        ##     cv2.polylines(vis, np.int32(np.array(tr)), False, (0, 255, 0))
        ## print np.int32(tracks_cen), np.shape( np.int32(tracks_cen) )
        cv2.polylines(vis, new_tracks, False, (0, 255, 0))        
        opacity = 1.0
        cv2.addWeighted(vis, opacity, img, 1 - opacity, 0, img)




        return img



            
        
    def reset(self, init_time):
        self.init_time = init_time
        self.isReset = True

        # Reset containers
        self.time_data = []

        
    def isReady(self):
        if self.images is not None:
          return True
        else:
          return False




if __name__ == '__main__':
    rospy.init_node('kinect_vision_image')

    kv = kinect_vision_image()
    kv.test(True)


        
