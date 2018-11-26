#!/usr/bin/env python

#By Henry M. Clever


import rospy, roslib
import sys, os
import random, math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import cPickle as pkl
import tf


from time import sleep
from scipy import ndimage
from scipy.stats import mode
from hrl_msgs.msg import FloatArrayBare
import rosbag
import copy


from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

plt.ion()
fig = plt.figure(num=None, figsize=(6, 10))
plt.xlabel("Something on the X")
plt.ylabel("Something on the Y")
plt.show(block=False)




roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle



MAT = 'seat'

MAT_WIDTH = 0.74#0.762 #metres
MAT_HEIGHT = 1.75 #1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2 
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30 
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1) 
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1) 

class RealTimePose():
    '''Detects the head of a person sleeping on the autobed'''
    def __init__(self, database_path):
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.elapsed_time = []
        self.database_path = database_path
        #[self.p_world_mat, self.R_world_mat] = load_pickle('/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/mat_axes15.p')

        self.mat_pose_sampled = False
        self.ok_to_read_pose = False
        self.subject = 9
        self.bedangle = 0.
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)

        rospy.Subscriber("/fsascan", FloatArrayBare, self.pressure_map_callback)
        self.count = 0
        self.colorbar = True


    def pressure_map_callback(self, data):
        '''This callback accepts incoming pressure map from
        the Vista Medical Pressure Mat and sends it out.'''

        p_mat_raw  = data.data
        self.p_mat = np.array(p_mat_raw)
        self.ok_to_read_pose = True


    def mypause(self, interval):
        backend= plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return



    def rviz_publish_input(self, image, angle):
        mat_size = (84, 47)
        #mat_size = (41, 30)

        image = np.reshape(image, mat_size)

        #print np.fliplr(image[10:30, 32:36])
        image = np.clip(image*4, 0, 100)
        #print np.fliplr(image[10:30, 32:36])

        markerArray = MarkerArray()
        for j in range(10, image.shape[0]-10):
            for i in range(10, image.shape[1]-10):
                imagePublisher = rospy.Publisher("/pressure_image", MarkerArray)

                marker = Marker()
                marker.header.frame_id = "autobed/base_link"
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.a = 1.0
                if image[j,i] > 60:
                    marker.color.r = 1.
                    marker.color.b = (100 - image[j, i])*.9 / 60.
                else:
                    marker.color.r = image[j, i] / 40.
                    marker.color.b = 1.
                marker.color.g = (70-np.abs(image[j,i]-50))/100.

                marker.pose.orientation.w = 1.0

                marker.pose.position.x = i*0.0286
                if j > 33:
                    marker.pose.position.y = (84-j)*0.0286 - 0.0286*3*np.sin(np.deg2rad(angle))
                    marker.pose.position.z = -0.1
                    #print marker.pose.position.x, 'x'
                else:

                    marker.pose.position.y = (51) * 0.0286 + (33 - j) * 0.0286 * np.cos(np.deg2rad(angle)) - (0.0286*3*np.sin(np.deg2rad(angle)))*0.85
                    marker.pose.position.z = ((33-j)*0.0286*np.sin(np.deg2rad(angle)))*0.85 -0.1
                    #print j, marker.pose.position.z, marker.pose.position.y, 'head'

                # We add the new marker to the MarkerArray, removing the oldest
                # marker from it when necessary
                #if (self.count > 100):
                 #   markerArray.markers.pop(0)

                markerArray.markers.append(marker)

                #print self.count

                # Renumber the marker IDs
                id = 0
                for m in markerArray.markers:
                    m.id = id
                    id += 1
        imagePublisher.publish(markerArray)


    def preprocessing_create_pressure_angle_stack_realtime(self, p_map, bedangle, mat_size):
        '''This is for creating a 2-channel input using the height of the bed. '''
        p_map = np.reshape(p_map, mat_size)



        p_map_dataset = []

        #this is for the bed height matrix
        #height_strip = np.zeros(np.shape(p_map)[0])
        #height_strip[0:25] = np.flip(np.linspace(0, 1, num=25) * 25 * 2.86 * np.sin(np.deg2rad(bedangle)), axis=0)
        #height_strip = np.repeat(np.expand_dims(height_strip, axis=1), 27, 1)
        #a_map = height_strip
        a_map = np.zeros_like(p_map)


        # this makes a sobel edge on the image
        sx = ndimage.sobel(p_map, axis=0, mode='constant')
        sy = ndimage.sobel(p_map, axis=1, mode='constant')
        p_map_inter = np.hypot(sx, sy)

        #print np.shape(p_map_inter)
        p_map_dataset.append([p_map, p_map_inter, a_map])

        return p_map_dataset


    def run(self):

        '''This code just collects the first 1200 samples of the
                pressure mat that will come in through the bagfile and
                will label them with the label'''

        while not rospy.is_shutdown():

            if self.ok_to_read_pose == True:
                self.count += 1

                #print self.p_mat.shape

                input_stack = self.preprocessing_create_pressure_angle_stack_realtime(self.p_mat, self.bedangle, self.mat_size)

                viz_image = np.array(input_stack)[0,:,:,:]
                #viz_image = np.pad(viz_image[0,:,:], 10, 'constant', constant_values=(0.))
                
                viz_image = viz_image[0, :, :]

                #if True: #publish in rviz
                #    self.rviz_publish_input(viz_image * 1.3, self.bedangle)

                plt.gca().clear()
                plt.imshow(viz_image, interpolation='nearest', vmin = 0, vmax = 100, cmap=plt.cm.jet, origin='upper')
                if self.colorbar == True:
                    plt.colorbar()
                    self.colorbar = False
                self.mypause(0.1)


                print np.shape(viz_image)
                
                    

                self.ok_to_read_pose = False


if __name__ == '__main__':
    rospy.init_node('real_time_pose')

    #print file_details_dict['9']
    #database_path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials'
    database_path = '/home/henryclever/Autobed_OFFICIAL'

    getpose = RealTimePose(database_path)

    getpose.run()

