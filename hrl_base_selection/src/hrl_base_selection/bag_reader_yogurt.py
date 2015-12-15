#!/usr/bin/env python

import numpy as np
import math as m
import openravepy as op
import copy

import time
import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
import rospy, rospkg
import tf
from geometry_msgs.msg import PoseStamped
from rosgraph_msgs.msg import Clock
from tf.msg import tfMessage

from sensor_msgs.msg import JointState
from std_msgs.msg import String
import hrl_lib.transforms as tr
from hrl_base_selection.srv import BaseMove
from visualization_msgs.msg import Marker
from helper_functions import createBMatrix
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import logging
import genpy
import tf.transformations as tft

import sys
import numpy as np
import scipy.io

import roslib
#roslib.load_manifest('hrl_phri_2011')
import rospy

import rosbag


def main():
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('hrl_base_selection')
    #bag_path = '/home/ari/svn/robot1_data/2011_ijrr_phri/log_data/bag/'
    bag_path = '/home/ari/git/gt-ros-pkg.hrl-assistive/hrl_base_selection/data/yogurt/bags/'

    bag_name_list = ['2014-06-17-16-13-46',
                     #'2014-06-17-16-17-20', This is a bad bag
                     '2014-06-17-16-19-02',
                     '2014-06-17-16-21-57',
                     '2014-06-17-16-24-03',
                     '2014-06-17-16-26-12',
                     '2014-06-17-16-27-46',
                     '2014-06-17-16-29-18',
                     '2014-06-17-16-31-24',
                     '2014-06-17-16-33-13',
                     '2014-06-18-12-37-23',
                     '2014-06-18-12-39-43',
                     '2014-06-18-12-41-42',
                     '2014-06-18-12-43-45',
                     '2014-06-18-12-45-26',
                     '2014-06-18-12-47-22',
                     '2014-06-18-12-49-04',
                     '2014-06-18-12-50-52',
                     '2014-06-18-12-52-39',
                     '2014-06-18-12-54-26']

#    bag_name = '2014-06-17-16-17-20'
#    bag_name = '2014-06-17-16-19-02'
#    bag_name = 
#    bag_name = 
#    bag_name = 
#    bag_name = 
#    bag_name = 
#    bag_name = 
#    bag_name = 
#    bag_name = 
#    bag_name = 
    for num,bag_name in enumerate(bag_name_list):
        print 'Going to open bag: ',bag_name,'.bag'
        print 'Starting on a tool log file! \n'
        bag = rosbag.Bag(''.join([bag_path,bag_name,'.bag']), 'r')
        tool = open(''.join([pkg_path,'/data/yogurt/logs/',bag_name,'_tool.log']), 'w')
        for topic, msg, t in bag.read_messages():
            if topic == "/spoon/pose":
                tx = copy.copy(msg.transform.translation.x)
                ty = copy.copy(msg.transform.translation.y)
                tz = copy.copy(msg.transform.translation.z)
                rx = copy.copy(msg.transform.rotation.x)
                ry = copy.copy(msg.transform.rotation.y)
                rz = copy.copy(msg.transform.rotation.z)
                rw = copy.copy(msg.transform.rotation.w)
                world_B_spoon = createBMatrix([tx,ty,tz],[rx,ry,rz,rw])#*adl2_B_pr2goal
                eul = tft.euler_from_matrix(world_B_spoon,'rxyz')
                #print world_B_shaver,'\n'
                #print eul,'\n'
                #print 'time: \n',t
                tool.write(''.join([str(t),' %f %f %f %f %f %f \n' % (world_B_spoon[0,3],world_B_spoon[1,3],world_B_spoon[2,3],eul[0],eul[1],eul[2])]))
                #tool.write(''.join([world_B_shaver[0,3],' ',world_B_shaver[1,3],' ',world_B_shaver[2,3],' ',eul[0],' ',eul[1],' ',eul[2],'\n']))
        bag.close()
        tool.close()
        print 'Starting on a head log file! \n'
        bag = rosbag.Bag(''.join([bag_path,bag_name,'.bag']), 'r')
        head = open(''.join([pkg_path,'/data/yogurt/logs/',bag_name,'_head.log']), 'w')
        for topic, msg, t in bag.read_messages():
            if topic == "/head/pose":
                tx = copy.copy(msg.transform.translation.x)
                ty = copy.copy(msg.transform.translation.y)
                tz = copy.copy(msg.transform.translation.z)
                rx = copy.copy(msg.transform.rotation.x)
                ry = copy.copy(msg.transform.rotation.y)
                rz = copy.copy(msg.transform.rotation.z)
                rw = copy.copy(msg.transform.rotation.w)
                world_B_head = createBMatrix([tx,ty,tz],[rx,ry,rz,rw])#*adl2_B_pr2goal
                eul = copy.copy(tft.euler_from_matrix(world_B_head,'rxyz'))
                head.write(''.join([str(t),' %f %f %f %f %f %f \n' % (world_B_head[0,3],world_B_head[1,3],world_B_head[2,3],eul[0],eul[1],eul[2])]))
                #head.write(''.join([t,' ',world_B_head[0,3],' ',world_B_head[1,3],' ',world_B_head[2,3],' ',eul[0],' ',eul[1],' ',eul[2],' ',eul[3],'\n']))
        bag.close()
        head.close()

        print "Saved file! Finished bag #",num+1
    print 'Done with all bag files!'









if __name__ == "__main__":
    main()



    '''def __init__(self, transform_listener=None):
        self.listener = tf.TransformListener()
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('hrl_base_selection')
        logging.basicConfig(filename=''.join([pkg_path,'/data/sub1_shaver_head.log']), level=logging.INFO)
        logging.basicConfig(filename=''.join([pkg_path,'/data/sub1_shaver_tool.log']), level=logging.INFO)
        self.head = []
        self.tool = []
        self.clock = None
        self.clock_sub = rospy.Subscriber('/clock',Clock,self.clock_cb)
        #self.tf_sub = rospy.Subscriber('/tf', tfMessage, self.tf_cb)
        self.h = open(''.join([pkg_path,'/data/sub1_shaver_head.log']), 'w')
        self.t = open(''.join([pkg_path,'/data/sub1_shaver_tool.log']), 'w')
        self.start_time = time.time()
        self.timeout = 30

    def clock_cb(self,msg):
        #print 'msg',msg
        print 'clock',msg.clock#genpy.Time(msg.clock)
        #print time.
        self.clock = genpy.Time(msg.clock)

    def tf_cb(self,msg):
        #with self.clock:
        now = self.clock# + rospy.Duration(1.0)
        if True:#try:
            #self.listener.waitForTransform('/optitrak', '/head_center', now, rospy.Duration(10))
            #self.listener.waitForTransform('/optitrak', '/shaver', now, rospy.Duration(10))
            (transhead,rothead) = self.listener.lookupTransform('/optitrak', '/head_center', now)
            print transhead,rothead

            (transtool,rottool) = self.listener.lookupTransform('/optitrak', '/shaver', now)
            self.head.append([now,transhead[0],transhead[1],transhead[2],rothead[0],rothead[1],rothead[2],rothead[3]])
            self.h.write(''.join([now,' ',transhead[0],' ',transhead[1],' ',transhead[2],' ',rothead[0],' ',rothead[1],' ',rothead[2],' ',rothead[3],'\n']))
            self.tool.append([now,transtool[0],transtool[1],transtool[2],rottool[0],rottool[1],rottool[2],rottool[3]])
            self.t.write(''.join([now,' ',transtool[0],' ',transtool[1],' ',transtool[2],' ',rottool[0],' ',rottool[1],' ',rottool[2],' ',rottool[3],'\n']))
        #except:
            #print 'Something funny happened with the tf callback'
        #    pass
        if time.time()-self.start_time >self.timeout:
            self.close_files

    def close_files(self):
        self.h.close()
        self.t.close()







if __name__ == "__main__":
    rospy.init_node('bag_reader')
    reader = BagReader()
    #starttime= time.time()
    #while time.time()-starttime < 30:
    rospy.spin()
    #reader.close_files()





'''











