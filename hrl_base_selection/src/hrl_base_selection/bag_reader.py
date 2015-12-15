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
    bag_path = '/home/ari/svn/robot1_data/2011_ijrr_phri/log_data/bag/'
    
    #bag_name = 'sub1_shaver'
    #bag_name = 'sub3_shaver'
    bag_name = 'sub6_shaver'
    

    adl2_B_toolframe = createBMatrix([0.234, 0.041, -0.015], [0, 0, 0, 1])
    toolframe_B_shaver = createBMatrix([0.070, 0., 0.], [0, 0, 0, 1])
    shaver_B_pr2goal = createBMatrix([-0.050, 0., 0.], [0, 0, 0, 1])
    adl2_B_pr2goal = adl2_B_toolframe*toolframe_B_shaver*shaver_B_pr2goal
    head_B_headc = createBMatrix([0.21, 0.02, -0.087], [0, 0, 0, 1])
    print 'Starting on the tool log file! \n'
    bag = rosbag.Bag(''.join([bag_path,bag_name,'.bag']), 'r')
    tool = open(''.join([pkg_path,'/data/',bag_name,'_tool.log']), 'w')
    for topic, tf_msg, t in bag.read_messages():
        if topic == "/tf":
            if len(tf_msg.transforms) > 0 and tf_msg.transforms[0].child_frame_id == "/adl2":
                tx = copy.copy(tf_msg.transforms[0].transform.translation.x)
                ty = copy.copy(tf_msg.transforms[0].transform.translation.y)
                tz = copy.copy(tf_msg.transforms[0].transform.translation.z)
                rx = copy.copy(tf_msg.transforms[0].transform.rotation.x)
                ry = copy.copy(tf_msg.transforms[0].transform.rotation.y)
                rz = copy.copy(tf_msg.transforms[0].transform.rotation.z)
                rw = copy.copy(tf_msg.transforms[0].transform.rotation.w)
                world_B_shaver = createBMatrix([tx,ty,tz],[rx,ry,rz,rw])*adl2_B_pr2goal
                eul = tft.euler_from_matrix(world_B_shaver,'rxyz')
                #print world_B_shaver,'\n'
                #print eul,'\n'
                #print 'time: \n',t
                tool.write(''.join([str(t),' %f %f %f %f %f %f \n' % (world_B_shaver[0,3],world_B_shaver[1,3],world_B_shaver[2,3],eul[0],eul[1],eul[2])]))
                #tool.write(''.join([world_B_shaver[0,3],' ',world_B_shaver[1,3],' ',world_B_shaver[2,3],' ',eul[0],' ',eul[1],' ',eul[2],'\n']))
    bag.close()
    tool.close()
    print 'Starting on the head log file! \n'
    bag = rosbag.Bag(''.join([bag_path,bag_name,'.bag']), 'r')
    head = open(''.join([pkg_path,'/data/',bag_name,'_head.log']), 'w')
    for topic, tf_msg, t in bag.read_messages():
        if topic == "/tf":
            if len(tf_msg.transforms) > 0 and tf_msg.transforms[0].child_frame_id == "/head":
                tx = copy.copy(tf_msg.transforms[0].transform.translation.x)
                ty = copy.copy(tf_msg.transforms[0].transform.translation.y)
                tz = copy.copy(tf_msg.transforms[0].transform.translation.z)
                rx = copy.copy(tf_msg.transforms[0].transform.rotation.x)
                ry = copy.copy(tf_msg.transforms[0].transform.rotation.y)
                rz = copy.copy(tf_msg.transforms[0].transform.rotation.z)
                rw = copy.copy(tf_msg.transforms[0].transform.rotation.w)
                world_B_headc = createBMatrix([tx,ty,tz],[rx,ry,rz,rw])*head_B_headc
                eul = copy.copy(tft.euler_from_matrix(world_B_headc,'rxyz'))
                head.write(''.join([str(t),' %f %f %f %f %f %f \n' % (world_B_headc[0,3],world_B_headc[1,3],world_B_headc[2,3],eul[0],eul[1],eul[2])]))
                #head.write(''.join([t,' ',world_B_head[0,3],' ',world_B_head[1,3],' ',world_B_head[2,3],' ',eul[0],' ',eul[1],' ',eul[2],' ',eul[3],'\n']))
    bag.close()
    head.close()

    print "Saved files!"









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











