#!/usr/bin/env python

# System
import numpy as np, math
import time
import sys
import threading

# ROS
import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy
import tf
from geometry_msgs.msg import Wrench

# HRL
from hrl_srvs.srv import None_Bool, None_BoolResponse
from hrl_msgs.msg import FloatArray
import force_torque.FTClient as ftc

# Private
#import hrl_anomaly_detection.door_opening.mechanism_analyse_daehyung as mad
#import hrl_anomaly_detection.advait.arm_trajectories as at


class adl_recording():
    def __init__(self, obj_id_list, netft_flag_list):
        self.ftc_list = []                                                                                       
        for oid, netft in zip(obj_id_list, netft_flag_list):                                                     
            self.ftc_list.append(ftc.FTClient(oid, netft))                                                       
        self.oid_list = copy.copy(obj_id_list)
        
        ## self.initComms()
        pass

        
    def initComms(self):
        
        # service
        #rospy.Service('door_opening/mech_analyse_enable', None_Bool, self.mech_anal_en_cb)
        
        # Subscriber
        rospy.Subscriber('/netft_rdt', Wrench, self.ft_sensor_cb)                    

        
    # returns a dict of <object id: 3x1 np matrix of force>
    def get_forces(self, bias = True):
        f_list = []
        for i, ft_client in enumerate(self.ftc_list):
            f = ft_client.read(without_bias = not bias)
            f = f[0:3, :]

            ## trans, quat = self.tf_lstnr.lookupTransform('/torso_lift_link',
            ##                                             self.oid_list[i],
            ##                                             rospy.Time(0))
            ## rot = tr.quaternion_to_matrix(quat)
            ## f = rot * f
            f_list.append(-f)

        return dict(zip(self.oid_list, f_list))


    def bias_fts(self):
        for ftcl in self.ftc_list:
            ftcl.bias()
        
        
    # TODO
    def ft_sensor_cb(self, msg):

        with self.ft_lock:
            self.ft_data = [msg.force.x, msg.force.y, msg.force.z] # tool frame

            # need to convert into torso frame?


    def start(self, bManual=False):
        rospy.loginfo("ADL Online Recording Start")

        ar.bias_fts()
        rospy.loginfo("FT bias complete")

        
        rate = rospy.Rate(2) # 25Hz, nominally.
        rospy.loginfo("Beginning publishing waypoints")
        while not rospy.is_shutdown():         
            f = self.get_forces()[0]
            print f
            #print rospy.Time()
            rate.sleep()        
            
    

if __name__ == '__main__':

    ar = adl_recording()   
    ar.start()

    
