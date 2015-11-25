#!/usr/bin/env python  

#Chris Birmingham, HRL, 7/16/14
#This file takes predetermined tf transforms(relative to either the subjects
#head for feeding positions or the robots base for the bowl location)
#and broadcasts that transform. It then looks up the transform and converts
#it to a posed stamped message that the haptic mpc can use as a goal location.
#It acts upon recieving messages from the Main_Control and emergency topics.
#If STOP is recieved from the emergency topic it will dissable the Haptic mpc
#and exit.


import roslib
roslib.load_manifest('RYDS')
roslib.load_manifest('hrl_haptic_mpc')
import rospy
import math
import time
import numpy as np
import tf
import std_msgs.msg
import sys
import os
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from hrl_haptic_manipulation_in_clutter_srvs.srv import *
import geometry_msgs.msg
import tf_conversions.posemath as pm



class transformer():
    def __init__(self):
        rospy.init_node('convert_tf_to_pose')
        rospy.Subscriber('Main_Control', String, self.run)
        rospy.Subscriber('emergency', String, self.e_check)
        self.haptic = rospy.ServiceProxy('haptic_mpc/enable_mpc', EnableHapticMPC)
        self.broadcaster = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()
        self.pose_pub = rospy.Publisher('haptic_mpc/goal_pose', PoseStamped)
        self.rat = rospy.Rate(1)
        self.stop = ""
        self.i = 0

    #Called if msg sent to emergency topic
    def e_check(self, data):
        self.stop = data.data
        if self.stop != "STOP":
            return
        self.haptic('False')
        print "Stopped haptic control!"
        os._exit(0)


    #Broadcasts a set transform as the desired position based on task section
    #In Rviz, note translation is: (red,green,blue)
    #                      quaternian is: (red,green,blue,rotation)
    def broadcast(self, position):
        if position == "FacePos1":
            self.broadcaster.sendTransform((0.6, -0.1, -0.1),(3, 14, 13, -1),
                            rospy.Time.now(),"/FeedPos", "/head_frame")
            print "Broadcast transform for FacePos1"
        elif position =="FacePos2":
            self.broadcaster.sendTransform((0.26, -0.0, -0.0),(3, 14, 13, -1),
                            rospy.Time.now(),"/FeedPos", "/head_frame")
            print "Broadcast transform for FacePos2"
        elif position =="FacePos3":
            self.broadcaster.sendTransform((0.53, -0.0, -0.08),(3, 14, 13, -1),
                            rospy.Time.now(),"/FeedPos", "/head_frame")
            print "Broadcast transform for FacePos3"

    #Based on previously broadcast position, sends posed stamped msg to haptic mpc
    #Returns false if could not find the FeedPos frame, true if goal was successfully
    #   published to hmpc
    def send_to_hmpc(self, position):
        published = False
        try:
            (trans, rot) = self.listener.lookupTransform('/torso_lift_link', '/FeedPos', 
                                                                      rospy.Time(0))
            hdr = std_msgs.msg.Header()
            hdr.frame_id = '/torso_lift_link'
            hdr.stamp = rospy.Time.now()
            hdr.seq = self.i
            transa = geometry_msgs.msg.Point()
            transa.x = trans[0]
            transa.y = trans[1]
            transa.z = trans[2]
            rota = geometry_msgs.msg.Quaternion()
            rota.x = rot[0]
            rota.y = rot[1]
            rota.z = rot[2]
            rota.w = rot[3]          
            pose_msg = PoseStamped(header = hdr, pose = Pose(transa, rota))
            self.pose_pub.publish(pose_msg)
            published = True
            print "goal pose published to hmpc"
            self.i = self.i+1

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print "tf Exception, please broadcast FeedPos transform again"
        return published




    def run(self, data):
        #Initialization upon callback
        position = data.data
        
        if position[0] != "F":
            print ("Converter heard: %s, will keep listening." % position)
            return
        else:
            print ("Converter heard: %s, Starting." % position)

        task_set = rospy.Publisher('task_check', String)
        task_set.publish("Converter starting: %s" % position)

        #Broadcast the transform of the position for the first time
        self.broadcast(position)

        #Send desired pose to hmpc, returns true when pose has been sent
        published = False
        while (not rospy.is_shutdown() and published == False):
            published = self.send_to_hmpc(position)
            self.rat.sleep()

        #check if close enough to goal pose, ask for for feedback
        #when close enough, broadcast current location as goal pose, send to hmpc 
        #and tell main program it is done moving
        while (not rospy.is_shutdown() and self.stop != "STOP"):
            self.broadcast(position)
            try:
                (trans, rot) = self.listener.lookupTransform('/l_wrist_roll_link', '/FeedPos', rospy.Time(0))
                print trans, rot
                if (trans[1] < .09 and trans[2] < .09):
                    check = raw_input("(r)epeat printout of transform or (c)ontinue to next task?")
                    if check == 'c':
                        msg = position + "Done"
                        task_set.publish(msg)
                        break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print "lookup exception"
                continue
            self.rat.sleep()

if __name__ == '__main__':
    a = transformer()
    while not rospy.is_shutdown():
        rospy.spin()




       
