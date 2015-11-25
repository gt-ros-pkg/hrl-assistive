#!/usr/bin/env python  

#Daehyung Park, HRL, 8/10/14
#This file is updated version of goal_setter.py. Previous structure is kept, but
#I am gradually rewriting everything. Finally, the structure will be also replaced 
#reasonable structure. Currently, most new things are copies from hrl_mpc_action
#from sandbox_dpark_darpa_m3 package. 

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
roslib.load_manifest('tf')
roslib.load_manifest('tf_conversions')
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
import threading

import hrl_haptic_manipulation_in_clutter_msgs.msg as haptic_msgs
import hrl_msgs.msg
from hrl_srvs.srv import None_Bool, None_BoolResponse


class transformer():
    def __init__(self):
        rospy.init_node('goal_setter')
        
        self.rate = rospy.Rate(.5)
        self.stop = ""
        self.i = 0

        self.bowl_lock = threading.RLock() ## bowl state lock

        self.initComms()    
        #rospy.sleep(2.0)
        ## self.initArms()            
        pass


    def initComms(self):

        self.haptic = rospy.ServiceProxy('haptic_mpc/enable_mpc', EnableHapticMPC)
        self.r_haptic = rospy.ServiceProxy('right/haptic_mpc/enable_mpc', EnableHapticMPC)
        self.broadcaster = tf.TransformBroadcaster()
        self.r_broadcaster = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()
        self.pose_pub = rospy.Publisher('haptic_mpc/goal_pose', PoseStamped)
        self.r_pose_pub = rospy.Publisher('right/haptic_mpc/goal_pose', PoseStamped)
        self.r_pose_pub = rospy.Publisher('right/haptic_mpc/goal_pose', PoseStamped)
        self.task_set = rospy.Publisher('task_check', String)
        ## self.task_name_pub = rospy.Publisher('feeding/task_name', String)
        
        self.r_mpc_weights_pub = rospy.Publisher("right/haptic_mpc/weights", haptic_msgs.HapticMpcWeights)               
        self.mpc_weights_pub = rospy.Publisher("/haptic_mpc/weights", haptic_msgs.HapticMpcWeights)          
        self.goal_posture_pub = rospy.Publisher("haptic_mpc/goal_posture", hrl_msgs.msg.FloatArray)        
        self.r_goal_posture_pub = rospy.Publisher("right/haptic_mpc/goal_posture", hrl_msgs.msg.FloatArray)              
        rospy.Subscriber('Main_Control', String, self.run)
        rospy.Subscriber('emergency', String, self.e_check)
        rospy.Subscriber('RYDS_BowlConfirmation', PoseStamped, self.bowlPoseCallback)

        self.init_arm_service = rospy.Service('/feeding/init_arms', None_Bool, self.initArmsCallback)
        print "Init comms complete"

        rospy.sleep(2.0)
        l_joint_state = [1.7013504719569787, -0.2846619162464899, 1.0247881430005377, -1.0400059974175215, 0.7408476425758285, -0.9261340129014745, -0.8541080908968821]
        r_joint_state = [-1.3805018627854437, -0.3065720013305438, -0.6643104933210333, -1.6377642647201074, -0.014866701346294675, -0.9982517431192833, 2.579269529149009]

        print "Initializing both arm configurations!!"
        self.setPostureGoal(l_joint_state, arm='l')
        self.setPostureGoal(r_joint_state, arm='r')
        rospy.sleep(10.0)
        
        
    def initArmsCallback(self, req):

        self.haptic('enabled')
        
        l_joint_state = [1.7013504719569787, -0.2846619162464899, 1.0247881430005377, -1.0400059974175215, 0.7408476425758285, -0.9261340129014745, -0.8541080908968821]
        r_joint_state = [-1.3805018627854437, -0.3065720013305438, -0.6643104933210333, -1.6377642647201074, -0.014866701346294675, -0.9982517431192833, 2.579269529149009]

        print "Initializing both arm configurations!!"
        self.setPostureGoal(l_joint_state, arm='l')
        self.setPostureGoal(r_joint_state, arm='r')

        rospy.sleep(10.0)
        return None_BoolResponse(True)
        
        

    def setPostureGoal(self, lJoint, arm='l'):
        
        goal_data = hrl_msgs.msg.FloatArray()                                                                                                                                                                                        
        goal_data.header.stamp = rospy.Time.now() 
        goal_data.data = lJoint

        # Send a goal
        self.setPostureControl(arm)

        if arm=='l':
            self.goal_posture_pub.publish(goal_data)
        else:
            self.r_goal_posture_pub.publish(goal_data)

        ## return self.checkMovement(0.001, timeout)

        
    def setPositionControl(self, arm='l'):
        self.weights_msg = haptic_msgs.HapticMpcWeights()
        self.weights_msg.header.stamp = rospy.Time.now()
        self.weights_msg.position_weight = 5.0        
        self.weights_msg.orient_weight   = 0.0
        self.weights_msg.posture_weight  = 0.0                
        if arm=='l':
            self.mpc_weights_pub.publish(self.weights_msg) # Enable position tracking only - disable orientation by setting the weight to 0 
        else:
            self.r_mpc_weights_pub.publish(self.weights_msg) # Enable position tracking only - disable orientation by setting the weight to 0 

    def setOrientControl(self, arm='l'):
        self.weights_msg = haptic_msgs.HapticMpcWeights()
        self.weights_msg.header.stamp = rospy.Time.now()
        self.weights_msg.position_weight = 5.0        
        self.weights_msg.orient_weight   = 4.0
        self.weights_msg.posture_weight  = 0.0                
        if arm=='l':
            self.mpc_weights_pub.publish(self.weights_msg) # Enable position tracking only - disable orientation by setting the weight to 0 
        else:
            self.r_mpc_weights_pub.publish(self.weights_msg) # Enable position tracking only - disable orientation by setting the weight to 0 
            
    def setPostureControl(self, arm='l'):
        self.weights_msg = haptic_msgs.HapticMpcWeights()
        self.weights_msg.header.stamp = rospy.Time.now()
        self.weights_msg.position_weight = 0.0
        self.weights_msg.orient_weight   = 0.0
        self.weights_msg.posture_weight  = 5.0        
        if arm=='l':     
            self.mpc_weights_pub.publish(self.weights_msg) # Enable position and orientation tracking         
        else:
            self.r_mpc_weights_pub.publish(self.weights_msg) # Enable position and orientation tracking         

        
    
    #emergency callback function
    def e_check(self, data):
        self.stop = data.data
        if self.stop != "STOP":
            return
        self.haptic('False')
        print "Stopped haptic control!"
        os._exit(0)

    # Yogrt bowl pose callback function
    def bowlPoseCallback(self, msg):
        with self.bowl_lock:
            self.bowl_frame = msg.header.frame_id
            self.bowl_pos = np.matrix([[msg.pose.position.x], [msg.pose.position.y], [msg.pose.position.z]])
            #self.goal_orient_quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
            print "Bowl position confirmed" 
            
    #Broadcasts a set transform as the desired position based on task section
    #In Rviz, note translation is: (red,green,blue)
    #               quaternian is: (red,green,blue,rotation)
    def broadcast(self, position):

        self.setOrientControl("l")
        self.setOrientControl("r")

        x_offset = -0.025
        y_offset = -0.0
        #z_offset = -0.01

        x_head_offset = -0.13-0.05
            
        #going to home location in front of camera:
        if position == "Part0":
                
            ## self.broadcaster.sendTransform((0.5309877259429142, 0.4976163448816489, 0.16719537682372823),(0.740, 0.052, -0.100, 0.663),
            ##                                rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            self.broadcaster.sendTransform((0.5309877259429142, 0.4976163448816489, 0.16719537682372823),(0.740, 0.052, -0.100, 0.663),
                                           rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            print "Broadcast transform for Pos0"

            
            self.r_broadcaster.sendTransform((0.425, -0.480, 0.236),(0.021, -0.011, 0.158, 0.987),
                            rospy.Time.now(),"/r_GoalPos", "/torso_lift_link")
            print "Broadcast transforms for Pos0R"


        #moving vertically to over bowl:
        elif position == "Part1":
            self.broadcaster.sendTransform((self.bowl_pos[0]+x_offset, self.bowl_pos[1]+y_offset, self.bowl_pos[2]+0.2),(0.740, 0.052, -0.100, 0.663),
                            rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            # self.broadcaster.sendTransform((0.516341299985487, 0.8915608293219441, 0.1950343868326016),(0.6567058177198967, 0.16434420640210323, 0.0942917725129517, 0.7299571990406495),
            #                 rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            print "Broadcast transform for Pos1"
        #dipping at an angle over the bowl:
        elif position == "Part2":
            self.broadcaster.sendTransform((self.bowl_pos[0]+x_offset, self.bowl_pos[1]+y_offset, self.bowl_pos[2]+0.2),(0.601, 0.397, -0.412, 0.558),
                            rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            # self.broadcaster.sendTransform((0.5193456827844327, 0.900079836777675, -0.019204479089017762),(0.4954470843513707, 0.5023693425664104, -0.12672521702586453, 0.6972072501250012),
            #                 rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            print "Broadcast transform for Pos2"
        #going to the base of the bowl:
        elif position == "Part3":
            self.broadcaster.sendTransform((self.bowl_pos[0]-0.02+x_offset, self.bowl_pos[1]+y_offset, self.bowl_pos[2]+0.01),(0.601, 0.397, -0.412, 0.558),
                            rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            # self.broadcaster.sendTransform((0.5098543997629579, 0.8806008953235813, -0.0974591835731535),(0.45253993336907683, 0.533997128372586, -0.17283744712356874, 0.6929515801756649),
            #                 rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            print "Broadcast transform for Pos3"
        #going to other end of the bowl in a scooping motion:
        elif position == "Part4":
            self.broadcaster.sendTransform((self.bowl_pos[0]+0.03+x_offset, self.bowl_pos[1]+y_offset, self.bowl_pos[2]-0.0),(0.700, 0.201, -0.229, 0.646),
                            rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            # self.broadcaster.sendTransform((0.5418855469129493, 0.9140635229546514, -0.10433053967271771),(0.4903488201115364, 0.49639824283005096, -0.1400581861447639, 0.7025172763884889),
            #                 rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            print "Broadcast transform for Pos4"
        #going to the lip of the bowl in order to wipe of excess yogurt:
        elif position == "Part5":
            self.broadcaster.sendTransform((self.bowl_pos[0]+x_offset, self.bowl_pos[1]+y_offset, self.bowl_pos[2]+0.2),(0.740, 0.052, -0.100, 0.663),
                            rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            # self.broadcaster.sendTransform((0.5097778641738854, 0.8811538278444637, -0.07757980710747647),(0.3515887722286045, 0.6131005055762095, -0.07870017707244904, 0.7030642839980877),
            #                 rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            print "Broadcast transform for Pos5"
        #going back to above bowl:
        elif position == "Part6":
            self.broadcaster.sendTransform((self.bowl_pos[0]+x_offset, self.bowl_pos[1]+y_offset, self.bowl_pos[2]+0.2),(0.740, 0.052, -0.100, 0.663),
                            rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            # self.broadcaster.sendTransform((0.516341299985487, 0.8915608293219441, 0.1950343868326016),(0.6567058177198967, 0.16434420640210323, 0.0942917725129517, 0.7299571990406495),
            #                 rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            print "Broadcast transform for Pos6"
        #going to back to home location for histogram check:
        elif position == "Part7":
            self.broadcaster.sendTransform((0.5309877259429142, 0.4976163448816489, 0.16719537682372823),(0.7765742993649133, -0.37100605554316285, -0.27784851903166524, 0.42671660945891),
                            rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            print "Broadcast transform for Pos7" #
        #going to in front of subjects face:
        elif position == "Part8":
            self.broadcaster.sendTransform((0.2741387011303321, 0.03540-0.01, -0.011919598309888757+0.03),(-0.023580897114171894, 0.7483633417869068, 0.662774596931439, 0.011228696415565394),
                            rospy.Time.now(),"/GoalPos", "/head_frame")
            print "Broadcast transform for Pos8"
        #going to subjects mouth:
        elif position == "Part9":
            self.broadcaster.sendTransform((0.12608+x_head_offset, 0.03540, 0.00607),(-0.015224467044577382, 0.7345761465214938, 0.6783020152473445, -0.008513323454022942),
                            rospy.Time.now(),"/GoalPos", "/head_frame")
            print "Broadcast transform for Pos9"
        #going to eye level but at an angle so as to encourage swallowing through tongue depression:
        elif position == "Part10":
            self.broadcaster.sendTransform((0.28264, 0.03715, 0.01361),(-0.050599231123428158, 0.7586479615613725, 0.6342031296736002, 0.0013829359935748063),
                            rospy.Time.now(),"/GoalPos", "/head_frame")
            print "Broadcast transform for Pos10"

            #rospy.sleep(1)
            self.r_broadcaster.sendTransform((0.12608+x_head_offset+0.06, 0.03540, 0.00607),(-0.015224467044577382, 0.7345761465214938, 0.6783020152473445, -0.008513323454022942),
                            rospy.Time.now()+rospy.Duration(1),"/r_GoalPos", "/head_frame")
            print "Broadcast transforms for Pos10R"


        #going to eye level but at an angle so as to encourage swallowing through tongue depression:
        elif position == "Part11":
            self.broadcaster.sendTransform((0.2826437596817016, 0.03715712909254376, 0.0136172136416570133),(-0.050599231123428158, 0.7586479615613725, 0.6342031296736002, 0.0013829359935748063),
                            rospy.Time.now(),"/GoalPos", "/head_frame")
            print "Broadcast transform for Pos11"

            #rospy.sleep(1)
            self.r_broadcaster.sendTransform((0.12608632401364894+0.25, 0.03540318703608347, 0.00607600258150498),(-0.015224467044577382, 0.7345761465214938, 0.6783020152473445, -0.008513323454022942),
                            rospy.Time.now()+rospy.Duration(1),"/r_GoalPos", "/head_frame")
            print "Broadcast transforms for Pos11R"
            
        #going back to home poistion:
        elif position == "Part12":
            self.broadcaster.sendTransform((0.564, 0.797, 0.185),(0.740, 0.052, -0.100, 0.663),
                            rospy.Time.now(),"/GoalPos", "/torso_lift_link")
            print "Broadcast transforms for Pos12L"


            self.r_broadcaster.sendTransform((0.425, -0.480, 0.236),(0.021, -0.011, 0.158, 0.987),
                            rospy.Time.now(),"/r_GoalPos", "/torso_lift_link")
            #self.r_broadcaster.sendTransform((0.5407266943829659, 0.02987365197085315, 0.12104661416190984),(0.8281494537699263, 0.47993966894730516, -0.18113711788590195, 0.22586664409627374),
            #                rospy.Time.now(),"/r_GoalPos", "/torso_lift_link")
            print "Broadcast transforms for P12R"



    #Based on previously broadcast position, sends posed stamped msg to haptic mpc
    #Returns false if could not find the FeedPos frame, true if goal was successfully
    #   published to hmpc
    def send_to_hmpc(self):
        published = False
        try:
            self.listener.waitForTransform('/torso_lift_link', '/GoalPos', rospy.Time(0), rospy.Duration(3.0))   
            (trans, rot) = self.listener.lookupTransform('/torso_lift_link', '/GoalPos', rospy.Time(0))
            hdr = std_msgs.msg.Header()
            hdr.frame_id = '/torso_lift_link'
            hdr.stamp = rospy.Time.now()
            #hdr.seq = self.i
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

    def r_send_to_hmpc(self):
        
        published = False
        try:
            self.listener.waitForTransform('/torso_lift_link', '/r_GoalPos', rospy.Time(0), rospy.Duration(3.0))   
            (trans, rot) = self.listener.lookupTransform('/torso_lift_link', '/r_GoalPos', rospy.Time(0))
            hdr = std_msgs.msg.Header()
            hdr.frame_id = '/torso_lift_link'
            hdr.stamp = rospy.Time.now()
            #hdr.seq = self.i
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
            self.r_pose_pub.publish(pose_msg)
            published = True
            print "goal pose published to r_hmpc"
            self.i = self.i+1

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print "tf Exception, please broadcast FeedPos transform again RR"
        return published


    def run(self, data):

        #Initialization upon callback
        position = data.data
        
        if position[0] != "P":
            print ("Converter heard: %s, will keep listening." % position)
            return
        else:
            print ("Converter heard: %s, Starting." % position)

        
        self.task_set.publish("Converter starting: %s" % position)

        #Broadcast the transform of the position for the first time
        self.broadcast(position)

        #Send desired pose to hmpc, returns true when pose has been sent
        published = False
        while (not rospy.is_shutdown() and published == False):
            published = self.send_to_hmpc()
            if position == "Part0" or position == "Part10" or position == "Part12":
                published = self.r_send_to_hmpc()
            self.rate.sleep()

        #check if close enough to goal pose, ask for for feedback
        #when close enough, broadcast current location as goal pose, send to hmpc 
        #and tell main program it is done moving
        timeout = 0
        while (not rospy.is_shutdown() and self.stop != "STOP"):

            self.broadcast(position)
            if position == "Part0" or position == "Part10" or position == "Part12":
                published = self.r_send_to_hmpc()
            try:
                self.listener.waitForTransform('/l_gripper_spoon_frame', '/GoalPos', rospy.Time(0), rospy.Duration(3.0))   
                (trans, rot) = self.listener.lookupTransform('/l_gripper_spoon_frame', '/GoalPos', rospy.Time(0))
                print trans, rot
                if (trans[1] < .03 and trans[2] < .03 and trans[0] < .03 and rot[0] < .3 and rot[1] < .3 and rot[2] < .3):
                    #check = raw_input("(r)epeat printout of transform or (c)ontinue to next task?")
                    #if check == 'c':
                    rospy.sleep(2)
                    msg = position + "Done"
                    self.task_set.publish(msg)
                    print "done"
                    break
                self.send_to_hmpc()
                timeout = timeout +1
                if timeout == 4:
                    msg = position + "Done"
                    self.task_set.publish(msg)
                    print "done"
                    break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print "tf lookup exception"
                continue

            self.rate.sleep()

if __name__ == '__main__':
    a = transformer()
    while not rospy.is_shutdown():
        rospy.spin()




       
