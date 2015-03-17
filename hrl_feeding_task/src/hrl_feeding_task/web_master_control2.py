#! /usr/bin/python

#Chris Birmingham and Megan Rich, HRL, 7/16/14
#This file controls the state the robot is in and what actions
#it performs. It broadcasts the robot state to the Main_Control topic
#and listens to the task_check topic to know when action nodes are done
#performing an action and listens to emergency to know when an anomaly has
#occured. It also listens to the user_input to collect start, stop and continue
#commands from the HRL assitive_teleop web interface. 

from RYDS.srv import *
import roslib; roslib.load_manifest('RYDS')
roslib.load_manifest('snapshot')
roslib.load_manifest('rospy')
roslib.load_manifest('actionlib')
roslib.load_manifest('pr2_controllers_msgs')
roslib.load_manifest('geometry_msgs')
roslib.load_manifest('hrl_haptic_mpc')
from std_msgs.msg import String, Bool
import actionlib
from actionlib_msgs.msg import *
from pr2_controllers_msgs.msg import *
from geometry_msgs.msg import *
from hrl_haptic_manipulation_in_clutter_srvs.srv import *
import sys
import os
import rospy
from snapshot.srv import *
from hrl_srvs.srv import None_Bool, None_BoolResponse


###Initializations###
#starts publishers, subscribers, and services needed for running
#the yogurt feeding process

class Master_Control():
    def __init__(self):
        rospy.init_node('master_control')
        self.globev_task_check = ""
        self.globev_emergency = ""
        self.globev_main_ctrl = ""
        self.r = rospy.Rate(10)
        self.stopcheck = rospy.Publisher('emergency', String)
        self.continue_message=rospy.Publisher('Continue_message', String)
        self.stopcheck.publish("GO")
        self.send = rospy.Publisher('Main_Control', String)
        self.send.publish("Wait")
        self.task_set = rospy.Publisher('task_check', String)
        self.task_set.publish("no")
        print "waiting for compare_histo and snap_node and haptic mpc"
        rospy.wait_for_service('haptic_mpc/enable_mpc')
        rospy.wait_for_service('right/haptic_mpc/enable_mpc')
        self.haptic = rospy.ServiceProxy('haptic_mpc/enable_mpc', EnableHapticMPC)
        #self.haptic('False')
        self.r_haptic = rospy.ServiceProxy('right/haptic_mpc/enable_mpc', EnableHapticMPC)
        #self.r_haptic('False')
        rospy.wait_for_service('compare_histo')
        rospy.wait_for_service('snap_node')
        print "compare_histo and snap_node and haptic mpc found"
        self.pic = rospy.ServiceProxy('snap_node', CompareHisto)        
        #self.haptic('False')

        
        rospy.wait_for_service("/feeding/init_arms")
        self.init_arms = rospy.ServiceProxy("/feeding/init_arms", None_Bool)
        
        self.message=''
        self.part=0
        self.count=0
        rospy.Subscriber('emergency', String, self.check_emerg)
        rospy.Subscriber('task_check', String, self.check_task)
        rospy.Subscriber('Main_Control', String, self.check_mctrl)
        rospy.Subscriber('user_input', String, self.check_user)
        rospy.Subscriber('head_check_confirm', String, self.head_check)
        
        print "Init complete!!"
        
        

###Subscriber callbacks set global variables used by all functions###

    def check_task(self, word): #Checking if the task is complete
        self.globev_task_check = word.data

    def check_emerg(self, word): #Checking if anomaly detected
        self.globev_emergency = word.data
        print "Stop recieved!"
        os._exit(0) #exits when stop is recieved
    def check_mctrl(self, word): #Checking what state the robot is in
        self.globe_main_ctrl = word.data
    #def usr_input(self, words):
    #   self.reciever = raw_input('%s' % words)
    
    def head_check(self, data): #check if the head is within range of the PR2s arms
        mes=data.data
        if mes is 'Ready': 
            print('Head check confirmation received')
            #rospy.Subscriber('user_input', String, self.check_user)
    
    def check_user(self, msg): #check input from user (Phil's web interface)
        self.message=msg.data
        print self.message
        if self.message == 'Stop':
            print('Stop confirmed')
            self.send.publish("STOP")
            os._exit(0)
        if self.message == 'Start':
            print('Start confirmed')  
            self.run_task()
        
        if self.message == 'Continue':
           print('Continue confirmed')
           self.send.publish("STOP")
           self.part = 0
           self.run_task()
        

###Task Control###
#Publish msgs to control topic and wait for action node to tell when done while
#checking for anomaly on emergency topic. When finished will reset task done variable
    
    def task_control(self, task):  #controls what part of the task the robot is in
        if self.globev_emergency == "STOP":
            return
        print ("Main ctrl in %s task." % task)
        #publish the task to the goal setter that will send the correct pose to the haptic mpc
        self.send.publish(task)
        while (self.globev_emergency != "STOP" and self.globev_task_check != (task+"Done") and not rospy.is_shutdown()):
            self.r.sleep()
        self.task_set.publish("no")
        print ("Main ctrl done with %s" % task)

#Moves head to take picture of target frame
    def move_to_pic(self, frame):
        if self.globev_emergency == "STOP":
            return
        if frame == 'head_frame':
            client = actionlib.SimpleActionClient('/head_traj_controller/point_head_action', PointHeadAction)
            client.wait_for_server()
            g = PointHeadGoal()
            g.target.header.frame_id = frame
            g.target.point.x = 0.2
            g.target.point.y = -0.2
            g.target.point.z = 0.0
            g.min_duration = rospy.Duration(1.0)
            client.send_goal(g)
            client.wait_for_result()
        else: #move head head to look at the spoon frame
            client = actionlib.SimpleActionClient('/head_traj_controller/point_head_action', PointHeadAction)
            client.wait_for_server()
            g = PointHeadGoal()
            g.target.header.frame_id = frame
            g.target.point.x = 0.0
            g.target.point.y = 0.0
            g.target.point.z = 0.0
            g.min_duration = rospy.Duration(1.0)

            client.send_goal(g)
            client.wait_for_result() 
            
          


###Histogram Control###
#take_pic takes a picture of the spoon and saves the picture
#for making a comparison
    def take_pic(self):
        pit = -1
        while pit == -1:
            if self.globev_emergency == "STOP":
                return
            print "taking first picture"
            pit = 0
            pit = self.pic(0)
        print "picture returned"


    def compare_pic(self): #use the compare histogram service to compare the
    #before and after pictures of the yogurt scooping action
        if self.globev_emergency == "STOP":
            return
        print "comparing picture"
        pit = 0
        pit = self.pic(1)
        rec = rospy.ServiceProxy('compare_histo', CompareHisto)
        ret = rec(0)
        print "made it this far"
        print "%d"%ret.R
        got_yogurt = False
        if ret.R == 1:
            print "No major change, scoop again."
            got_yogurt = False
        elif ret.R == 0 :
            print "Changes occured, proceeding"
            got_yogurt = True
        else :
            print "Image Loaded for first time"

        got_yogurt = True
            
        return got_yogurt

###Main Function###
#
    def run_task(self):
        #move head to look at subject
        self.move_to_pic('/head_frame')
        #raw_input("Press enter when ready to begin")
          
        self.haptic('enabled')
        self.r_haptic('enabled')        
        ret = self.init_arms()        

        self.part = 0
        while not rospy.is_shutdown() and self.globev_emergency != "STOP":
            #if the task is over ask if the subject would like to continue or stop
            if self.part > 12 or self.part < 0:
                #self.send.publish("STOP")
                print ('Continue?')
                self.continue_message.publish('Continue?')
                self.send.publish("STOP")

                if self.message == 'Stop':
                    os._exit(0)
                    
                ret = self.init_arms()        

                self.part=0
                break
            elif self.part == 13:
                break

            #send part of task to the goal setter
            self.task_control("Part%s" %str(self.part))
            self.task_control("Part%s" %str(self.part)) #Done twice to ensure it reaches the correct location
            rospy.sleep(2)
             
            #move head to look at spoon and take picture
            if self.part == 0:
                #rospy.sleep(3)
                print "part 0 : take picture!!!"
                self.move_to_pic('/l_gripper_spoon_frame')
                self.take_pic()
                ## rospy.sleep(20)

            #take another picture and compare the two
            if self.part == 7:
                self.task_control("Part%s" %str(self.part))
                print "take picture!!!"
                test = self.compare_pic()
                ## rospy.sleep(20)

                if test == False:
                    ret = self.init_arms()
                    self.part = 0
            #pause between parts 10 and 11
            if self.part == 10:
                rospy.sleep(1)

            
            self.part = self.part + 1




if __name__ == "__main__":
    Master_Control()
    while not rospy.is_shutdown():
        rospy.spin()    


