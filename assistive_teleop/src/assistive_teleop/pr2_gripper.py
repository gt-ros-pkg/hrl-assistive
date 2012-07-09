#/usr/bin/python

import roslib; roslib.load_manifest('assistive_teleop')
import rospy

import actionlib
from pr2_controllers_msgs.msg import (Pr2GripperCommandGoal, Pr2GripperCommand,
                                      Pr2GripperCommandAction)

class PR2Gripper():
    def __init__(self, arm):
        self.arm = arm

        self.def_gripper_ac = actionlib.SimpleActionClient(
                         self.arm[0]+"_gripper_controller/gripper_action",
                         Pr2GripperCommandAction)
        if self.def_gripper_ac.wait_for_server(rospy.Duration(30)):
            rospy.loginfo("Found default "+self.arm+"_gripper action")
        else:
            rospy.logwarn("Cannot find default"+self.arm+"_gripper action")
            rospy.logwarn(self.arm+" GRIPPER ACTION NOT FOUND")

    
    def grab(self, position=0.0, max_effort=-1.0,  block=False, timeout=20):
        """Place-holder for more interesting grab"""           
        return self.gripper_action(position, max_effort, block, timeout)

    def release(self, position=0.0, max_effort=-1.0, block=False, timeout=20):
        """Place-holder for more interesting release"""           
        return self.gripper_action(position, max_effort, block, timeout)

    def gripper_action(self, position, max_effort=-1, 
                             block=False, timeout=20.0):
       """Send goal to gripper action server"""
       self.def_gripper_ac.send_goal(Pr2GripperCommand(position, max_effort))
       if block:
           return self.def_gripper_ac.wait_for_result(rospy.Duration(timeout))
        
