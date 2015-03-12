#!/usr/bin/env python  

import roslib; roslib.load_manifest('sandbox_dpark_darpa_m3')
roslib.load_manifest('hrl_feeding_task')
import rospy
import numpy as np, math
import time

import hrl_haptic_mpc.haptic_mpc_util as haptic_mpc_util

from hrl_srvs.srv import None_Bool, None_BoolResponse
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sandbox_dpark_darpa_m3.lib.hrl_mpc_base import mpcBaseAction


class armReachAction(mpcBaseAction):
    def __init__(self, d_robot, controller, arm='l'):

        mpcBaseAction.__init__(self, d_robot, controller, arm)
        
        # service request
        self.reach_service = rospy.Service('/arm_reach_enable', None_Bool, self.start_cb)


        rate = rospy.Rate(100) # 25Hz, nominally.                
        while not rospy.is_shutdown():

            if self.getJointAngles() != []:
                print "----------------------"
                print "Current joint angles"
                print self.getJointAngles()
                print "Current pose"
                print self.getEndeffectorPose()
                print "----------------------"
                break
        
    def start_cb(self, req):

        # Run manipulation tasks
        if self.run():
            return None_BoolResponse(True)
        else:
            return None_BoolResponse(False)

        
    def run(self):

        pos  = Point()
        quat = Quaternion()


        ## confirm = False
        ## while not confirm:
        ##     print "Current pose"
        ##     print self.getEndeffectorPose()
        ##     ans = raw_input("Enter y to confirm to start: ")
        ##     if ans == 'y':
        ##         confirm = True
        ## print self.getJointAngles()        
        

        #----MAIN STUFF

        # print "MOVES1"
        # (pos.x, pos.y, pos.z) = (1.0, 0.27, 0.005)
        # (quat.x, quat.y, quat.z, quat.w) = (0.253, -0.141, -0.252, 0.923)
        # timeout = 1.0
        # self.setOrientGoal(pos, quat, timeout)
        # ## print self.getEndeffectorPose()
        
        # print "MOVES2"
        # (pos.x, pos.y, pos.z) = (1.0, 0.27, 0.005-0.2)
        # (quat.x, quat.y, quat.z, quat.w) = (0.253, -0.141, -0.252, 0.923)
        # timeout = 1.0 
        # self.setOrientGoal(pos, quat, timeout)

        # print "MOVES3"
        # (pos.x, pos.y, pos.z) = (1.0, 0.27, 0.005)
        # (quat.x, quat.y, quat.z, quat.w) = (0.253, -0.141, -0.252, 0.923)
        # timeout = 1.0
        # self.setOrientGoal(pos, quat, timeout)
        # #raw_input("Enter anything to start: ")

        #!!---- HYDER TASK 1 BASIC SPOON IN BOWL

        print "MOVES1"
        (pos.x, pos.y, pos.z) = (0.471, -0.134, -0.041)
        (quat.x, quat.y, quat.z, quat.w) = (0.573, -0.451, -0.534, 0.428)
        timeout = 10.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        print "MOVES2"
        (pos.x, pos.y, pos.z) = (0.655, 0.451, -0.046)
        (quat.x, quat.y, quat.z, quat.w) = (0.690, -0.092, -0.112, 0.709)
        timeout = 10.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        print "MOVES3"
        (pos.x, pos.y, pos.z) = (0.671, 0.623, -0.206)
        (quat.x, quat.y, quat.z, quat.w) = (0.713, 0.064, -0.229, 0.659)
        timeout = 10.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        print "MOVES4"
        (pos.x, pos.y, pos.z) = (0.749, 0.592, -0.292)
        (quat.x, quat.y, quat.z, quat.w) = (0.700, 0.108, -0.321, 0.629)
        timeout = 10.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        print "MOVES5"
        (pos.x, pos.y, pos.z) = (0.763, 0.592, -0.301)
        (quat.x, quat.y, quat.z, quat.w) = (0.706, 0.068, -0.235, 0.664)
        timeout = 10.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        print "MOVES6"
        (pos.x, pos.y, pos.z) = (0.775, 0.591, -0.288)
        (quat.x, quat.y, quat.z, quat.w) = (0.672, 0.037, -0.137, 0.727)
        timeout = 10.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        print "MOVES7"
        (pos.x, pos.y, pos.z) = (0.641, 0.560, -0.001)
        (quat.x, quat.y, quat.z, quat.w) = (0.699, -0.044, -0.085, 0.709)
        timeout = 10.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        print "MOVES7"
        (pos.x, pos.y, pos.z) = (0.742, 0.540, 0.109)
        (quat.x, quat.y, quat.z, quat.w) = (0.677, -0.058, -0.004, 0.733)
        timeout = 10.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        
        return True

    
if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    haptic_mpc_util.initialiseOptParser(p)
    opt = haptic_mpc_util.getValidInput(p)

    # Initial variables
    d_robot    = 'pr2'
    controller = 'static'
    #controller = 'actionlib'
    #arm        = 'l'


    rospy.init_node('arm_reacher')    
    ara = armReachAction(d_robot, controller, opt.arm)
    rospy.spin()
