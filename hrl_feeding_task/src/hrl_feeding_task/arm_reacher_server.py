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

        print "MOVES1"
        (pos.x, pos.y, pos.z) = (1.0, 0.27, 0.005)
        (quat.x, quat.y, quat.z, quat.w) = (0.253, -0.141, -0.252, 0.923)
        timeout = 1.0
        self.setOrientGoal(pos, quat, timeout)
        print self.getEndeffectorPose()
        
        print "MOVES2"
        (pos.x, pos.y, pos.z) = (1.0, 0.27, 0.005-0.2)
        (quat.x, quat.y, quat.z, quat.w) = (0.253, -0.141, -0.252, 0.923)
        timeout = 1.0 
        self.setOrientGoal(pos, quat, timeout)

        print "MOVES3"
        (pos.x, pos.y, pos.z) = (1.0, 0.27, 0.005)
        (quat.x, quat.y, quat.z, quat.w) = (0.253, -0.141, -0.252, 0.923)
        timeout = 1.0
        self.setOrientGoal(pos, quat, timeout)
        raw_input("Enter anything to start: ")

        

        #!!---- HYDER TASK 1 BASIC SPOON IN BOWL

        # print "MOVES0"
        # (pos.x, pos.y, pos.z) = (0.586, 0.165, 0.175)
        # (quat.x, quat.y, quat.z, quat.w) = (-0.591, -0.378, -0.385, 0.600)
        # timeout = 1.0
        # self.setOrientGoal(pos, quat, timeout)
        # raw_input("Enter anything to start: ")

        # print "MOVES1"
        # (pos.x, pos.y, pos.z) = (0.450, 0.725, 0.212)
        # (quat.x, quat.y, quat.z, quat.w) = (-0.508, -0.453, -0.113, 0.723)
        # timeout = 1.0
        # self.setOrientGoal(pos, quat, timeout)
        # raw_input("Enter anything to start: ")


        # print "MOVES2"
        # (pos.x, pos.y, pos.z) = (0.702, 0.694, -0.111)
        # (quat.x, quat.y, quat.z, quat.w) = (-0.747, 0.003, 0.019, 0.665)
        # timeout = 1.0
        # self.setOrientGoal(pos, quat, timeout)
        # raw_input("Enter anything to start: ")

        # print "MOVES3"
        # (pos.x, pos.y, pos.z) = (0.805, 0.711, -0.249)
        # (quat.x, quat.y, quat.z, quat.w) = (-0.724, 0.070, 0.121, 0.676)
        # timeout = 1.0
        # self.setOrientGoal(pos, quat, timeout)
        # raw_input("Enter anything to start: ")

        # print "MOVES4"
        # (pos.x, pos.y, pos.z) = (0.747, 0.719, 0.050)
        # (quat.x, quat.y, quat.z, quat.w) = (-0.734, -0.052, -0.018, 0.677)
        # timeout = 1.0
        # self.setOrientGoal(pos, quat, timeout)
        # raw_input("Enter anything to start: ")

        # print "MOVES5"
        # (pos.x, pos.y, pos.z) = (0.591, 0.206, 0.208)
        # (quat.x, quat.y, quat.z, quat.w) = (-0.674, 0.207, -0.304, 0.641)
        # timeout = 1.0
        # self.setOrientGoal(pos, quat, timeout)
        # raw_input("Enter anything to start: ")

        # print "MOVES6"
        # (pos.x, pos.y, pos.z) = (0.790, -0.292, 0.059)
        # (quat.x, quat.y, quat.z, quat.w) = (-0.617, 0.343, -0.409, 0.579)
        # timeout = 1.0
        # self.setOrientGoal(pos, quat, timeout)
        # raw_input("Enter anything to start: ")



        

        
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
