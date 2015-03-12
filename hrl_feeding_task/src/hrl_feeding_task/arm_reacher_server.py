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
        
        print "MOVES1"
        (pos.x, pos.y, pos.z) = (1.0, 0.27, 0.005)
        (quat.x, quat.y, quat.z, quat.w) = (0.253, -0.141, -0.252, 0.923)
        timeout = 1.0
        self.setOrientGoal(pos, quat, timeout)
        ## print self.getEndeffectorPose()
        
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
        #raw_input("Enter anything to start: ")
        
        
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
