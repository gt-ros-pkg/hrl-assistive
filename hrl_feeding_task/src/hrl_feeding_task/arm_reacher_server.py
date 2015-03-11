#!/usr/bin/env python  

import roslib; roslib.load_manifest('sandbox_dpark_darpa_m3')
roslib.load_manifest('hrl_feeding_task')
import rospy
import numpy as np, math
import time

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
        
        #--------------------------------------------------------------        
        # Microwave or cabinent closing
        #--------------------------------------------------------------        
        ## print "MOVES1"
        ## (pos.x, pos.y, pos.z) = (0.56, -0.59, -0.24)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.0, 0.0, 0.0, 1.0)
        ## timeout = 2.0
        ## self.setOrientGoal(pos, quat, timeout)
        
        ## print "MOVES2"
        ## (pos.x, pos.y, pos.z) = (0.67, -0.63, -0.24)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.0, 0.0, 0.0, 1.0)
        ## timeout = 1.0
        ## self.setOrientGoal(pos, quat, timeout)
        ## ## raw_input("Enter anything to start: ")

        ## print "MOVES3"
        ## (pos.x, pos.y, pos.z) = (0.58, -0.56, -0.24)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.0, 0.0, 0.0, 1.0)
        ## timeout = 2.0
        ## self.setOrientGoal(pos, quat, timeout)

        ## # Front 
        ## print "MOVES4"
        ## (pos.x, pos.y, pos.z) = (0.56, -0.59, -0.24)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.0, 0.0, 0.0, 1.0)
        ## timeout = 3.0
        ## self.setOrientGoal(pos, quat, timeout)

        ## print "MOVES1"
        ## (pos.x, pos.y, pos.z) = (0.39-0.15, -0.65, 0.05)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.0, 0.0, 0.0, 1.0)
        ## timeout = 1.0
        ## self.setOrientGoal(pos, quat, timeout)
        
        ## print "MOVES2"
        ## (pos.x, pos.y, pos.z) = (0.525-0.15, -0.65, 0.05)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.0, 0.0, 0.0, 1.0)
        ## timeout = 1.0
        ## self.setOrientGoal(pos, quat, timeout)
        ## ## raw_input("Enter anything to start: ")

        ## print "MOVES3"
        ## (pos.x, pos.y, pos.z) = (0.39-0.15, -0.65, 0.05)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.0, 0.0, 0.0, 1.0)
        ## timeout = 2.0
        ## self.setOrientGoal(pos, quat, timeout)


        
        ## #--------------------------------------------------------------        
        ## # Staples
        ## position_step_scaling_radius: 0.05 #0.25 #meters. default=0.25
        ## goal_velocity_for_hand: 0.60 #meters/sec. default=0.5    
        ## #--------------------------------------------------------------        
        ## # Up
        ## print "MOVES1"
        ## (pos.x, pos.y, pos.z) = (0.32, -0.63, -0.29)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.0, 0.0, 0.0, 1.0)
        ## timeout = 1.5
        ## self.setOrientGoal(pos, quat, timeout)
        
        ## print "MOVES2"
        ## (pos.x, pos.y, pos.z) = (0.35+0.07, -0.63, -0.29)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.0, 0.0, 0.0, 1.0)
        ## timeout = 2.0
        ## self.setOrientGoal(pos, quat, timeout)

        ## ## # Front 
        ## print "MOVES3"
        ## (pos.x, pos.y, pos.z) = (0.35, -0.63, -0.29)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.0, 0.0, 0.0, 1.0)
        ## timeout = 1.0
        ## self.setOrientGoal(pos, quat, timeout)
        

        ## #--------------------------------------------------------------        
        ## # key
        ## #--------------------------------------------------------------        
        ## # Up
        ## print "MOVES1"
        ## (pos.x, pos.y, pos.z) = (0.38, -0.59, -0.44)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.745, -0.034, -0.666, -0.011)
        ## timeout = 1.0
        ## self.setOrientGoal(pos, quat, timeout)
        
        ## print "MOVES2"
        ## (pos.x, pos.y, pos.z) = (0.38, -0.59, -0.467)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.745, -0.034, -0.666, -0.011)
        ## timeout = 0.3
        ## self.setOrientGoal(pos, quat, timeout)

        ## print "MOVES3"
        ## (pos.x, pos.y, pos.z) = (0.38, -0.59, -0.44)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.745, -0.034, -0.666, -0.011)
        ## timeout = 0.3
        ## self.setOrientGoal(pos, quat, timeout)

        ## #--------------------------------------------------------------        
        ## # case
        ## #--------------------------------------------------------------        
        ## # Up
        ## print "MOVES1"
        ## (pos.x, pos.y, pos.z) = (0.42, -0.48, -0.35)
        ## (quat.x, quat.y, quat.z, quat.w) = (-0.146, 0.675, 0.172, 0.702)
        ## timeout = 4.0
        ## self.setOrientGoal(pos, quat, timeout)
        ## ## print self.getEndeffectorPose()
        ## ## raw_input("Enter anything to start: ")
        
        ## print "MOVES2"
        ## (pos.x, pos.y, pos.z) = (0.42, -0.48, -0.45)
        ## (quat.x, quat.y, quat.z, quat.w) = (-0.146, 0.675, 0.172, 0.702)
        ## timeout = 1.0 
        ## self.setOrientGoal(pos, quat, timeout)

        ## print "MOVES3"
        ## (pos.x, pos.y, pos.z) = (0.44, -0.48, -0.33)
        ## (quat.x, quat.y, quat.z, quat.w) = (-0.146, 0.675, 0.172, 0.702)
        ## timeout = 3.0
        ## self.setOrientGoal(pos, quat, timeout)
        
        ## #--------------------------------------------------------------        
        ## # Switch
        ## position_step_scaling_radius: 0.05 #0.25 #meters. default=0.25
        ## goal_velocity_for_hand: 0.60 #meters/sec. default=0.5    
        ## #--------------------------------------------------------------        
        ## print "MOVES1"
        ## (pos.x, pos.y, pos.z) = (0.38, -0.59, -0.42)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.745, -0.034, -0.666, -0.011)
        ## timeout = 1.0
        ## self.setOrientGoal(pos, quat, timeout)
        ## ## print self.getEndeffectorPose()
        ## ## raw_input("Enter anything to start: ")
        
        ## print "MOVES2"
        ## (pos.x, pos.y, pos.z) = (0.38, -0.59, -0.445)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.745, -0.034, -0.666, -0.011)
        ## timeout = 1.0 
        ## self.setOrientGoal(pos, quat, timeout)

        ## print "MOVES3"
        ## (pos.x, pos.y, pos.z) = (0.38, -0.59, -0.42)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.745, -0.034, -0.666, -0.011)
        ## timeout = 1.0
        ## self.setOrientGoal(pos, quat, timeout)

        ## #--------------------------------------------------------------        
        ## # Switch -wall
        ## position_step_scaling_radius: 0.05 #0.25 #meters. default=0.25
        ## goal_velocity_for_hand: 0.60 #meters/sec. default=0.5    
        ## #--------------------------------------------------------------        
        print "MOVES1"
        (pos.x, pos.y, pos.z) = (0.63-0.15, -0.52, 0.19-0.3)
        (quat.x, quat.y, quat.z, quat.w) = (0.849, -0.019, 0.526, 0.026)
        timeout = 1.0
        self.setOrientGoal(pos, quat, timeout)
        ## print self.getEndeffectorPose()
        raw_input("Enter anything to start: ")
        
        print "MOVES2"
        (pos.x, pos.y, pos.z) = (0.63-0.15, -0.52, 0.225-0.3)
        (quat.x, quat.y, quat.z, quat.w) = (0.849, -0.019, 0.526, 0.026)
        timeout = 1.0 
        self.setOrientGoal(pos, quat, timeout)

        print "MOVES3"
        (pos.x, pos.y, pos.z) = (0.63-0.15, -0.52, 0.19-0.3)
        (quat.x, quat.y, quat.z, quat.w) = (0.849, -0.019, 0.526, 0.026)
        timeout = 1.0
        self.setOrientGoal(pos, quat, timeout)
        
        ## #--------------------------------------------------------------        
        ## # Toaster
        ## position_step_scaling_radius: 0.05 #0.25 #meters. default=0.25
        ## goal_velocity_for_hand: 0.60 #meters/sec. default=0.5    
        ## #--------------------------------------------------------------        
        ## print "MOVES1"
        ## (pos.x, pos.y, pos.z) = (0.3, -0.59, -0.36)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.745, -0.034, -0.666, -0.011)
        ## timeout = 1.0
        ## self.setOrientGoal(pos, quat, timeout)
        
        ## print "MOVES2"
        ## (pos.x, pos.y, pos.z) = (0.3, -0.59, -0.44)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.745, -0.034, -0.666, -0.011)
        ## timeout = 1.0 
        ## self.setOrientGoal(pos, quat, timeout)

        ## print "MOVES3"
        ## (pos.x, pos.y, pos.z) = (0.3, -0.59, -0.36)
        ## (quat.x, quat.y, quat.z, quat.w) = (0.745, -0.034, -0.666, -0.011)
        ## timeout = 1.0
        ## self.setOrientGoal(pos, quat, timeout)

        
        return True

    
if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    opt, args = p.parse_args()

    # Initial variables
    d_robot    = 'pr2'
    controller = 'static'
    #controller = 'actionlib'
    arm        = 'r'

    rospy.init_node('arm_reacher')    
    ara = armReachAction(d_robot, controller, arm)
    rospy.spin()
