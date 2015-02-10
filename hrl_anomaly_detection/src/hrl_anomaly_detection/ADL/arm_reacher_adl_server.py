#!/usr/bin/env python  

import roslib; roslib.load_manifest('sandbox_dpark_darpa_m3')
roslib.load_manifest('hrl_anomaly_detection')
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
        self.reach_service = rospy.Service('/adl/arm_reach_enable', None_Bool, self.start_cb)


        rate = rospy.Rate(100) # 25Hz, nominally.                
        while not rospy.is_shutdown():

            if self.getJointAngles() != []:
                print "----------------------"
                print "Current joint angles"
                print self.getJointAngles()
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


        # going to home
        print "MOVE1"
        lJoint = [-1.0279779716247095, 1.1623408571610438, -0.5189701523715264, -1.886861849092819, 3.8320233539566377, -0.8064352219188305, 2.7413668578022]
        timeout = 10.0
        self.setPostureGoal(lJoint, timeout)
        
        # Front 
        print "MOVES1"
        (pos.x, pos.y, pos.z) = (0.452, -0.469, -0.215)
        (quat.x, quat.y, quat.z, quat.w) = (1.0, 0.0, 0.0, 0.0)
        timeout = 10.0
        self.setOrientGoal(pos, quat, timeout)

        print "MOVES2"
        (pos.x, pos.y, pos.z) = (0.72, -0.469, -0.215)
        (quat.x, quat.y, quat.z, quat.w) = (1.0, 0.0, 0.0, 0.0)
        timeout = 10.0
        self.setOrientGoal(pos, quat, timeout)
        
        return True

    
if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    opt, args = p.parse_args()

    # Initial variables
    d_robot    = 'pr2'
    ## controller = 'static'
    controller = 'actionlib'
    arm        = 'r'

    rospy.init_node('arm_reacher')    
    ara = armReachAction(d_robot, controller, arm)
    rospy.spin()
