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
        lJoint = [-1.0309357552025427, 1.1668491091040116, -0.5191943954490865, -1.8896653025274457, 3.8369242574394975, -0.8069487133997452, 2.7471289084492376]
        timeout = 10.0
        self.setPostureGoal(lJoint, timeout)

        # Front 
        print "MOVES1"
        (pos.x, pos.y, pos.z) = (0.552, -0.469, -0.215)
        (quat.x, quat.y, quat.z, quat.w) = (1.0, 0.0, 0.0, 0.0)
        timeout = 10.0
        self.setOrientGoal(pos, quat, timeout)

        print "MOVES2"
        (pos.x, pos.y, pos.z) = (0.82, -0.469, -0.215)
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
