#!/usr/bin/env python  

import roslib; roslib.load_manifest('sandbox_dpark_darpa_m3')
roslib.load_manifest('hrl_base_selection')
import rospy
import numpy as np, math
import time

from hrl_srvs.srv import None_Bool, None_BoolResponse
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sandbox_dpark_darpa_m3.lib.hrl_mpc_base import mpcBaseAction


class armReachAction(mpcBaseAction):
    def __init__(self, d_robot, controller, arm='l'):

        mpcBaseAction.__init__(self, d_robot, controller, arm)
        
        # service for ari's request
        self.reach_service = rospy.Service('/base_selection/arm_reach_enable', None_Bool, self.start_cb)

        
    def start_cb(self, req):

        # Run manipulation tasks
        if self.run():
            return None_BoolResponse(True)
        else:
            return None_BoolResponse(False)

    def run(self):

        self.setOrientationControl()

        pos  = Point()
        quat = Quaternion()

        #going to home with arm curled high near left shoulder:
        (pos.x, pos.y, pos.z) = (0.301033944729, 0.461276517595, 0.196885866571)
        (quat.x, quat.y, quat.z, quat.w) = (0.553557277528, 0.336724229346, -0.075691681684, 0.757932650828)
        timeout = 20.0
        self.setOrientGoal(pos, quat, timeout)

        #moving to high in front of chest, pointing down:
        (pos.x, pos.y, pos.z) = (0.377839595079, 0.11569018662, 0.0419789999723)
        (quat.x, quat.y, quat.z, quat.w) = (0.66106069088, 0.337429642677, -0.519856214523, 0.422953367233)
        timeout = 20.0
        self.setOrientGoal(pos, quat, timeout)

        # #going to home location in front of camera:
        # (pos.x, pos.y, pos.z) = (0.5309877259429142, 0.4976163448816489, 0.16719537682372823)
        # (quat.x, quat.y, quat.z, quat.w) = (0.7765742993649133, -0.37100605554316285, -0.27784851903166524, 0.42671660945891)
        # timeout = 35.0
        # self.setOrientGoal(pos, quat, timeout)
        #
        # #moving vertically to over bowl:
        # (pos.x, pos.y, pos.z) = (0.516341299985487, 0.8915608293219441, 0.1950343868326016)
        # (quat.x, quat.y, quat.z, quat.w) = (0.6567058177198967, 0.16434420640210323, 0.0942917725129517, 0.7299571990406495)
        # timeout = 35.0
        # self.setOrientGoal(pos, quat, timeout)

        # These are the goals for autobed's data
        frame_id = '/head_frame'
        # #going to in front of subjects face:        
        (pos.x, pos.y, pos.z) = (0.2741387011303321, 0.05522571699560719, -0.011919598309888757)
        (quat.x, quat.y, quat.z, quat.w) = (-0.023580897114171894, 0.7483633417869068, 0.662774596931439, 0.011228696415565394)
        timeout = 10.0
        self.setOrientGoal(pos, quat, timeout, frame_id)

        # #going to subjects mouth:
        (pos.x, pos.y, pos.z) = (0.20608632401364894, 0.03540318703608347, 0.00607600258150498)
        (quat.x, quat.y, quat.z, quat.w) = (-0.015224467044577382, 0.7345761465214938, 0.6783020152473445, -0.008513323454022942)
        timeout = 35.0
        self.setOrientGoal(pos, quat, timeout, frame_id)

        # These are the goals for wheelchair's new data (10/13/14)
        # frame_id = '/head_frame'
        # #going to in front of subjects face:        
        # (pos.x, pos.y, pos.z) = (0.2741387011303321, 0.005522571699560719, -0.011919598309888757)
        # (quat.x, quat.y, quat.z, quat.w) = (-0.023580897114171894, 0.7483633417869068, 0.662774596931439, 0.011228696415565394)
        # timeout = 10.0
        # self.setOrientGoal(pos, quat, timeout, frame_id)

        # #going to subjects mouth:
        # (pos.x, pos.y, pos.z) = (0.13608632401364894, 0.003540318703608347, 0.00607600258150498)
        # (quat.x, quat.y, quat.z, quat.w) = (-0.015224467044577382, 0.7345761465214938, 0.6783020152473445, -0.008513323454022942)
        # timeout = 35.0
        # self.setOrientGoal(pos, quat, timeout, frame_id)
        
        return True
        
if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    opt, args = p.parse_args()

    # Initial variables
    d_robot    = 'pr2'
    controller = 'static'
    arm        = 'l'

    rospy.init_node('arm_reacher')    
    ara = armReachAction(d_robot, controller, arm)
    rospy.spin()
