#!/usr/bin/env python

import roslib; roslib.load_manifest('sandbox_dpark_darpa_m3')
roslib.load_manifest('hrl_summer_2015')
import rospy
import numpy as np, math
import time
import tf

import hrl_haptic_mpc.haptic_mpc_util as haptic_mpc_util

from hrl_srvs.srv import None_Bool, None_BoolResponse
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from sandbox_dpark_darpa_m3.lib.hrl_mpc_base import mpcBaseAction
from hrl_feeding_task.srv import PosQuatTimeoutSrv, AnglesTimeoutSrv
from hrl_summer_2015.src.record_data import robot_kinematics as kinematicsRecord
from hrl_summer_2015.src.record_data import tool_audio as audioRecord
from hrl_summer_2015.src.record_data import tool_ft as ftRecord

import hrl_lib.quaternion as quatMath
from std_msgs.msg import String
from pr2_controllers_msgs.msg import JointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class leftArmDoor(mpcBaseAction):

        def __init__(self, d_robot, controller, arm): #removed arm= 'l' so I can use right arm as well as an option
            mpcBaseAction.__init__(self, d_robot, controller, arm)
            self.reach_service = rospy.Service('/arm_reach_enable', None_Bool, self.start_cb)

            handlePos = np.array[[0, 0, 0]]
            framePos = np.array[[0, 0, 0]]
            actionPositions = np.array[[0, 0, 0]]
            actionEulers = np.array[[90, 0, 0]]
            actionQuats = euler2quatArray(actionEulers)

    def start_cb(self, req):

        # Run manipulation tasks
        if self.run():
            return None_BoolResponse(True)
        else:
            return None_BoolResponse(False)

    def run(self):
        posL = Point()
        posR = Point()
        quatL = Quaternion()
        quatR = Quaternion()

        posL.x, posL.y, posL.z = framePos[0][0], framePos[0][1], framePos[0][2], framePos[0][3]
        quatR.x, quatR.y, quatR.z, quatR.w = actionQuats[0][0], actionQuats[0][1], actionQuats[0][2], actionQuats[0][3]
        timeout = 10
        input = raw_input("Press 'y' to confirm that robot is grasping door handle")
        if input == 'y':
            self.setOrientGoal(posL, quatL, timeout)
            raw_input("Press Enter to confirm motion")



    def euler2quatArray(self, eulersIn): #converts an array of euler angles (in degrees) to array of quaternions
        (rows, cols) = np.shape(eulersIn)
        quatArray = np.zeros((rows, cols+1))
        for r in xrange(0, rows):
            rads = np.radians([eulersIn[r][0], eulersIn[r][2], eulersIn[r][1]]) #CHECK THIS ORDER!!!
            quats = quatMath.euler2quat(rads[2], rads[1], rads[0])
            quatArray[r][0], quatArray[r][1], quatArray[r][2], quatArray[r][3] = quats[0], quats[1], quats[2], quats[3]

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

    #try:
        #arm = opt.arm1 #added/changed due to new launch file controlling both arms (arm1, arm2)
    #except:
        #arm = opt.arm

    arm = 'l'

    rospy.init_node('arm_reacher_door')
    ara = armReachAction(d_robot, controller, arm)
    rospy.spin()
