#!/usr/bin/env python

import roslib; roslib.load_manifest('sandbox_dpark_darpa_m3')
roslib.load_manifest('hrl_feeding_task')
import rospy
import numpy as np, math
import time
import tf

import hrl_haptic_mpc.haptic_mpc_util as haptic_mpc_util

from hrl_srvs.srv import None_Bool, None_BoolResponse
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from sandbox_dpark_darpa_m3.lib.hrl_mpc_base import mpcBaseAction
import hrl_lib.quaternion as quatMath #Used for quaternion math :)
from std_msgs.msg import String
from pr2_controllers_msgs.msg import JointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import hrl_haptic_manipulation_in_clutter_msgs.msg as haptic_msgs

class rightArmControl(mpcBaseAction):
    def __init__(self, d_robot, controller, arm): #removed arm= 'l' so I can use right arm as well as an option

        mpcBaseAction.__init__(self, d_robot, controller, arm)

    def setOrientRight(self, pos, quat, timeout):
        try:
            self.setOrientGoal(pos, quat, timeout)
            outputString = "Right arm pos: " + pos + "\n" + "Right arm quat: " + quat + "\n"
            return outputString
        except:
            return "Could not set right arm end effector orientation"

    def setStopRight(self):
        try:
            self.setStop()
            return "Stopped right arm"
        except:
            return "Could not stop right arm"

    def setPostureGoalRight(self, angles, timeout):
        try:
            self.setPostureGoal(angles, timeout)
            return "Set right arm joint angles:"
            return self.getJointAngles()
        except:
            return "Could not set right arm joint angles"

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    haptic_mpc_util.initialiseOptParser(p)
    opt = haptic_mpc_util.getValidInput(p)

    # Initial variables
    d_robot    = 'pr2'
    controller = 'static'
    #controller = 'actionlib'
    #arm        = 'r'
    try:
        arm = opt.arm2
    except:
        arm = opt.arm

    rospy.init_node('arm_reacher_helper')
    ara = rightArmControl(d_robot, controller, arm)
    rospy.spin()
