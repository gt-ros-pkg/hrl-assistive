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
from hrl_feeding_task.srv import PosQuatTimeoutSrv, AnglesTimeoutSrv
import hrl_lib.quaternion as quatMath #Used for quaternion math :)
from std_msgs.msg import String
from pr2_controllers_msgs.msg import JointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import hrl_haptic_manipulation_in_clutter_msgs.msg as haptic_msgs

class rightArmControl(mpcBaseAction):
    def __init__(self, d_robot, controller, arm): #removed arm= 'l' so I can use right arm as well as an option

        mpcBaseAction.__init__(self, d_robot, controller, arm)
        #Allows another node (client) to request specific actions corresponding to the right arm mpcBaseAction instance...
        try:
		self.setOrientRightGoalService = rospy.Service('/setOrientGoalRightService', PosQuatTimeoutSrv, self.setOrientGoalRight)

        	self.setStopRightService = rospy.Service('/setStopRightService', None_Bool, self.setStopRight)

        	self.setPostureGoalRightService = rospy.Service('/setPostureGoalRightService', AnglesTimeoutSrv, self.setPostureGoalRight)
		print "Right arm services started!"
	except:
		print "Right arm services NOT started!"

    def setOrientGoalRight(self, msg):
        try:
            self.setOrientGoal(msg.position, msg.orientation, msg.timeout)
            outputString = "Right arm pos: " + msg.position + "\n" + "Right arm quat: " + msg.orientation + "\n"
            print outputString
	    return "Worked"
        except:
            print "Could not set right arm end effector orientation"
	    return "Didn't work"

    def setStopRight(self, msg):
        try:
            self.setStop()
            print "Stopped right arm"
	    return "Worked"
        except:
            print "Could not stop right arm"
	    return "Didn't work"

    def setPostureGoalRight(self, msg):
        try:
            self.setPostureGoal(msg.angles, msg.timeout)
            outputString = "Set right arm joint angles: " + msg.angles + "\n"
            print outputString
	    return "Worked"
        except:
	    print "Could not set right arm joint angles"
	    return "Didn't work"

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

    arm = 'r'

    rospy.init_node('arm_reacher_helper')
    ara2 = rightArmControl(d_robot, controller, arm)
    rospy.spin()
