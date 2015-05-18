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

class armMovements(mpcBaseAction):
    def __init__(self, d_robot, controller, arm): #removed arm= 'l' so I can use right arm as well as an option

        mpcBaseAction.__init__(self, d_robot, controller, arm)

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
	#Run manipulation tasks

        if self.run():
		return None_BoolResponse(True)
        else:
		return None_BoolResponse(False)

    def run(self):

        pos = Point()
        quat = Quaternion()

        quatOrEulerSet = False

	selection = raw_input("Which action? Type 'setQuat' or 'setEuler' or afterwards 'setPos' :")

            if selection == 'setQuat':
                pos.x = raw_input("pos.x = :")
                pos.y = raw_input("pos.y = :")
                pos.z = raw_input("pos.z = :")
                quat.x = raw_input("quat.x = :")
                quat.y = raw_input("quat.y = :")
                quat.z = raw_input("quat.z = :")
                quat.w = raw_input("quat.w = :")
                timeout = raw_input("timeout = :")

                quatOrEulerSet = True

                raw_input("Set position: [%d]; Set quaternions: [%d]; Enter anything to move here" % pos, quat)

                self.setOrientGoal(pos, quat, timeout)

                raw_input("Moved to... Position: [%d]; Quaternions: [%d]; Enter anything to continue" % pos, quat)

            if selection == 'setEuler':
                pos.x = raw_input("pos.x = :")
                pos.y = raw_input("pos.y = :")
                pos.z = raw_input("pos.z = :")
                roll = raw_input("roll = :")
                pitch = raw_input("pitch = :")
                yaw = raw_input("yaw = :")
                quat = quatMath.euler2quat(roll, pitch, yaw)
                timeout = raw_input("timeout = :")

                quatOrEulerSet = True

                raw_input("Set position: [%d]; Set euler angles: [%d, %d, %d]; Calculated quaternion angles: [%d]; Enter anything to move here" % pos, roll, pitch, yaw, quat)

                self.setOrientGoal(pos, quat, timeout)

                raw_input("Moved to... Position: [%d]; Euler angles: [%d, %d, %d]; Quaternion angles: %d; Enter anything to continue" % pos, roll, pitch, yaw, quat)

            if selection == 'setPos':
                if not quatOrEulerSet:
                    raw_input("Set quaternions or euler angles first, press Enter to continue:")
                    return
                if quatOrEulerSet:
                    pos.x = raw_input("pos.x = :")
                    pos.y = raw_input("pos.y = :")
                    pos.z = raw_input("pos.z = :")
                    timeout = raw_input("timeout = :")

                    raw_input("Set new position: [%d]; Set previous quaternions: [%d]; Enter anything to move here" % pos, quat)

                    self.setOrientGoal(pos, quat, timeout)

                    raw_input("Moved to... New position: [%d]; Previous quaternions: [%d]; Enter anything to continue" % pos, quat)

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

    rospy.init_node('arm_movement_tests')
    ara = armMovements(d_robot, controller, opt.arm)
    rospy.spin()
