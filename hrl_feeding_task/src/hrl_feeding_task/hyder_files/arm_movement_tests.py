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

	if arm == 'r':
		self.initialJointAnglesSideOfBody = [-1.570, 0, 0, -1.570, 3.141, 0, -1.570]
		self.initialJointAnglesSideFacingFoward = [-1.570, 0, 0, -1.570, 1.570, -1.570, -1.570]
	else:
		self.initialJointAnglesSideOfBody = [1.570, 0, 0, -1.570, 3.141, 0, -4.712]
		self.initialJointAnglesSideFacingFoward = [1.570, 0, 0, -1.570, 1.570, -1.570, -4.712]
	self.pos = Point()
        self.quat = Quaternion()
	self.quatOrEulerSet = False

    def start_cb(self, req):
	#Run manipulation tasks

        if self.run():
		return None_BoolResponse(True)
        else:
		return None_BoolResponse(False)

    def run(self):

	while not rospy.is_shutdown():

		calibrateJoints = raw_input("Enter 'front' or 'side' to calibrate joint angles to front or side of robot: ")
        	if calibrateJoints == 'front':
            		print "Setting initial joint angles... "
            		self.setPostureGoal(self.initialJointAnglesFrontOfBody, 7)

        	elif calibrateJoints == 'side':
            		print "Setting initial joint angles..."
            		self.setPostureGoal(self.initialJointAnglesSideOfBody, 7)

		#pos = Point()
		#quat = Quaternion()

		selection = raw_input("Which action? Type 'setQuat' or 'setEuler' or afterwards 'setPos' :")

		if selection == 'setQuat':
			try:
			    self.pos.x = float(raw_input("pos.x = "))
			    self.pos.y = float(raw_input("pos.y = "))
			    self.pos.z = float(raw_input("pos.z = "))
			    self.quat.x = float(raw_input("quat.x = "))
			    self.quat.y = float(raw_input("quat.y = "))
			    self.quat.z = float(raw_input("quat.z = "))
			    self.quat.w = float(raw_input("quat.w = "))
			    self.timeout = float(raw_input("timeout = "))

			    self.quatOrEulerSet = True

			    raw_input("Set position: [%d, %d, %d]; Set quaternions: [%d, %d, %d, %d]; Enter anything to move here" % (self.pos.x, self.pos.y, self.pos.z, self.quat.x, self.quat.y, self.quat.z, self.quat.w))

			    self.setOrientGoal(self.pos, self.quat, self.timeout)

			    raw_input("Moved to... Position: [%d, %d, %d]; Quaternions: [%d, %d, %d, %d]; Enter anything to continue" % (self.pos.x, self.pos.y, self.pos.z, self.quat.x, self.quat.y, self.quat.z))
			except:
			    print "Oops, error! Make sure you enter only numbers!"  

		if selection == 'setEuler':
			#try: 
		    self.pos.x = float(raw_input("pos.x = "))
		    self.pos.y = float(raw_input("pos.y = "))
		    self.pos.z = float(raw_input("pos.z = "))
		    self.eulerX = float(raw_input("eulerX = "))
		    self.eulerY = float(raw_input("eulerY = "))
		    self.eulerZ = float(raw_input("eulerZ = "))
		    self.eulerRads = np.radians([self.eulerX, self.eulerZ, self.eulerY]) #euler2quat needs radian values !! Z AMD Y ARE SWAPPED DUE TO OBSERVED REVERSAL WHEN TESTING CODE, FIGURE OUT AND FIX ROOT PROBLEM!! LIES SOMEWHERE IN THIS CODE BLOCK!!
		    self.quatConv = quatMath.euler2quat(self.eulerRads[2], self.eulerRads[1], self.eulerRads[0]) #input in order Z, Y, X
		    self.quat.x, self.quat.y, self.quat.z, self.quat.w = self.quatConv[0], self.quatConv[1], self.quatConv[2], self.quatConv[3]
		    self.timeout = float(raw_input("timeout = "))

		    self.quatOrEulerSet = True
		    print "Calculated quat ="
		    print self.quat
		    raw_input("Set position: [%f, %f, %f]; Set euler angles: [%f, %f, %f]; Calculated quaternion angles: [%f, %f, %f, %f]; Enter anything to move here" % (self.pos.x, self.pos.y, self.pos.z, self.eulerX, self.eulerY, self.eulerZ, self.quat.x, self.quat.y, self.quat.z, self.quat.w))

		    self.setOrientGoal(self.pos, self.quat, self.timeout)
		    raw_input("MOved to... Position: [%f, %f, %f]; Euler angles: [%f, %f, %f]; Quaternion angles: [%f, %f, %f, %f]; Enter anything to continue" % (self.pos.x, self.pos.y, self.pos.z, self.eulerX, self.eulerY, self.eulerZ, self.quat.x, self.quat.y, self.quat.z, self.quat.w))
			#except:
				#print "Oops, error! Make sure you enter only numbers!"  
		if selection == 'setPos':
			#try:
		    if not self.quatOrEulerSet:
			raw_input("Set quaternions or euler angles first, press Enter to continue:")
		    if self.quatOrEulerSet:
			self.pos.x = float(raw_input("pos.x = "))
			self.pos.y = float(raw_input("pos.y = "))
			self.pos.z = float(raw_input("pos.z = "))
			self.timeout = float(raw_input("timeout = "))

			raw_input("Set new position: [%d, %d, %d]; Set previous quaternions: [%d, %d, %d, %d]; Enter anything to move here" % ( self.pos.x, self.pos.y, self.pos.z, self.quat.x, self.quat.y, self.quat.z, self.quat.w))

			self.setOrientGoal(self.pos, self.quat, self.timeout)

			raw_input("Moved to... New position: [%d, %d, %d]; Previous quaternions: [%d, %d, %d, %d]; Enter anything to continue" % (self.pos.x, self.pos.y, self.pos.z, self.quat.x, self.quat.y, self.quat.z, self.quat.w))
			#except:
				#print "Oops, error! Make sure you enter only numbers!"

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
   # ara.run()
    rospy.spin()
