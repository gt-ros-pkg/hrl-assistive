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
from std_msgs.msg import String
from pr2_controllers_msgs.msg import JointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import hrl_haptic_manipulation_in_clutter_msgs.msg as haptic_msgs

class armReachAction(mpcBaseAction):
    def __init__(self, d_robot, controller, arm = 'l'):

        mpcBaseAction.__init__(self, d_robot, controller, arm)

        #Subscribers to publishers of bowl location data
        rospy.Subscriber('bowl_location', PoseStamped, self.bowlPoseCallback)
        rospy.Subscriber('RYDS_CupLocation', PoseStamped, self.bowlPoseKinectCallback)

        rospy.Subscriber('emergency_arm_stop', String, self.reverseMotion)

        #Be able to read the current state of the robot, used for setting new joint angles for init
        rospy.Subscriber('haptic_mpc/robot_state', haptic_msgs.RobotHapticState, self.robotStateCallback)

        #Be able to publish new data to the robot state, especially new joint angles
        #self.robotState_pub = rospy.Publisher('haptic_mpc/robot_state', haptic_msgs.RobotHapticState)

        #self.tfListener = tf.TransformListener()
        #self.tfBroadcaster = tf.TransformBroadcaster()

        # service request
        self.reach_service = rospy.Service('/arm_reach_enable', None_Bool, self.start_cb)

        #VARIABLES!
        self.jointAnglesFront = list([0.02204231193041639, 0.72850849623194, 0.08302486916827911, -2.051374187142846, -3.1557218713638484, 0, 45])
        self.jointAnglesSide = list([1.5607891300760723, -0.019056839242125957, 0.08462841743197802, -1.5716040496178354, 3.0047615005230432, -0.09718467633646749, -1.5831090362171292])
        self.jointAnglesSideForward = list([1.4861731744547848, -0.18900803975897545, -0.0010010598495409084, -1.2796015247572599, 1.625224170926076, -1.5317317839135611, -1.481043325223495])
        #[0.02204231193041639, 0.72850849623194, 0.08302486916827911, -2.051374187142846, -3.1557218713638484, -1.2799710435005978, 10.952306846165152]
        #These joint angles correspond to 'Pose 1' - See Keep note

        #!self.previousGoals = JointTrajectory()
        #NEED TO APPEND A JointTrajectoryPoint to the end!!! FIX THIS!!! NOT FIXED AS OF FRIDAY MARCH 27, 2015
        #!self.point = JointTrajectoryPoint() #ATTEMPT AT FIRST FIX!


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

        rospy.spin()

    def start_cb(self, req):

        # Run manipulation tasks
        if self.run():
            return None_BoolResponse(True)
        else:
            return None_BoolResponse(False)

    def bowlPoseCallback(self, data):
        self.bowl_frame = data.header.frame_id
        self.bowl_pos = np.matrix([ [data.pose.position.x], [data.pose.position.y], [data.pose.position.z] ])
        self.bowl_quat = np.matrix([ [data.pose.orientation.x], [data.pose.orientation.y], [data.pose.orientation.z], [data.pose.orientation.w] ])

        #rospy.loginfo(self.bowl_pos)
        #rospy.loginfo(self.bowl_quat)


    def bowlPoseKinectCallback(self, data):
        #Takes in a PointStamped() type message, contains Header() and Pose(), from Kinect bowl location publisher
        self.bowl_frame = data.header.frame_id
        self.bowl_pos = np.matrix([ [data.pose.position.x], [data.pose.position.y], [data.pose.position.z] ])
        self.bowl_quat = np.matrix([ [data.pose.orientation.x], [data.pose.orientation.y], [data.pose.orientation.z], [data.pose.orientation.w] ])

	print 'Bowl Pos: '
	print self.bowl_pos
	print 'Bowl Quaternions: '
	print self.bowl_quat

    def robotStateCallback(self, data): #Read from RobotState topic and saves the current data, including joint angles, used to create new message containing initialized joint angles to publish to RobotState topic
        self.robotData = data
        self.joint_angles_raw = data.joint_angles
        #rospy.loginfo("Raw Robot State Data: ")
        #rospy.loginfo(self.robotData)
        #rospy.loginfo("Raw Robot Joint Angles: ")
        #rospy.loginfo(self.joint_angles_raw)
        #rospy.sleep(1)

    def run(self):

        pos  = Point()
        quat = Quaternion()

        confirm = False
        while not confirm:
            print "Current pose"
            print self.getEndeffectorPose()
            ans = raw_input("Enter y to confirm to start: ")
            if ans == 'y':
                confirm = True
            print self.getJointAngles()

        #Variables...!
        self.iteration = 0

        # self.previousPoints = [] #for example... 7 previous points
        # self.previousPoints.angles = list()

        #^NOTE, AS OF MARCH 26, 2015 THIS DOESN'T WORK AS YOU CAN'T ARBITRARILY ADD ATTRIBUTE TO MADE-UP CLASSES!
        #I NEED A BETTER WAY TO STOE THIS INFORMATION! MAYBE A CUSTOM MESSAGE?
        #FIX THIS!!!

        #self.getJointPlan()
        calibrateJoints = raw_input("Enter 'front' or 'side' to calibrate joint angles to front or side of robot: ")
        if calibrateJoints == 'front':
            print "Setting initial joint angles... "
            self.setPostureGoal(self.jointAnglesFront, 5)

        elif calibrateJoints == 'side':
            print "Setting initial joint angles..."
            self.setPostureGoal(self.jointAnglesSideForward, 5)

            #!!---- BASIC SCOOPING MOTION WITH BOWL POSITION OFFSET
    	#Flat Gripper Orientation Values:
    	#(0.642, 0.150, 0.154, 0.736)

	    #raw_input("Enter anything to go to initial position: ")
	    #print "MOVE0, Initial Position"
        #(pos.x, pos.y, pos.z) = (0.803, 0.182, 0.067)
        #(quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)
        #timeout = 1.0
        #self.setOrientGoal(pos, quat, timeout)
        raw_input("Iteration # %d. Enter anything to start: " % self.iteration)

        #---------------------------------------------------------------------------------------#

    	kinectPose = raw_input('Press k in order to position spoon to Kinect-provided bowl position, used for testing: ')
    	if kinectPose == 'k':
    		#THIS IS SOME TEST CODE, BASICALLY PUTS THE SPOON WHERE THE KINECT THINKS THE BOWL IS, USED TO COMPARE ACTUAL BOWL POSITION WITH KINECT-PROVIDED BOWL POSITION!! UNCOMMENT ALL THIS OUT IF NOT USED MUCH!!#
    		print "MOVES_KINECT_BOWL_POSITION"
            	(pos.x, pos.y, pos.z) = (self.bowl_pos[0], self.bowl_pos[1], self.bowl_pos[2])
            	(quat.x, quat.y, quat.z, quat.w) = (0, 0, 0, 1)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) # (0.573, -0.451, -0.534, 0.428)
            	timeout = 7
            	#self.setPositionGoal(pos, quat, timeout)
            	self.setOrientGoal(pos, quat, timeout)
    		raw_input('Press Enter to continue: ' )

	print "MOVES1 - Moving over bowl... "
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + -0.09743569, self.bowl_pos[1] + -0.11179373, self.bowl_pos[2] + 0.18600000)
        (quat.x, quat.y, quat.z, quat.w) = (0.580, 0.333, 0.050, 0.742)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) # (0.573, -0.451, -0.534, 0.428)
        timeout = 4
        #self.setPositionGoal(pos, quat, timeout)
        self.setOrientGoal(pos, quat, timeout)

        # #Code for storing current joint angles in case of playback...
        # self.currentAngles = self.getJointAngles()
        # print "Current Angles:"
        # print self.currentAngles
        # self.point.positions = self.currentAngles
        # self.previousGoals.points.append(self.point)
        # print "EVERYTHING:"
        # print self.previousGoals
        # print "resized Points:"
        # print self.previousGoals.points[self.iteration]
        #
        # print "Stored joint angles: "
        # print self.previousGoals.points[self.iteration].positions[2]
        raw_input("Iteration # %d. Enter anything to continue: " % self.iteration)
        # self.iteration += 1

        #---------------------------------------------------------------------------------------#

	print "MOVES2 - Pointing down into bottom of bowl..."
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + -0.03143569, self.bowl_pos[1] + -0.09879372, self.bowl_pos[2] + 0.02800000)
        (quat.x, quat.y, quat.z, quat.w) = (0.484, 0.487, -0.164, 0.708)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.690, -0.092, -0.112, 0.709)
        timeout = 4
        #self.setPositionGoal(pos, quat, timeout)
        self.setOrientGoal(pos, quat, timeout)

        # #Code for storing current joint angles in case of playback...
        # currentAngles = self.getJointAngles()
        # np.resize(self.previousGoals.points, self.iteration+1)
        # np.resize(self.previousGoals.points[self.iteration].positions, 7)
        # self.previousGoals.points[self.iteration].positions = currentAngles
	self.iteration += 1
        #
        # print "Stored joint angles: "
        # print self.previousGoals.points[iteration].positions
        raw_input("Iteration # %d. Enter anything to continue: " % self.iteration)

        #---------------------------------------------------------------------------------------#

	print "MOVES3 - Scooping/pushing down into bottom of bowl..."
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + -0.03543569, self.bowl_pos[1] + -0.09179373, self.bowl_pos[2] + -0.02600000)
        (quat.x, quat.y, quat.z, quat.w) = (0.505, 0.516, -0.160, 0.673)  #  (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.713, 0.064, -0.229, 0.659)
        timeout = 3
        #self.setPositionGoal(pos, quat, timeout)
        self.setOrientGoal(pos, quat, timeout)

        # #Code for storing current joint angles in case of playback...
        # currentAngles = self.getJointAngles()
        # np.resize(self.previousGoals.points, self.iteration+1)
        # np.resize(self.previousGoals.points[self.iteration].positions, 7)
        # self.previousGoals.points[self.iteration].positions = currentAngles
	self.iteration += 1
        #
        # print "Stored joint angles: "
        # print self.previousGoals.points[iteration].positions
        raw_input("Iteration # %d. Enter anything to continue: " % self.iteration)

        #---------------------------------------------------------------------------------------#

	print "MOVES4 - Lifting a little out of bottom of bowl..."
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + 0.00156431, self.bowl_pos[1] +  -0.07279373, self.bowl_pos[2] + 0.04900000)
        (quat.x, quat.y, quat.z, quat.w) = (0.617, 0.300, -0.035, 0.726)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.700, 0.108, -0.321, 0.629)
        timeout = 3
        #self.setPositionGoal(pos, quat, timeout)
        self.setOrientGoal(pos, quat, timeout)

        # #Code for storing current joint angles in case of playback...
        # currentAngles = self.getJointAngles()
        # np.resize(self.previousGoals.points, self.iteration+1)
        # np.resize(self.previousGoals.points[self.iteration].positions, 7)
        # self.previousGoals.points[self.iteration].positions = currentAngles
	self.iteration += 1
        #
        # print "Stored joint angles: "
        # print self.previousGoals.points[iteration].positions
        raw_input("Iteration # %d. Enter anything to continue: " % self.iteration)

        #---------------------------------------------------------------------------------------#

	print "MOVES5 - Lifting above bowl..."
        (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + 0.00756431, self.bowl_pos[1] + -0.13179373, self.bowl_pos[2] +  0.59400000)
        (quat.x, quat.y, quat.z, quat.w) = (0.702, 0.168, 0.132, 0.679)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.706, 0.068, -0.235, 0.664)
        timeout = 1
        #self.setPositionGoal(pos, quat, timeout)
        self.setOrientGoal(pos, quat, timeout)

        # #Code for storing current joint angles in case of playback...
        # currentAngles = self.getJointAngles()
        # np.resize(self.previousGoals.points, self.iteration+1)
        # np.resize(self.previousGoals.points[self.iteration].positions, 7)
        # self.previousGoals.points[self.iteration].positions = currentAngles
	self.iteration += 1
        #
        # print "Stored joint angles: "
        # print self.previousGoals.points[iteration].positions
        raw_input("Iteration # %d. Enter anything to continue: " % self.iteration)

        return True

    #Just added this function, stops and reverses motion back over previous points/joint angles

    def reverseMotion(self, msg):
	self.testVar = 100
    #     self.emergencyMsg = msg
    #     self.setPostureGoal(list(self.getJointAngles()), 0.5) #stops posture at current joint angles, stops moving
    #
    #     for x in range (0, self.iteration - 1):
    #         desiredJointAngles = list(self.previousGoals.points[self.iteration].positions)
    #         self.setPostureGoal(desiredJointAngles, 1)
    #         raw_input("Reversing motion, %d nth point, press Enter to continue :" % (self.iteration - x))


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


    # def heardEnter():
    #     i,o,e = select.select([sys.stdin],[],[],1)
    #     for s in i:
    #         if s == sys.stdin:
    #             input = sys.stdin.readline()
    #             return True
    #         return False

#ORIGINAL SCOOPING CODE BEFORE CONSIDERING ANY KINECT STUFF!!!

# print "MOVES1"
# (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + -0.2920, self.bowl_pos[1] + -0.7260, self.bowl_pos[2] + 0.2600)
# (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) # (0.573, -0.451, -0.534, 0.428)
# timeout = 4
# #self.setPositionGoal(pos, quat, timeout)
# self.setOrientGoal(pos, quat, timeout)
#
# # #Code for storing current joint angles in case of playback...
# # self.currentAngles = self.getJointAngles()
# # print "Current Angles:"
# # print self.currentAngles
# # self.point.positions = self.currentAngles
# # self.previousGoals.points.append(self.point)
# # print "EVERYTHING:"
# # print self.previousGoals
# # print "resized Points:"
# # print self.previousGoals.points[self.iteration]
# #
# # print "Stored joint angles: "
# # print self.previousGoals.points[self.iteration].positions[2]
# raw_input("Iteration # %d. Enter anything to continue: " % self.iteration)
# # self.iteration += 1
#
# #---------------------------------------------------------------------------------------#
#
# print "MOVES2"
# (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + -0.1080, self.bowl_pos[1] + -0.1410, self.bowl_pos[2] + 0.2550)
# (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.690, -0.092, -0.112, 0.709)
# timeout = 4
# #self.setPositionGoal(pos, quat, timeout)
# self.setOrientGoal(pos, quat, timeout)
#
# # #Code for storing current joint angles in case of playback...
# # currentAngles = self.getJointAngles()
# # np.resize(self.previousGoals.points, self.iteration+1)
# # np.resize(self.previousGoals.points[self.iteration].positions, 7)
# # self.previousGoals.points[self.iteration].positions = currentAngles
# # self.iteration += 1
# #
# # print "Stored joint angles: "
# # print self.previousGoals.points[iteration].positions
# raw_input("Iteration # %d. Enter anything to continue: " % self.iteration)
#
# #---------------------------------------------------------------------------------------#
#
# print "MOVES3"
# (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + -0.0920, self.bowl_pos[1] + 0.0310, self.bowl_pos[2] + 0.0950)
# (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  #  (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.713, 0.064, -0.229, 0.659)
# timeout = 3
# #self.setPositionGoal(pos, quat, timeout)
# self.setOrientGoal(pos, quat, timeout)
#
# # #Code for storing current joint angles in case of playback...
# # currentAngles = self.getJointAngles()
# # np.resize(self.previousGoals.points, self.iteration+1)
# # np.resize(self.previousGoals.points[self.iteration].positions, 7)
# # self.previousGoals.points[self.iteration].positions = currentAngles
# # self.iteration += 1
# #
# # print "Stored joint angles: "
# # print self.previousGoals.points[iteration].positions
# raw_input("Iteration # %d. Enter anything to continue: " % self.iteration)
#
# #---------------------------------------------------------------------------------------#
#
# print "MOVES4"
# (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + -0.0140, self.bowl_pos[1] +  0, self.bowl_pos[2] + 0.00900)
# (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.700, 0.108, -0.321, 0.629)
# timeout = 3
# #self.setPositionGoal(pos, quat, timeout)
# self.setOrientGoal(pos, quat, timeout)
#
# # #Code for storing current joint angles in case of playback...
# # currentAngles = self.getJointAngles()
# # np.resize(self.previousGoals.points, self.iteration+1)
# # np.resize(self.previousGoals.points[self.iteration].positions, 7)
# # self.previousGoals.points[self.iteration].positions = currentAngles
# # self.iteration += 1
# #
# # print "Stored joint angles: "
# # print self.previousGoals.points[iteration].positions
# raw_input("Iteration # %d. Enter anything to continue: " % self.iteration)
#
# #---------------------------------------------------------------------------------------#
#
# print "MOVES5"
# (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + 0, self.bowl_pos[1] + 0, self.bowl_pos[2] +  0) #REACHED THE ACTUAL BOWL
# (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.706, 0.068, -0.235, 0.664)
# timeout = 1
# #self.setPositionGoal(pos, quat, timeout)
# self.setOrientGoal(pos, quat, timeout)
#
# # #Code for storing current joint angles in case of playback...
# # currentAngles = self.getJointAngles()
# # np.resize(self.previousGoals.points, self.iteration+1)
# # np.resize(self.previousGoals.points[self.iteration].positions, 7)
# # self.previousGoals.points[self.iteration].positions = currentAngles
# # self.iteration += 1
# #
# # print "Stored joint angles: "
# # print self.previousGoals.points[iteration].positions
# raw_input("Iteration # %d. Enter anything to continue: " % self.iteration)
#
# #---------------------------------------------------------------------------------------#
#
# print "MOVES6"
# (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + 0.0120, self.bowl_pos[1] + -0.00100, self.bowl_pos[2] +  0.0130)
# (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.672, 0.037, -0.137, 0.727)
# timeout = 1
# #self.setPositionGoal(pos, quat, timeout)
# self.setOrientGoal(pos, quat, timeout)
#
# # #Code for storing current joint angles in case of playback...
# # currentAngles = self.getJointAngles()
# # np.resize(self.previousGoals.points, self.iteration+1)
# # np.resize(self.previousGoals.points[self.iteration].positions, 7)
# # self.previousGoals.points[self.iteration].positions = currentAngles
# # self.iteration += 1
# #
# # print "Stored joint angles: "
# # print self.previousGoals.points[iteration].positions
# raw_input("Iteration # %d. Enter anything to continue: " % self.iteration)
#
# #---------------------------------------------------------------------------------------#
#
# print "MOVES7"
# (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + -0.122, self.bowl_pos[1] + -0.0319, self.bowl_pos[2] + 0.300)
# (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.699, -0.044, -0.085, 0.709)
# timeout = 1
# #self.setPositionGoal(pos, quat, timeout)
# self.setOrientGoal(pos, quat, timeout)
#
# # #Code for storing current joint angles in case of playback...
# # currentAngles = self.getJointAngles()
# # np.resize(self.previousGoals.points, self.iteration+1)
# # np.resize(self.previousGoals.points[self.iteration].positions, 7)
# # self.previousGoals.points[self.iteration].positions = currentAngles
# # self.iteration += 1
# #
# # print "Stored joint angles: "
# # print self.previousGoals.points[iteration].positions
# raw_input("Iteration # %d. Enter anything to continue: " % self.iteration)
#
# #---------------------------------------------------------------------------------------#
#
# print "MOVES8"
# (pos.x, pos.y, pos.z) = (self.bowl_pos[0] + -0.0210, self.bowl_pos[1] +  -0.0519, self.bowl_pos[2] +  0.410)
# (quat.x, quat.y, quat.z, quat.w) = (0.696, 0.035, -0.008, 0.717)  # (self.bowl_quat[0], self.bowl_quat[1], self.bowl_quat[2], self.bowl_quat[3]) #  (0.677, -0.058, -0.004, 0.733)
# timeout = 1
# #self.setPositionGoal(pos, quat, timeout)
# self.setOrientGoal(pos, quat, timeout)
#
# # #Code for storing current joint angles in case of playback...
# # currentAngles = self.getJointAngles()
# # np.resize(self.previousGoals.points, self.iteration+1)
# # np.resize(self.previousGoals.points[self.iteration].positions, 7)
# # self.previousGoals.points[self.iteration].positions = currentAngles
# # self.iteration += 1
# #
# #
# # print "Stored joint angles: "
# # print self.previousGoals.points[iteration].positions
# raw_input("Iteration # %d. Enter anything to continue: " % self.iteration)
