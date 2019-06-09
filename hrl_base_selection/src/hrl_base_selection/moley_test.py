#!/usr/bin/env python

#
# Author: Ariel Kapusta
# Contact: akapusta@gmail.com
#
# Date: November 11, 2017
#
# Code for technical test
#

import numpy as np
import math as m
import copy, os, threading

import rospy
from std_msgs.msg import UInt8

import openravepy as op
from openravepy.misc import InitOpenRAVELogging

class ArmWorkspaceTester(object):
    def __init__(self, visualize=False, reset_save_file=True, visualize_ignoring_collisions=False):
        self.visualize = visualize
        self.visualize_ignoring_collisions = visualize_ignoring_collisions
        self.save_file_path = '/home/ari/git/catkin_ws/src/hrl-assistive/hrl_base_selection/data/test/'
        self.save_file_name = 'test_save.csv'

        file_list = os.listdir(self.save_file_path)
        if reset_save_file or self.save_file_name not in file_list:
            open(self.save_file_path + self.save_file_name, 'w').close()

        self.frame_lock = threading.RLock()
        self.setup_openrave()
        self.setup_comms()

    def setup_comms(self):
        self.workspace_check_sub = rospy.Subscriber('/workspace_check_request', UInt8, self.workspace_check_cb)
        rospy.loginfo('Arm Workspace Tester is now ready!')

    def workspace_check_cb(self, msg):
        with self.frame_lock:
            results = self.sample_poses(msg.data)

    ## Input: number of poses tested
    ## Output: A list where each entry is of the form: [pose_number, x, y, z, rz, ry, rx, result]
    ## Pose number: the integer of the pose being tested
    ## x, y, z: the x, y, z cartesian position of the pose being tested
    ## rz, ry, rx: euler angles (yaw, pitch, roll) about the z, y, and x axes in that order. Axes
    ## are static (they do not rotate with previous rotations)
    ## result: a string with 'successful' if the robot can acheive the pose, 'unsuccessful' otherwise
    def sample_poses(self, number):
        output = []
        uniform_samples = np.random.uniform([ 0., -1., 0., 0., 0., 0.], [ 0.7, 1., 1.5, 1.0, 1.0, 1.0], [number, 6])
        for i in xrange(number):
            ## Sample positions and orientations uniformly
            x = uniform_samples[i, 0]
            y = uniform_samples[i, 1]
            z = uniform_samples[i, 2]
            theta = 2. * m.pi * uniform_samples[i, 3]
            phi = m.acos(1. - 2. * uniform_samples[i, 4])
            roll = 2. * m.pi * uniform_samples[i, 5]

            ## Calculate the goal pose homogeneous transform
            translation = np.matrix([[1., 0., 0, x], [0., 1., 0, y], [0, 0, 1., z], [0, 0, 0, 1.]])
            rotz = np.matrix([[m.cos(theta), -m.sin(theta), 0, 0], [m.sin(theta), m.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            roty = np.matrix([[m.cos(phi), 0, m.sin(phi), 0], [0, 1, 0, 0], [-m.sin(phi), 0, m.cos(phi), 0], [0, 0, 0, 1]])
            rotx = np.matrix([[1, 0, 0, 0], [0, m.cos(roll), -m.sin(roll), 0], [0, m.sin(roll), m.cos(roll), 0], [0, 0, 0, 1]])
            goal_pose = translation * rotz * roty * rotx
            sol = None
            sol = self.manip.FindIKSolution(np.array(goal_pose), filteroptions=op.IkFilterOptions.CheckEnvCollisions)
            if sol is not None:
                result = 'successful'
            else:
                result = 'unsuccessful'
            rospy.loginfo('Pose number: ' + str(i + 1))
            rospy.loginfo('%f, %f, %f, %f, %f, %f' % (x, y, z, theta, phi, roll))
            rospy.loginfo(str(result))
            with open(self.save_file_path + self.save_file_name, 'a') as myfile:
                myfile.write('%i,%f,%f,%f,%f,%f,%f,%s\n' % (i + 1, x, y, z, theta, phi, roll, result))
            output.append([i + 1, x, y, z, theta, phi, roll, result])
            if sol is None and self.visualize_ignoring_collisions:
                sol = self.manip.FindIKSolution(np.array(goal_pose),
                                                filteroptions=op.IkFilterOptions.IgnoreSelfCollisions)
            if sol is not None and (self.visualize or self.visualize_ignoring_collisions):
                self.robot.SetDOFValues(sol, self.manip.GetArmIndices())
                self.env.UpdatePublishedBodies()
                rospy.sleep(1.)
        return output

    def setup_openrave(self):
        ## Setup Openrave ENV
        InitOpenRAVELogging()
        self.env = op.Environment()

        # Lets you visualize openrave.
        if self.visualize or self.visualize_ignoring_collisions:
            self.env.SetViewer('qtcoin')

        ## Load OpenRave ur5-with-barrett-hand Model
        ## There is a commented line to use a wam7 arm with barrett hand instead
        # self.env.Load('robots/barrettwam.robot.xml')
        self.env.Load('robots/ur5-barretthand.xml')
        self.robot = self.env.GetRobots()[0]

        ## Initialize the position of the base of the robot arm
        tx,ty,tz = 0., 0., 1.
        translation = np.matrix([[1., 0., 0, tx], [0., 1., 0, ty], [0, 0, 1., tz], [0, 0, 0, 1.]])
        thz = m.radians(0.)
        z = np.matrix([[m.cos(thz), -m.sin(thz), 0, 0], [m.sin(thz), m.cos(thz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        thy = m.radians(180.)
        y = np.matrix([[m.cos(thy), 0, m.sin(thy), 0], [0, 1, 0, 0], [-m.sin(thy), 0, m.cos(thy), 0], [0, 0, 0, 1]])
        thx = m.radians(0.)
        x = np.matrix([[1, 0, 0, 0], [0, m.cos(thx), -m.sin(thx), 0], [0, m.sin(thx), m.cos(thx), 0], [0, 0, 0, 1]])
        robot_start = translation * z * y * x
        self.robot.SetTransform(np.array(robot_start))

        ## Initialize the environment
        self.environment_box = op.RaveCreateKinBody(self.env, '')
        self.environment_box.SetName('Environment_Box')

        box_list = []
        ## Front of robot environment
        new_box = op.KinBody.Link.GeometryInfo()
        new_box._type = op.KinBody.Link.GeomType.Box
        new_box._fTransparency = 0.7
        new_box._t = np.eye(4)
        new_box._t[0:3, 3] = [-0.005, 0., 0.75]
        new_box._vGeomData=[0.005, 1.0, 0.75]
        box_list.append(copy.copy(new_box))

        ## Back of robot environment
        new_box = op.KinBody.Link.GeometryInfo()
        new_box._type = op.KinBody.Link.GeomType.Box
        new_box._fTransparency = 0.7
        new_box._t = np.eye(4)
        new_box._t[0:3, 3] = [0.705, 0., 0.75]
        new_box._vGeomData = [0.005, 1.0, 0.75]
        box_list.append(copy.copy(new_box))

        ## Top of robot environment
        new_box = op.KinBody.Link.GeometryInfo()
        new_box._type = op.KinBody.Link.GeomType.Box
        new_box._fTransparency = 0.7
        new_box._t = np.eye(4)
        new_box._t[0:3, 3] = [0.35, 0., 1.505]
        new_box._vGeomData = [0.35, 1.0, 0.005]
        box_list.append(copy.copy(new_box))

        ## Bottom of robot environment
        new_box = op.KinBody.Link.GeometryInfo()
        new_box._type = op.KinBody.Link.GeomType.Box
        new_box._fTransparency = 0.7
        new_box._t = np.eye(4)
        new_box._t[0:3, 3] = [0.35, 0., -0.005]
        new_box._vGeomData = [0.35, 1.0, 0.005]
        box_list.append(copy.copy(new_box))

        ## Left side of robot environment
        new_box = op.KinBody.Link.GeometryInfo()
        new_box._type = op.KinBody.Link.GeomType.Box
        new_box._fTransparency = 0.7
        new_box._t = np.eye(4)
        new_box._t[0:3, 3] = [0.35, -1.005, 0.75]
        new_box._vGeomData = [0.35, 0.005, 0.75]
        box_list.append(copy.copy(new_box))

        ## Right side of robot environment
        new_box = op.KinBody.Link.GeometryInfo()
        new_box._type = op.KinBody.Link.GeomType.Box
        new_box._fTransparency = 0.7
        new_box._t = np.eye(4)
        new_box._t[0:3, 3] = [0.35, 1.005, 0.75]
        new_box._vGeomData = [0.35, 0.005, 0.75]
        box_list.append(copy.copy(new_box))

        self.environment_box.InitFromGeometries(box_list)
        self.env.AddKinBody(self.environment_box)
        self.env.UpdatePublishedBodies()

        ## Set up the robot manipulator and inverse kinematics solver
        self.robot.SetActiveManipulator('arm_hand')
        self.manip = self.robot.GetActiveManipulator()
        self.ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot,
                                                                             iktype=op.IkParameterization.Type.Transform6D)
        if not self.ikmodel.load():
            rospy.loginfo('IK model not found for robot. Will now generate an IK model. This will take a while!')
            self.ikmodel.autogenerate()
        rospy.loginfo('IK for robot is ready!')


if __name__ == "__main__":
    rospy.init_node('workspace_tester')
    import optparse
    import ast

    p = optparse.OptionParser()

    # Option to select whether to visualize the simulator as it checks if collision-free IK
    # solutions exist. This slows down the process because it pauses at each solution for 1 second
    # so you can see what the IK solution looks like.
    p.add_option('--visualize', action='store', dest='visualize', default='False',
                 help='Choose whether or not to visualize the simulator.')

    # Option to select if you want to reset the save file. It clears any existing save file if it exists.
    # If false, it adds results to the end of the existing file, if it exists.
    p.add_option('--reset_save_file', action='store', dest='reset_save_file', default='False',
                 help='If true, clears the save file to store any new results. If false, adds to the end of the '
                      'existing file, if it exists.')

    # Option to select whether to visualize the simulator as it checks if collision-free IK
    # solutions exist. This slows down the process because it pauses at each solution for 1 second
    # so you can see what the IK solution looks like. If this is true, it also will visualize
    # solutions that are in collision. Results saved to file and output to ros info are unchanged
    # (they still describe results in collision as unsuccessful).
    p.add_option('--visualize_ignoring_collisions', action='store', dest='visualize_ignoring_collisions',
                 default='False',
                 help='If true, includes visualization of the configurations in collision. Results saved to file'
                      'and output to ros info are unchanged.')
    opt, args = p.parse_args()
    try:
        tester = ArmWorkspaceTester(visualize=ast.literal_eval(opt.visualize),
                                    reset_save_file=ast.literal_eval(opt.reset_save_file),
                                    visualize_ignoring_collisions=ast.literal_eval(opt.visualize_ignoring_collisions))
        rospy.spin()
    except ValueError:
        rospy.loginfo('Options must be of the form \'True\' or \'False\' so they can be evaluated to '
                      'a python boolean. Try again!')
