#!/usr/bin/env python

import numpy as np
import math as m
import openravepy as op
import copy

import time
import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
import rospy, rospkg
import tf
from geometry_msgs.msg import PoseStamped
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from sensor_msgs.msg import JointState
from std_msgs.msg import String
import hrl_lib.transforms as tr
from hrl_base_selection.srv import BaseMove_multi
from visualization_msgs.msg import Marker
from helper_functions import createBMatrix, is_number, Bmat_to_pos_quat
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from itertools import combinations as comb
import tf.transformations as tft
from matplotlib.cbook import flatten
from sensor_msgs.msg import JointState
from hrl_msgs.msg import FloatArrayBare

import pickle
roslib.load_manifest('hrl_lib')
from hrl_lib.util import load_pickle



class BaseSelector(object):
    joint_names = ['l_shoulder_pan_joint',
                   'l_shoulder_lift_joint',
                   'l_upper_arm_roll_joint',
                   'l_elbow_flex_joint',
                   'l_forearm_roll_joint',
                   'l_wrist_flex_joint',
                   'l_wrist_roll_joint']

    def __init__(self, transform_listener=None, model='chair', testing=False):
        if transform_listener is None:
            self.listener = tf.TransformListener()
        else:
            self.listener = transform_listener

        # self.model = model
        self.vis_pub = rospy.Publisher("~service_subject_model", Marker, latch=True)

        self.bed_state_z = 0.
        self.bed_state_head_theta = 0.
        self.bed_state_leg_theta = 0.

        self.robot_z = 0
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)
        self.pr2_B_ar = None
        # Publisher to let me test things with arm_reacher
        #self.wc_position = rospy.Publisher("~pr2_B_wc", PoseStamped, latch=True)

        # Just for testing
        self.testing = testing
        if self.testing:
            angle = -m.pi/2
            pr2_B_head1  =  np.matrix([[   m.cos(angle),  -m.sin(angle),          0.,        0.],
                                       [   m.sin(angle),   m.cos(angle),          0.,       2.5],
                                       [             0.,             0.,          1.,       1.1546],
                                       [             0.,             0.,          0.,        1.]])
            an = -m.pi/4
            pr2_B_head2 = np.matrix([[  m.cos(an),   0.,  m.sin(an),       0.],
                                     [         0.,   1.,         0.,       0.],
                                     [ -m.sin(an),   0.,  m.cos(an),       0.],
                                     [         0.,   0.,         0.,       1.]])
            self.pr2_B_head = pr2_B_head1*pr2_B_head2

            self.pr2_B_ar = np.matrix([[   m.cos(angle),  -m.sin(angle),          0.,       1.],
                                       [   m.sin(angle),   m.cos(angle),          0.,       0.],
                                       [             0.,             0.,          1.,       .5],
                                       [             0.,             0.,          0.,       1.]])

        start_time = time.time()
        print 'Loading data, please wait.'
        self.chair_scores = self.load_task('yogurt', 'chair')
        self.autobed_scores = self.load_task('yogurt', 'chair')
        print 'Time to receive load data: %fs' % (time.time()-start_time)
        # Service
        self.base_service = rospy.Service('select_base_position', BaseMove_multi, self.handle_select_base)
        
        # Subscriber to update robot joint state
        #self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)
        
        print "Ready to select base."

        '''
        self.joint_names = []
        self.joint_angles = []
        self.selection_mat = np.zeros(11)


        self.setup_openrave()
        
        self.POSES = []
        TARGETS =  np.array([[[0.252, -0.067, -0.021], [0.102, 0.771, 0.628, -0.002]],    #Face area
                             [[0.252, -0.097, -0.021], [0.102, 0.771, 0.628, -0.002]],    #Face area
                             [[0.252, -0.097, -0.061], [0.102, 0.771, 0.628, -0.002]],    #Face area
                             [[0.252,  0.067, -0.021], [0.102, 0.771, 0.628, -0.002]],    #Face area
                             [[0.252,  0.097, -0.061], [0.102, 0.771, 0.628, -0.002]],    #Face area
                             [[0.252,  0.097, -0.021], [0.102, 0.771, 0.628, -0.002]],    #Face area
                             [[0.108, -0.236, -0.105], [0.346, 0.857, 0.238,  0.299]],    #Shoulder area
                             [[0.108, -0.256, -0.105], [0.346, 0.857, 0.238,  0.299]],    #Shoulder area
                             [[0.443, -0.032, -0.716], [0.162, 0.739, 0.625,  0.195]],    #Knee area
                             [[0.443, -0.032, -0.716], [0.162, 0.739, 0.625,  0.195]],    #Knee area
                             [[0.337, -0.228, -0.317], [0.282, 0.850, 0.249,  0.370]],    #Arm area
                             [[0.367, -0.228, -0.317], [0.282, 0.850, 0.249,  0.370]]])   #Arm area

                             
        #self.pr2_B_wc = []
        self.Tgrasps = []
        #self.weights = []
        self.best_score = 0
        self.goals = []
        for target in TARGETS:
            self.POSES.append(createBMatrix(target[0],target[1]))
        '''

    def setup_openrave(self):
        '''
        # Setup Openrave ENV
        self.env = op.Environment()

        # Lets you visualize openrave. Uncomment to see visualization. Does not work through ssh.
        #self.env.SetViewer('qtcoin')



        ## Load PR2 Model
        self.env.Load('robots/pr2-beta-static.zae')
        self.robot = self.env.GetRobots()[0]
        v = self.robot.GetActiveDOFValues()
        v[self.robot.GetJoint('l_shoulder_pan_joint').GetDOFIndex()]= 3.14/2
        v[self.robot.GetJoint('r_shoulder_pan_joint').GetDOFIndex()] = -3.14/2
        v[self.robot.GetJoint('l_gripper_l_finger_joint').GetDOFIndex()] = .54
        v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = .3
        self.robot.SetActiveDOFValues(v)
        robot_start = np.matrix([[m.cos(0.), -m.sin(0.), 0., 0.],
                                 [m.sin(0.),  m.cos(0.), 0., 0.],
                                 [0.       ,         0., 1., 0.],
                                 [0.       ,         0., 0., 1.]])
        self.robot.SetTransform(np.array(robot_start))

        ## Set robot manipulators, ik, planner
        self.robot.SetActiveManipulator('leftarm')
        self.manip = self.robot.GetActiveManipulator()
        ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Transform6D)
        if not ikmodel.load():
            ikmodel.autogenerate()
        # create the interface for basic manipulation programs
        self.manipprob = op.interfaces.BaseManipulation(self.robot)

        v = self.robot.GetActiveDOFValues()
        for name in self.joint_names:
            v[self.robot.GetJoint(name).GetDOFIndex()] = self.joint_angles[self.joint_names.index(name)]
        self.robot.SetActiveDOFValues(v)

        # Set up inverse reachability
        #self.irmodel = op.databases.inversereachability.InverseReachabilityModel(robot=self.robot)
        #print 'loading irmodel'
        #starttime = time.time()
        #if not self.irmodel.load():
        #    print 'do you want to generate irmodel for your robot? it might take several hours'
        #    print 'or you can go to http://people.csail.mit.edu/liuhuan/pr2/openrave/openrave_database/ to get the database for PR2'
        #    input = raw_input('[Y/n]\n')
        #    if input == 'y' or input == 'Y' or input == '\n' or input == '':
        #        self.irmodel.autogenerate()
       #         self.irmodel.load()
       #     else:
       #         raise ValueError('')
       # print 'time to load inverse-reachability model: %fs'%(time.time()-starttime)
        # make sure the robot and manipulator match the database
       # assert self.irmodel.robot == self.robot and self.irmodel.manip == self.robot.GetActiveManipulator()

        ## Find and load Wheelchair Model
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('hrl_base_selection')
        self.env.Load(''.join([pkg_path, '/models/ADA_Wheelchair.dae']))
        self.wheelchair = self.env.GetBodies()[1]
        '''
        print 'I ran openrave setup despite it not doing anything'

    def joint_state_cb(self, msg):
        #This gets the joint states of the entire robot.
        for num, name in enumerate(msg.name):
            if name == 'torso_lift_joint':
                self.robot_z = msg.position[num]

    # Publishes the wheelchair model location used by openrave to rviz so we can see how it overlaps with the real
    # wheelchair
    def publish_subject_marker(self, pos, ori):
        marker = Marker()
        marker.header.frame_id = "/base_link"
        marker.header.stamp = rospy.Time()
        marker.ns = "base_service_subject_model"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE;
        marker.action = Marker.ADD
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = 0
        marker.pose.orientation.x = ori[0]
        marker.pose.orientation.y = ori[1]
        marker.pose.orientation.z = ori[2]
        marker.pose.orientation.w = ori[3]
        marker.scale.x = .0254
        marker.scale.y = .0254
        marker.scale.z = .0254
        marker.color.a = 1.
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        if self.model=='chair':
            name = 'wc_model'
            marker.mesh_resource = "package://hrl_base_selection/models/ADA_Wheelchair.dae"
            marker.scale.x = .0254
            marker.scale.y = .0254
            marker.scale.z = .0254
        elif self.model=='bed':
            name = 'bed_model'
            marker.mesh_resource = "package://hrl_base_selection/models/head_bed.dae"
            marker.scale.x = 1.0
            marker.scale.y = 1.
            marker.scale.z = 1.0
        elif self.model=='autobed':
            name = 'autobed_model'
            marker.mesh_resource = "package://hrl_base_selection/models/bed_and_body_v3_rviz.dae"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        else:
            print 'I got a bad model. What is going on???'
            return None
        self.vis_pub.publish(marker)

    def bed_state_cb(self, data):
        self.bed_state_z = data.data[1]
        self.bed_state_head_theta = data.data[0]
        self.bed_state_leg_theta = data.data[2]

     # Function that determines a good base location to be able to reach the goal location.
    #def handle_select_base(self, req):#, task):
    def handle_select_base(self, req):
        model = req.model
        self.model = model
        task = req.task
        self.task = task
        if model == 'autobed':
            self.autobed_sub = rospy.Subscriber('/bed_states', FloatArrayBare, self.bed_state_cb)
        print 'I have received inputs!'
        start_time = time.time()
        #print 'My given inputs were: \n'
        #print 'head is: \n', req.head

        # The head location is received as a posestamped message and is converted and used as the head location.
        # pos_temp = [req.head.pose.position.x,
        #             req.head.pose.position.y,
        #             req.head.pose.position.z]
        # ori_temp = [req.head.pose.orientation.x,
        #             req.head.pose.orientation.y,
        #             req.head.pose.orientation.z,
        #             req.head.pose.orientation.w]
        # self.pr2_B_head = createBMatrix(pos_temp, ori_temp)
        # #print 'head from input: \n', head
        if not self.testing:
            try:
                now = rospy.Time.now()
                self.listener.waitForTransform('/base_link', '/head_frame', now, rospy.Duration(10))
                (trans, rot) = self.listener.lookupTransform('/base_link', '/head_frame', now)
                self.pr2_B_head = createBMatrix(trans, rot)
                if model == 'chair':
                    now = rospy.Time.now()
                    self.listener.waitForTransform('/base_link', '/ar_marker', now, rospy.Duration(10))
                    (trans, rot) = self.listener.lookupTransform('/base_link', '/ar_marker', now)
                    self.pr2_B_ar = createBMatrix(trans, rot)
                elif model == 'autobed':
                    now = rospy.Time.now()
                    self.listener.waitForTransform('/base_link', '/ar_marker_autobed', now, rospy.Duration(10))
                    (trans, rot) = self.listener.lookupTransform('/base_link', '/ar_marker_autobed', now)
                    self.pr2_B_ar = createBMatrix(trans, rot)
                    ar_trans_B = np.eye(4)
                    ar_trans_B[0:3,3] = np.array([0.5, .5, .3+self.bed_state_z/100])
                    ar_rotz_B = np.eye(4)
                    ar_rotz_B [0:2,0:2] = np.array([[-1,0],[0,-1]])
                    ar_rotx_B = np.eye(4)
                    ar_rotx_B[1:3,1:3] = np.array([[0,1],[1,0]])
                    self.model_B_ar = np.matrix(ar_trans_B)*np.matrix(rotz_B)*np.matrix(ar_rotx_B)
                    # now = rospy.Time.now()
                    # self.listener.waitForTransform('/ar_marker', '/bed_frame', now, rospy.Duration(3))
                    # (trans, rot) = self.listener.lookupTransform('/ar_marker', '/bed_frame', now)
                    # self.ar_B_model = createBMatrix(trans, rot)
                if np.linalg.norm(trans) > 2:
                    rospy.loginfo('AR tag is too far away. Use the \'Testing\' button to move PR2 to 1 meter from AR '
                                  'tag. Or just move it closer via other means. Alternatively, the PR2 may have lost '
                                  'sight of the AR tag or it is having silly issues recognizing it. ')
                    return None

            except Exception as e:
                rospy.loginfo("TF Exception. Could not get the AR_tag location, bed location, or "
                              "head location:\r\n%s" % e)
                return None

        print 'The homogeneous transfrom from PR2 base link to head: \n', self.pr2_B_head
        z_origin = np.array([0, 0, 1])
        x_head = np.array([self.pr2_B_head[0, 0], self.pr2_B_head[1, 0], self.pr2_B_head[2, 0]])
        y_head_project = np.cross(z_origin, x_head)
        y_head_project = y_head_project/np.linalg.norm(y_head_project)
        x_head_project = np.cross(y_head_project, z_origin)
        x_head_project = x_head_project/np.linalg.norm(x_head_project)
        self.pr2_B_head_project = np.eye(4)
        for i in xrange(3):
            self.pr2_B_head_project[i, 0] = x_head_project[i]
            self.pr2_B_head_project[i, 1] = y_head_project[i]
            self.pr2_B_head_project[i, 3] = self.pr2_B_head[i, 3]

        self.pr2_B_headfloor = copy.copy(np.matrix(self.pr2_B_head_project))
        self.pr2_B_headfloor[2, 3] = 0.
        print 'The homogeneous transform from PR2 base link to the head location projected onto the ground plane: \n', \
            self.pr2_B_headfloor




        print 'I will now determine a good base location.'
        headx = 0
        heady = 0
        # Sets the wheelchair location based on the location of the head using a few homogeneous transforms.
        if model == 'chair':
            # self.pr2_B_headfloor = copy.copy(np.matrix(self.pr2_B_head_project))
            # self.pr2_B_headfloor[2, 3] = 0.


            # Transform from the coordinate frame of the wc model in the back right bottom corner, to the head location
            originsubject_B_headfloor = np.matrix([[m.cos(0.), -m.sin(0.),  0., .442603], #.45 #.438
                                                   [m.sin(0.),  m.cos(0.),  0., .384275], #0.34 #.42
                                                   [       0.,         0.,  1.,      0.],
                                                   [       0.,         0.,  0.,      1.]])
            self.origin_B_pr2 = copy.copy(self.pr2_B_headfloor.I)
            # self.origin_B_pr2 = self.headfloor_B_head * self.pr2_B_head.I
            # reference_floor_B_pr2 = self.pr2_B_head * self.headfloor_B_head.I * originsubject_B_headfloor.I

        # Regular bed is now deprecated. To use need to fix to be similar to chair.
        if model =='bed':
            an = -m.pi/2
            self.headfloor_B_head = np.matrix([[  m.cos(an),   0.,  m.sin(an),       0.], #.45 #.438
                                               [         0.,   1.,         0.,       0.], #0.34 #.42
                                               [ -m.sin(an),   0.,  m.cos(an),   1.1546],
                                               [         0.,   0.,         0.,       1.]])
            an2 = 0
            originsubject_B_headfloor = np.matrix([[ m.cos(an2),  0., m.sin(an2),  .2954], #.45 #.438
                                                   [         0.,  1.,         0.,     0.], #0.34 #.42
                                                   [-m.sin(an2),  0., m.cos(an2),     0.],
                                                   [         0.,  0.,         0.,     1.]])
            self.origin_B_pr2 = self.headfloor_B_head * self.pr2_B_head.I
            # subject_location = self.pr2_B_head * self.headfloor_B_head.I * originsubject_B_headfloor.I

        if model == 'autobed':
            # an = -m.pi/2
            # self.headfloor_B_head = np.matrix([[  m.cos(an),   0.,  m.sin(an),       0.], #.45 #.438
            #                                    [         0.,   1.,         0.,       0.], #0.34 #.42
            #                                    [ -m.sin(an),   0.,  m.cos(an),   1.1546],
            #                                    [         0.,   0.,         0.,       1.]])
            # an2 = 0
            # self.origin_B_model = np.matrix([[       1.,        0.,   0.,              0.0],
            #                                  [       0.,        1.,   0.,              0.0],
            #                                  [       0.,        0.,   1., self.bed_state_z],
            #                                  [       0.,        0.,   0.,              1.0]])
            self.model_B_pr2 = self.model_B_ar * self.pr2_B_ar.I
            self.origin_B_pr2 = copy.copy(self.model_B_pr2)
            model_B_head = self.model_B_pr2 * self.pr2_B_headfloor

            ## This next bit selects what entry in the dictionary of scores to use based on the location of the head
            # with respect to the bed model. Currently it just selects the dictionary entry with the closest relative
            # head location. Ultimately it should use a gaussian to get scores from the dictionary based on the actual
            # head location.

            if model_B_head[0, 3] > -.025 and model_B_head[0, 3] < .025:
                headx = 0.
            elif model_B_head[0, 3] >= .025 and model_B_head[0, 3] < .75:
                headx = 0.
            elif model_B_head[0, 3] <= -.025 and model_B_head[0, 3] > -.75:
                headx = 0.
            elif model_B_head[0, 3] >= .075:
                headx = 0.
            elif model_B_head[0, 3] <= -.075:
                headx = 0.

            if model_B_head[1, 3] > -.025 and model_B_head[1, 3] < .025:
                heady = 0
            elif model_B_head[1, 3] >= .025 and model_B_head[1, 3] < .075:
                heady = .05
            elif model_B_head[1, 3] > -.075 and model_B_head[1, 3] <= -.025:
                heady = -.05
            elif model_B_head[1, 3] >= .075:
                heady = .1
            elif model_B_head[1, 3] <= -.075:
                heady = -.1

            # subject_location = self.pr2_B_head * self.headfloor_B_head.I * originsubject_B_headfloor.I

        # subject_location = self.pr2_B_head * self.headfloor_B_head.I * originsubject_B_headfloor.I
        subject_location = self.origin_B_pr2.I
        #self.subject.SetTransform(np.array(subject_location))


        # Get score data and convert to goal locations
        print 'Time to receive head location and start things off: %fs' % (time.time()-start_time)
        start_time = time.time()
        if self.model == 'chair':
            all_scores = self.chair_scores
        elif self.model == 'autobed':
            all_scores = self.autobed_scores
        # all_scores = self.load_task(task, model)
        scores = all_scores[headx, heady]
        if scores == None:
            print 'Failed to load precomputed reachability data. That is a problem. Abort!'
            return None, None
#        for score in scores:
#            if score[0][4]!=0:
#                print score[0]
        
        # print 'Time to load pickle: %fs' % (time.time()-start_time)
        start_time = time.time()

        ## Set the weights for the different scores.
        alpha = 0.000  # Weight on base's closeness to goal
        beta = 1.  # Weight on number of reachable goals
        gamma = 1.  # Weight on manipulability of arm at each reachable goal
        zeta = .2  # Weight on distance to move to get to that goal location
        # pr2_B_headfloor = self.pr2_B_head*self.headfloor_B_head.I
        # headfloor_B_pr2 = pr2_B_headfloor.I
        pr2_loc = np.array([self.origin_B_pr2[0, 3], self.origin_B_pr2[1, 3]])
        length = len(scores)
        temp_scores = np.zeros([length, 1])
        temp_locations = scores[:, 0]
        print 'Original number of scores: ', length
        print 'Time to start data processing: %fs' % (time.time()-start_time)
        for j in xrange(length):
            #print 'score: \n',score
            dist_score = 0
            for i in xrange(len(scores[j, 0][0])):
                #headfloor_B_goal = createBMatrix([scores[j,0][i],scores[j,1][i],0],tr.matrix_to_quaternion(tr.rotZ(scores[j,2][i])))
                #print 'from createbmatrix: \n', headfloor_B_goal
                #headfloor_B_goal = np.matrix([[  m.cos(scores[j,2][i]), -m.sin(scores[j,2][i]),                0.,        scores[j,0][i]],
                #                              [  m.sin(scores[j,2][i]),  m.cos(scores[j,2][i]),                0.,        scores[j,1][i]],
                #                              [                  0.,                  0.,                1.,                 0.],
                #                              [                  0.,                  0.,                0.,                 1.]]) 
                #dist = np.linalg.norm(pr2_loc-scores[j,0:2][i])
                #headfloor_B_goal = np.diag([1.,1.,1.,1.])
                #headfloor_B_goal[0,3] = scores[j,0][i]
                #headfloor_B_goal[1,3] = scores[j,1][i]
                #th = scores[j,2][i]
                #ct=m.cos(th)
                #st = m.sin(th)
                #headfloor_B_goal[0,0] = ct
                #headfloor_B_goal[0,1] = -st
                #headfloor_B_goal[1,0] = st
                #headfloor_B_goal[1,1] = ct
                #headfloor_B_goal[0:2,0:2] = np.array([[m.cos(scores[j,2][i]),-m.sin(scores[j,2][i])],[m.sin(scores[j,2][i]),m.cos(scores[j,2][i])]])
                #print 'from manual: \n', headfloor_B_goal
                #dist_score += np.linalg.norm((pr2_B_headfloor*headfloor_B_goal)[0:2,3])
                # print pr2_loc
                # print scores[0, 0]
                # print 'i ', i
                dist_score += np.linalg.norm([pr2_loc[0]-scores[j, 0][0][i], pr2_loc[1]-scores[j, 0][1][i]])

            # This adds to dist score a cost for moving the robot in the z axis. Commented out currently.
            # dist_score += np.max([t for t in ((np.linalg.norm(self.robot_z - scores[j, 0][3][i]))
            #                                   for i in xrange(len(scores[j, 0][0]))
            #                                   )
            #                       ])


            thisScore = -alpha*scores[j, 1][0]+beta*scores[j, 1][1]+gamma*scores[j, 1][2]-zeta*dist_score
            if thisScore < 0:
                thisScore = 0
            #print 'This score: ',thisScore
            # temp_scores[j,0] = 0
            temp_scores[j, 0] = copy.copy(thisScore)
            #if thisScore>0:
            #    temp_scores[j] = copy.copy(thisScore)
                #temp_locations[num] = np.vstack([temp_locations,score[0]])
            #else: 
            #    temp_scores = np.delete(temp_scores,j,0)
            #    temp_locations = np.delete(temp_locations,j,0)
        #print 'Time to run through data: %fs'%(time.time()-start_time)
        #temp_locations = np.delete(temp_scores,0,0)
        temp_scores = np.hstack([list(temp_locations), temp_scores])
        out_score = []
        for i in xrange(length):
            out_score.append([temp_locations[i], temp_scores[i]])
        out_score = np.array(out_score)

                #reachable.append(score[1])
                #manipulable.append(score[2])
        print 'Final version of scores is: \n', out_score[0]
        self.score_sheet = np.array(sorted(out_score, key=lambda t:t[6], reverse=True))
        self.score_length = len(self.score_sheet)
        print 'Best score and configuration is: \n', self.score_sheet[0]
        print 'Number of scores in score sheet: ', self.score_length

        print 'I have finished preparing the data for the task!'

        if self.score_sheet[0, 6] == 0:
            print 'There are no base locations with a score greater than 0. There are no good base locations!!'
            return None, None
                          
        print 'Time to adjust base scores: %fs' % (time.time()-start_time)
        
        # Published the wheelchair location to create a marker in rviz for visualization to compare where the service believes the wheelchair is to
        # where the person is (seen via kinect).
        pos_goal, ori_goal = Bmat_to_pos_quat(subject_location)
        self.publish_subject_marker(pos_goal, ori_goal)
        visualize_plot = False
        if visualize_plot:
            self.plot_final_score_sheet()

        return self.handle_returning_base_goals()
        
    def handle_returning_base_goals(self, data=None):
        if data != None:
            score_sheet = copy.copy(data)
        else:
            score_sheet = copy.copy(self.score_sheet)

        # now = rospy.Time.now() + rospy.Duration(1.0)
        # self.listener.waitForTransform('/odom_combined', '/base_link', now, rospy.Duration(10))
        # (trans, rot) = self.listener.lookupTransform('/odom_combined', '/base_link', now)
        # odom_B_pr2 = createBMatrix(trans, rot)

        best_score = score_sheet[0]

        pr2_base_output = []
        configuration_output = []

        # The output is a list of floats that are the position and quaternions for the transform from the goal location
        # to the ar tag. It also outputs a list of floats that is [robot z axis, bed height, head rest angle (degrees)].
        for i in xrange(len(best_score[0])):
            origin_B_goal = np.matrix([[m.cos(best_score[2][i]), -m.sin(best_score[2][i]), 0., best_score[0][i]],
                                       [m.sin(best_score[2][i]),  m.cos(best_score[2][i]), 0., best_score[1][i]],
                                       [0.,                      0.,                           1.,           0.],
                                       [0.,                      0.,                           0.,           1.]])
            pr2_B_goal = self.origin_B_pr2.I * origin_B_goal
            goal_B_ar = pr2_B_goal.I * self.pr2_B_ar
            pos_goal, ori_goal = Bmat_to_pos_quat(goal_B_ar)
            # odom_B_goal = odom_B_pr2 * self.origin_B_pr2.I * origin_B_goal
            # pos_goal, ori_goal = Bmat_to_pos_quat(odom_B_goal)
            pr2_base_output.append([pos_goal, ori_goal])
            configuration_output.append([best_score[3][i], best_score[4][i], np.degrees(best_score[5][i])])

            ## I no longer return posestamped messages. Now I return a list of floats.
            # psm = PoseStamped()
            # psm.header.frame_id = '/odom_combined'
            # psm.pose.position.x=pos_goal[0]
            # psm.pose.position.y=pos_goal[1]
            # psm.pose.position.z=pos_goal[2]
            # psm.pose.orientation.x=ori_goal[0]
            # psm.pose.orientation.y=ori_goal[1]
            # psm.pose.orientation.z=ori_goal[2]
            # psm.pose.orientation.w=ori_goal[3]
            # #print 'The quaternion to the goal location #',i,' is: \n',psm
            # output.append(psm)
        print 'Base selection service is done and has output a result.'
        ## Format of output is a list. Output is position [x,y,z] then quaternion [x,y,z,w] for each base location
        # (could output multiple base locations). So each set of 7 values is for one base location.
        return list(flatten(pr2_base_output)), list(flatten(configuration_output))
            
        # Visualize the solutions
        #with self.robot:
            #print 'checking goal base location: \n' , np.array(base_position)
            #sol = None
            #if not self.manip.CheckIndependentCollision(op.CollisionReport()):
                #for Tgrasp in self.Tgrasps:
                    #sol = self.manip.FindIKSolution(Tgrasp,filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                    #if sol is not None:
                        #self.robot.SetDOFValues(sol,self.manip.GetArmIndices())
                        #print 'displaying an IK solution!'
                #rospy.sleep(1.5)

 

            
            
            # Visualize the solutions
            #for Tgrasp in self.Tgrasps:
            #    sol = self.manip.FindIKSolution(Tgrasp,filteroptions=op.IkFilterOptions.CheckEnvCollisions)
            #    if sol is not None:
            #        self.robot.SetDOFValues(sol,self.manip.GetArmIndices())
            #        print 'displaying an IK solution!'
            #    rospy.sleep(1.5)
            
       
            # Commented out for testing. Uncomment FOR IT TO WORK!!!
            #now = rospy.Time.now() + rospy.Duration(1.0)
            #self.listener.waitForTransform('/odom_combined', '/base_link', now, rospy.Duration(10))
            #(trans,rot) = self.listener.lookupTransform('/odom_combined', '/base_link', now)

            #odom_goal = createBMatrix(trans, rot) * base_location
            #pos_goal = odom_goal[:3,3]
            #ori_goal = tr.matrix_to_quaternion(odom_goal[0:3,0:3])
            #print 'Got an iksolution! \n', sol
            #psm = PoseStamped()
            #psm.header.frame_id = '/odom_combined'
            #psm.pose.position.x=pos_goal[0]
            #psm.pose.position.y=pos_goal[1]
            #psm.pose.position.z=pos_goal[2]
            #psm.pose.orientation.x=ori_goal[0]
            #psm.pose.orientation.y=ori_goal[1]
            #psm.pose.orientation.z=ori_goal[2]
            #psm.pose.orientation.w=ori_goal[3]

                            # This is to publish WC position w.r.t. PR2 after the PR2 reaches goal location.
                            # Only necessary for testing in simulation to set the wheelchair in reach of PR2.
                            #goalpr2_B_wc = wc_B_goalpr2.I
                            #print 'pr2_B_wc is: \n',goalpr2_B_wc
                            #pos_goal = goalpr2_B_wc[:3,3]
                            #ori_goal = tr.matrix_to_quaternion(goalpr2_B_wc[0:3,0:3])
                            #psm_wc = PoseStamped()
                            #psm_wc.header.frame_id = '/odom_combined'
                            #psm_wc.pose.position.x=pos_goal[0]
                            #psm_wc.pose.position.y=pos_goal[1]
                            #psm_wc.pose.position.z=pos_goal[2]
                            #psm_wc.pose.orientation.x=ori_goal[0]
                            #psm_wc.pose.orientation.y=ori_goal[1]
                            #psm_wc.pose.orientation.z=ori_goal[2]
                            #psm_wc.pose.orientation.w=ori_goal[3]
                            #self.wc_position.publish(psm_wc)

        #print 'The quaternion to the goal location is: \n',psm

    def plot_final_score_sheet(self):
        visualize_plot = True
        if visualize_plot:
            # Plot the score as a scatterplot heat map
            #print 'score_sheet:',score_sheet
            score2d_temp = []
            #print t
            for i in np.arange(-1.5,1.55,.05):
                for j in np.arange(-1.5,1.55,.05):
                    temp = []
                    for item in score_sheet:
                    #print 'i is:',i
                    #print 'j is:',j
                        if item[0]==i and item[1]==j:
                            temp.append(item[3])
                    if temp != []:
                        score2d_temp.append([i,j,np.max(temp)])
    
            seen_items = []
            score2d = [] 
            for item in score2d_temp:
    
                #print 'seen_items is: ',seen_items
                #print 'item is: ',item
                #print (any((item == x) for x in seen_items))
                if not (any((item == x) for x in seen_items)):
                #if item not in seen_items:
                    #print 'Just added the item to score2d'
                    score2d.append(item)
                    seen_items.append(item)
            score2d = np.array(score2d)
            #print 'score2d with no repetitions',score2d
        
            fig, ax = plt.subplots()
                
            X  = score2d[:,0]
            Y  = score2d[:,1]
            #Th = score_sheet[:,2]
            c  = score2d[:,2]
            #surf = ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.5)
            surf = ax.scatter(X, Y, s=60,c=c,alpha=1)
            #surf = ax.scatter(X, Y,s=40, c=c,alpha=.6)
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            #ax.set_zlabel('Theta Axis')
       
            fig.colorbar(surf, shrink=0.5, aspect=5)
    
    
            verts_wc = [(-.438, -.32885), # left, bottom
                        (-.438, .32885), # left, top
                        (.6397, .32885), # right, top
                        (.6397, -.32885), # right, bottom
                        (0., 0.), # ignored
                        ]
            
            verts_pr2 = [(-1.5,  -1.5), # left, bottom
                         ( -1.5, -.835), # left, top
                         (-.835, -.835), # right, top
                         (-.835,  -1.5), # right, bottom
                         (   0.,    0.), # ignored
                         ]

            codes = [Path.MOVETO,
                     Path.LINETO,
                     Path.LINETO,
                     Path.LINETO,
                     Path.CLOSEPOLY,
                     ]
           
            path_wc = Path(verts_wc, codes)
            path_pr2 = Path(verts_pr2, codes)
    
            patch_wc = patches.PathPatch(path_wc, facecolor='orange', lw=2)        
            patch_pr2 = patches.PathPatch(path_pr2, facecolor='orange', lw=2)

            ax.add_patch(patch_wc)
            ax.add_patch(patch_pr2)
            ax.set_xlim(-2,2)
            ax.set_ylim(-2,2)
            plt.show()


                            #self.robot.SetDOFValues(sol,self.manip.GetArmIndices()) # set the current solution
                            #Tee = self.manip.GetEndEffectorTransform()
                            #self.env.UpdatePublishedBodies() # allow viewer to update new robot
#                            traj = None
#                            try:
#                                #res = self.manipprob.MoveToHandPosition(matrices=[np.array(pr2_B_goal)],seedik=10) # call motion planner with goal joint angles
#                                traj = self.manipprob.MoveManipulator(goal=sol, outputtrajobj=True)
#                                print 'Got a trajectory! \n'#,traj
#                            except:
#                                #print 'traj = \n',traj
#                                traj = None
#                                print 'traj failed \n'
#                                pass
#                            #traj =1 #This gets rid of traj
#                            if traj is not None:
        #else:
            #print 'I found a bad goal location. Trying again!'
                            #rospy.sleep(.1)
        #print 'I found nothing! My given inputs were: \n', req.task, req.head
        return None


    def load_task(self, task, model):

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('hrl_base_selection')
        
        return load_pickle(''.join([pkg_path,'/data/', task, '_', model, '_score_data.pkl']))



if __name__ == "__main__":
    #model = 'bed'
    rospy.init_node('select_base_server')
    selector = BaseSelector(testing=False)
    rospy.spin()


