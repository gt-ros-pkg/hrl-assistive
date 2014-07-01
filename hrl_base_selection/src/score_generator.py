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
from hrl_base_selection.srv import BaseMove, BaseMove_multi
from visualization_msgs.msg import Marker
from helper_functions import createBMatrix
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import pickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle

class ScoreGenerator(object):

    def __init__(self,visualize=False,targets='all_goals',goals = None,model='chair'):
        self.visualize=visualize
        self.model=model
     
        # Sets the wheelchair location based on the location of the head using a few homogeneous transforms.
        if self.model == 'chair':
            pr2_B_head =   np.matrix([[       1.,        0.,   0.,         0.0],
                                      [       0.,        1.,   0.,         0.0],
                                      [       0.,        0.,   1.,         1.35],
                                      [       0.,        0.,   0.,         1.0]])

        if self.model == 'bed':
            an = -m.pi/2
            pr2_B_head = np.matrix([[ m.cos(an),  0., m.sin(an),     0.], #.45 #.438
                                    [        0.,  1.,        0.,     0.], #0.34 #.42
                                    [-m.sin(an),  0., m.cos(an), 1.1546],
                                    [        0.,  0.,        0.,     1.]])
        # Sets the wheelchair location based on the location of the head using a few homogeneous transforms.
        self.pr2_B_headfloor =   np.matrix([[       1.,        0.,   0.,         0.],
                                            [       0.,        1.,   0.,         0.],
                                            [       0.,        0.,   1.,         0.],
                                            [       0.,        0.,   0.,         1.]])
        # Gripper coordinate system has z in direction of the gripper, x is the axis of the gripper opening and closing.
        # This transform corrects that to make x in the direction of the gripper, z the axis of the gripper open. Centered at the very tip of the gripper.
        goal_B_gripper =  np.matrix([[  0.,  0.,   1.,   0.0],
                                     [  0.,  1.,   0.,   0.0],
                                     [ -1.,  0.,   0.,   0.0],
                                     [  0.,  0.,   0.,   1.0]])
        
        self.selection_mat = []
        self.Tgrasps = []
        self.weights = []
        self.goal_list = []
        if goals == None:
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
            for target in TARGETS:
                self.goal_list.append(pr2_B_head*createBMatrix(target[0],target[1])*goal_B_gripper)
            self.choose_task(targets)
        else:
            print 'Score generator received a list of desired goal locations. It contains ',len(goals),' goal locations.'
            for target in goals:
                self.goal_list.append(pr2_B_head*np.matrix(target[0])*goal_B_gripper)
                self.selection_mat.append(target[1])
            self.set_goals()
            #self.goal_list = goals

        
        

        self.setup_openrave()

    def set_goals(self):
        self.Tgrasps = []
        self.weights = []
        
        for num,selection in enumerate(self.selection_mat):
            if selection!=0:
                self.Tgrasps.append(np.array(self.goal_list[num]))
                self.weights.append(selection)


    def choose_task(self,task):
        if task == 'all_goals':
            self.selection_mat = np.ones(len(self.goal_list))
        elif task == 'wipe_face':
            self.selection_mat = np.array([1,1,1,1,1,1,0,0,0,0,0,0,0])
        elif task == 'shoulder':
            self.selection_mat = np.array([0,0,0,0,0,0,1,1,0,0,0,0])
        elif task == 'knee':
            self.selection_mat = np.array([0,0,0,0,0,0,0,0,1,1,0,0])
        elif task == 'arm':
            self.selection_mat = np.array([0,0,0,0,0,0,0,1,0,0,1,1])
        else:
            print 'Somehow I got a bogus task!? \n'
            return None
        self.set_goals()
        print 'The task was just set. We are generating score data for the task: ',task
        return self.selection_mat



    def handle_score(self):
        print 'Starting to generate the score. This is going to take a while. Estimated 60-100 seconds per goal location.'
        score_sheet = np.array([t for t in ( (i, j, k, self.generate_score(i,j,k))
                         for i in np.arange(-1.5,1.55,.05)
                             for j in np.arange(-1.5,1.55,.05)
                                 for k in np.arange(m.pi,-m.pi,-m.pi/4)
                                  )
                      ])
        return score_sheet
        



    

    def generate_score(self,i,j,k):
        base_position = self.pr2_B_headfloor * np.matrix([[ m.cos(k), -m.sin(k),     0.,         i],
                                                          [ m.sin(k),  m.cos(k),     0.,         j],
                                                          [       0.,        0.,     1.,        0.],
                                                          [       0.,        0.,     0.,        1.]])
        self.robot.SetTransform(np.array(base_position))
        reach_score = 0
        with self.robot:
            if not self.manip.CheckIndependentCollision(op.CollisionReport()):
                for num,Tgrasp in enumerate(self.Tgrasps):
                    sol = self.manip.FindIKSolution(Tgrasp,filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                    if sol is not None:
                        reach_score += self.weights[num]
        return reach_score


    def setup_openrave(self):
        # Setup Openrave ENV
        self.env = op.Environment()

        # Lets you visualize openrave. Uncomment to see visualization. Does not work through ssh.
        if self.visualize:
            self.env.SetViewer('qtcoin')


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

        ## Find and load Wheelchair Model
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('hrl_base_selection')

        # Transform from the coordinate frame of the wc model in the back right bottom corner, to the head location on the floor
        if self.model=='chair':
            self.env.Load(''.join([pkg_path, '/models/ADA_Wheelchair.dae']))
            originsubject_B_headfloor = np.matrix([[m.cos(0.), -m.sin(0.),  0.,     .45], #.45 #.438
                                                   [m.sin(0.),  m.cos(0.),  0.,  .32885], #0.34 #.42
                                                   [       0.,         0.,  1.,      0.],
                                                   [       0.,         0.,  0.,      1.]])
        elif self.model=='bed':
            self.env.Load(''.join([pkg_path, '/models/head_bed.dae']))
            an = 0#m.pi/2
            originsubject_B_headfloor = np.matrix([[ m.cos(an),  0., m.sin(an),  .2954], #.45 #.438
                                                   [        0.,  1.,        0.,     0.], #0.34 #.42
                                                   [-m.sin(an),  0., m.cos(an),     0.],
                                                   [        0.,  0.,        0.,     1.]])
        else:
            print 'I got a bad model. What is going on???'
            return None
        self.subject = self.env.GetBodies()[1]
        subject_location = self.pr2_B_headfloor * originsubject_B_headfloor.I
        self.subject.SetTransform(np.array(subject_location))

        print 'OpenRave has succesfully been initialized. \n'


if __name__ == "__main__":
    rospy.init_node('score_generator')
    mytask = 'shoulder'
    mymodel = 'bed'
    #mytask = 'all_goals'
    start_time = time.time()
    selector = ScoreGenerator(visualize=False,task=mytask,goals = None,model=mymodel)
    #selector.choose_task(mytask)
    score_sheet = selector.handle_score()
    
    print 'Time to load find generate all scores: %fs'%(time.time()-start_time)

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('hrl_base_selection')
    save_pickle(score_sheet,''.join([pkg_path, '/data/',mymodel,'_',mytask,'.pkl']))
    print 'Time to complete program, saving all data: %fs'%(time.time()-start_time)









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

    #print '2d score:',np.array(score2d_temp)

    seen_items = []
    score2d = [] 
    for item in score2d_temp:
#any((a == x).all() for x in my_list)
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

    if mymodel == 'chair':
        verts_subject = [(-.438, -.32885), # left, bottom
                         (-.438, .32885), # left, top
                         (.6397, .32885), # right, top
                         (.6397, -.32885), # right, bottom
                         (0., 0.), # ignored
                        ]
    elif mymodel == 'bed':
        verts_subject = [(-.2954, -.475), # left, bottom
                         (-.2954, .475), # left, top
                         (1.805, .475), # right, top
                         (1.805, -.475), # right, bottom
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
       
    path_subject = Path(verts_subject, codes)
    path_pr2 = Path(verts_pr2, codes)

    patch_subject = patches.PathPatch(path_subject, facecolor='orange', lw=2)        
    patch_pr2 = patches.PathPatch(path_pr2, facecolor='orange', lw=2)

    ax.add_patch(patch_subject)
    ax.add_patch(patch_pr2)
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)


    plt.show()

            

    

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X  = score_sheet[:,0]
    Y  = score_sheet[:,1]
    Th = score_sheet[:,2]
    c  = score_sheet[:,3]
    surf = ax.scatter(X, Y, Th,s=40, c=c,alpha=.6)
    #surf = ax.scatter(X, Y,s=40, c=c,alpha=.6)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Theta Axis')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
'''


    

