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
    def __init__(self, transform_listener=None):
        if transform_listener is None:
            self.listener = tf.TransformListener()
        self.vis_pub = rospy.Publisher("~wc_model", Marker, latch=True)
        
        # Publisher to let me test things with arm_reacher
        #self.wc_position = rospy.Publisher("~pr2_B_wc", PoseStamped, latch=True)

        # Service
        self.base_service = rospy.Service('select_base_position', BaseMove_multi, self.handle_select_base)
        
        # Subscriber to update robot joint state
        #self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)

        self.joint_names = []
        self.joint_angles = []
        self.selection_mat = np.zeros(11)


        self.setup_openrave()
        print "Ready to select base."
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




    def setup_openrave(self):
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


    def joint_state_cb(self, msg):
        #This gets the joint states of the entire robot.
        
        self.joint_angles = copy.copy(msg.position)
        self.joint_names = copy.copy(msg.name)


    # Publishes the wheelchair model location used by openrave to rviz so we can see how it overlaps with the real wheelchair
    def publish_wc_marker(self, pos, ori):
        marker = Marker()
        marker.header.frame_id = "/base_footprint"
        marker.header.stamp = rospy.Time()
        marker.ns = "base_service_wc_model"
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
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.mesh_resource = "package://hrl_base_selection/models/ADA_Wheelchair.dae"
        self.vis_pub.publish(marker)

    # Function that determines a good base location to be able to reach the goal location.
    #def handle_select_base(self, req):#, task):
    def handle_select_base(self, req):
        print 'I have received inputs!'

        #print 'My given inputs were: \n'
        #print 'goal is: \n',req.goal
        #print 'head is: \n', req.head

        # The head location is received as a posestamped message and is converted and used as the head location.
        pos_temp = [req.head.pose.position.x,
                    req.head.pose.position.y,
                    req.head.pose.position.z]
        ori_temp = [req.head.pose.orientation.x,
                    req.head.pose.orientation.y,
                    req.head.pose.orientation.z,
                    req.head.pose.orientation.w]
        head = createBMatrix(pos_temp, ori_temp)
        #print 'head from input: \n', head

        

        # This lets the service use TF to get the head location instead of requiring it as an input.
        #(trans,rot) = self.listener.lookupTransform('/base_link', '/head_frame', rospy.Time(0))
        #pos_temp = trans
        #ori_temp = rot
        #head = createBMatrix(pos_temp,ori_temp)
        #print 'head from tf: \n',head

        print 'I will now determine a good base location.'

        # Sets the wheelchair location based on the location of the head using a few homogeneous transforms.
        pr2_B_wc =   np.matrix([[head[0,0], head[0,1],   0.,  head[0,3]],
                                     [head[1,0], head[1,1],   0.,  head[1,3]],
                                     [       0.,        0.,   1.,         0.],
                                     [       0.,        0.,   0.,         1]])



        # Transform from the coordinate frame of the wc model in the back right bottom corner, to the head location
        corner_B_head = np.matrix([[m.cos(0.), -m.sin(0.),  0.,  .438],
                                   [m.sin(0.),  m.cos(0.),  0.,  .32885], #0.34
                                   [       0.,         0.,  1.,   0.],
                                   [       0.,         0.,  0.,   1.]])
        wheelchair_location = pr2_B_wc * corner_B_head.I
        self.wheelchair.SetTransform(np.array(wheelchair_location))


        # Get score data and convert to goal locations
        
        scores = self.load_task(req.task)
        if scores == None:
            print 'Failed to load precomputed reachability data. That is a problem. Abort!'
            return None
        temp_scores = []
        for score in scores:
            #if score[3]>0:
            wc_B_goal = np.matrix([[  m.cos(score[2]), -m.sin(score[2]),                0.,        score[0]],
                                       [  m.sin(score[2]),  m.cos(score[2]),                0.,        score[1]],
                                       [               0.,               0.,                1.,              0.],
                                       [               0.,               0.,                0.,              1.]])
            score[3]-=np.linalg.norm((pr2_B_wc*wc_B_goal)[0:2,3])
            if score[3]<0:
                score[3]=0
            temp_scores.append(score)
        if temp_scores == []:
            print 'None of the base positions checked can reach any goal location! We have no solution...'
            return None
        score_sheet = np.array(sorted(temp_scores, key=lambda t:t[3], reverse=True))
        print 'I have loaded the data for the task!'



        
        # Published the wheelchair location to create a marker in rviz for visualization to compare where the service believes the wheelchair is to
        # where the person is (seen via kinect).
        pos_goal = wheelchair_location[0:3,3]
        ori_goal = tr.matrix_to_quaternion(wheelchair_location[0:3,0:3])
        self.publish_wc_marker(pos_goal, ori_goal)

        #Setup for inside loop
        # Transforms from end of wrist of PR2 to the end of its gripper. Openrave has a weird coordinate system for the gripper so I use a transform to correct.
        goal_B_gripper =  np.matrix([[   0,   0,   1.,   .1],
                                     [   0,   1,   0.,   0.],
                                     [ -1.,  0.,   0.,   0.],
                                     [  0.,  0.,   0.,   1.]])
        #pr2_B_goal = head * head_B_goal
        goal_list = []
        for pose in self.POSES:
            goal_list.append(head*pose*goal_B_gripper)

        for goal in goal_list:
            self.Tgrasps.append(np.array(goal))
            #self.weights.append(selection)


        N = 10
        self.best_score = 0
        score = 0
        self.goals = []
        steps = 0
        starttime = time.time()
        timeout = 15 
        

        

        #angle_base = m.pi/2
        #np.abs(m.acos(op.matrixFromPose(pose)[0,0]))<m.pi/2:
        #for k in [0,m.pi/2,-m.pi/2,m.pi]:
        #    if np.abs(m.acos(k))<m.pi/2:

#                th.append(k)
#        print 'th: ',th
        #score_sheet = np.zeros([x_range,y_range,th_range,score])
        #for t in self.score_sheet:
        #    t[3] += np.linalg.norm(goal[0:2,3])
        print 'Time to load find a base location: %fs'%(time.time()-starttime)
        # Plot the score as a scatterplot heat map
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #X  = score_sheet[:,0]
        #Y  = score_sheet[:,1]
        #Th = score_sheet[:,2]
        #c  = score_sheet[:,3]
        #surf = ax.scatter(X, Y, Th,s=40, c=c,alpha=.6)
        #ax.set_xlabel('X Axis')
        #ax.set_ylabel('Y Axis')
        #ax.set_zlabel('Theta Axis')

        #fig.colorbar(surf, shrink=0.5, aspect=5)
        #plt.show()
        
        
        #print score_sheet
        #self.check(i,j,k)
        #    for i in xrange(x_range)
        #        for j in xrange(y_range)
        #            for k in [0]#,m.pi/2,-m.pi/2,m.pi]
        wc_B_goal = np.matrix([[  m.cos(score_sheet[0,2]), -m.sin(score_sheet[0,2]),                0.,  score_sheet[0,0]],
                               [  m.sin(score_sheet[0,2]),  m.cos(score_sheet[0,2]),                0.,  score_sheet[0,1]],
                               [                       0.,                       0.,                1.,                0.],
                               [                       0.,                       0.,                0.,                1.]])
        base_location = np.array(pr2_B_wc*wc_B_goal)
        self.robot.SetTransform(base_location)


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
        psm = PoseStamped()
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
        print 'I found a goal location! It is at B transform: \n',base_location
        print 'The quaternion to the goal location is: \n',psm

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


        return psm

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
        print 'I found nothing! My given inputs were: \n', req.task, req.head
        return None


    def load_task(self,task):

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('hrl_base_selection')
        
        return load_pickle(''.join([pkg_path,'/data/',task,'.pkl']))



if __name__ == "__main__":
    rospy.init_node('select_base_server')
    selector = BaseSelector()
    rospy.spin()


