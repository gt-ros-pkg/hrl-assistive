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

from sensor_msgs.msg import JointState
from std_msgs.msg import String
import hrl_lib.transforms as tr
from hrl_base_selection.srv import BaseMove
from visualization_msgs.msg import Marker
from helper_functions import createBMatrix
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint



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
        self.base_service = rospy.Service('select_base_position', BaseMove, self.handle_select_base)
        
        # Subscriber to update robot joint state
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)

        self.joint_names = []
        self.joint_angles = []
        self.selection_mat = np.zeros(11)


        self.setup_openrave()
        print "Ready to select base."


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
        self.irmodel = op.databases.inversereachability.InverseReachabilityModel(robot=self.robot)
        print 'loading irmodel'
        starttime = time.time()
        if not self.irmodel.load():            
            print 'do you want to generate irmodel for your robot? it might take several hours'
            print 'or you can go to http://people.csail.mit.edu/liuhuan/pr2/openrave/openrave_database/ to get the database for PR2'
            input = raw_input('[Y/n]\n')
            if input == 'y' or input == 'Y' or input == '\n' or input == '':
                self.irmodel.autogenerate()
                self.irmodel.load()
            else:
                raise ValueError('')
        print 'time to load inverse-reachability model: %fs'%(time.time()-starttime)
        # make sure the robot and manipulator match the database
        assert self.irmodel.robot == self.robot and self.irmodel.manip == self.robot.GetActiveManipulator()   

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
    def handle_select_base(self, req):#, task):
    def handle_select_base(self, req):#, task):
        #choose_task(req.task)
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

        # The goal location is received as a posestamped message and is converted and used as the goal location.
        pos_temp = [req.goal.pose.position.x,
                    req.goal.pose.position.y,
                    req.goal.pose.position.z]
        ori_temp = [req.goal.pose.orientation.x,
                    req.goal.pose.orientation.y,
                    req.goal.pose.orientation.z,
                    req.goal.pose.orientation.w]
        goal = createBMatrix(pos_temp,ori_temp)
        #print 'goal: \n',goal

        print 'I will move to be able to reach the goal.'

        # Sets the wheelchair location based on the location of the head using a few homogeneous transforms.
        pr2_B_wc =   np.matrix([[head[0,0], head[0,1],   0.,  head[0,3]],
                                [head[1,0], head[1,1],   0.,  head[1,3]],
                                [       0.,        0.,   1.,         0.],
                                [       0.,        0.,   0.,         1]])

        # Transform from the coordinate frame of the wc model in the back right bottom corner, to the head location
        corner_B_head = np.matrix([[m.cos(0.), -m.sin(0.),  0.,  .45],
                                   [m.sin(0.),  m.cos(0.),  0.,  .42], #0.34
                                   [       0.,         0.,  1.,   0.],
                                   [       0.,         0.,  0.,   1.]])
        wheelchair_location = pr2_B_wc * corner_B_head.I
        self.wheelchair.SetTransform(np.array(wheelchair_location))

        
        # Published the wheelchair location to create a marker in rviz for visualization to compare where the service believes the wheelchair is to
        # where the person is (seen via kinect).
        pos_goal = wheelchair_location[0:3,3]

        ori_goal = tr.matrix_to_quaternion(wheelchair_location[0:3,0:3])
        self.publish_wc_marker(pos_goal, ori_goal)

        #Setup for inside loop
        angle = m.pi

        # Transforms from end of wrist of PR2 to the end of its gripper. Openrave has a weird coordinate system for the gripper so I use a transform to correct.
        goal_B_gripper =  np.matrix([[   0,   0,   1.,   .1],
                                     [   0,   1,   0.,   0.],
                                     [ -1.,  0.,   0.,   0.],
                                     [  0.,  0.,   0.,   1.]])
        #pr2_B_goal = head * head_B_goal
        pr2_B_goal = goal*goal_B_gripper
        angle_base = m.pi/2
        #print 'The goal gripper pose is: \n' , np.array(pr2_B_goal)

        # Find a base location using Inverse Reachability
        Tgrasp = np.array(copy.copy(pr2_B_goal))
        densityfn,samplerfn,bounds = self.irmodel.computeBaseDistribution(Tgrasp,logllthresh=1.8)
        if densityfn == None:
            print 'The specified grasp is not reachable using IR method! \n'
            return None
        N = 10
        goals = []
        numfailures = 0
        starttime = time.time()
        timeout = float('inf')
        with self.robot:
            while len(goals) < N and numfailures<50000:
                #print numfailures
                if time.time()-starttime > timeout:
                    break
                poses,jointstate = samplerfn(20*N)
                #poses,jointstate = samplerfn(N-len(goals))
                for pose in poses:
                    #print 'pose is: \n',pose
                    loc = op.matrixFromPose(pose)
                    #print 'B transform is: \n',loc
                    self.robot.SetTransform(loc)
                    self.robot.SetDOFValues(*jointstate)
                    #print op.matrixFromPose(pose), '\n'
                    #print m.acos(op.matrixFromPose(pose)[0,0]),'\n'
                    #rospy.sleep(.1)
                    #print 'Made it past sleep'
                    # validate that base is not in collision
                    if not self.manip.CheckIndependentCollision(op.CollisionReport()):
                        q = self.manip.FindIKSolution(Tgrasp,filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                        if q is not None:
                            if np.abs(m.acos(op.matrixFromPose(pose)[0,0]))<m.pi/4:
                                values = self.robot.GetDOFValues()
                                values[self.manip.GetArmIndices()] = q
                                goals.append((Tgrasp,pose,values))
                            else:
                                numfailures += 1
                        elif self.manip.FindIKSolution(Tgrasp,0) is None:
                            numfailures += 1
                            #print 'failure: \n',numfailures
                    else:
                        numfailures += 1
                        #print 'collision failure! \n', numfailures
        #print 'showing %d results'%N
        goal = goals[0]
        print 'total failed base positions: \n', numfailures
        Tgrasp_best,pose_best,values_best = copy.copy(goal)
        #print 'Tgrasp_best is: \n',Tgrasp_best
        #print 'pose_best is: \n',pose_best
        #print 'values_best is: \n',values_best
        for ind,goal in enumerate(goals):
            #raw_input('press ENTER to show goal %d'%ind)
            #print 'Shoinw show goal %d \n'%ind
            Tgrasp,pose,values = copy.copy(goal)
            if np.linalg.norm(op.matrixFromPose(pose)[0:3,3])<np.linalg.norm(op.matrixFromPose(pose_best)[0:3,3]):
                Tgrasp_best = Tgrasp
                pose_best = pose
                values_best = values
            #self.robot.SetTransform(pose)
            #self.robot.SetDOFValues(values)
        base_location = op.matrixFromPose(pose_best)
        self.robot.SetTransform(op.matrixFromPose(pose_best))
        self.robot.SetActiveDOFValues(values_best)
        
        print 'The best base location according the inverse reachability is: \n', base_location

        now = rospy.Time.now() + rospy.Duration(1.0)
        self.listener.waitForTransform('/odom_combined', '/base_link', now, rospy.Duration(10))
        (trans,rot) = self.listener.lookupTransform('/odom_combined', '/base_link', now)

        odom_goal = createBMatrix(trans, rot) * base_location
        pos_goal = odom_goal[:3,3]
        ori_goal = tr.matrix_to_quaternion(odom_goal[0:3,0:3])
        #print 'Got an iksolution! \n', sol
        psm = PoseStamped()
        psm.header.frame_id = '/odom_combined'
        psm.pose.position.x=pos_goal[0]
        psm.pose.position.y=pos_goal[1]
        psm.pose.position.z=pos_goal[2]
        psm.pose.orientation.x=ori_goal[0]
        psm.pose.orientation.y=ori_goal[1]
        psm.pose.orientation.z=ori_goal[2]
        psm.pose.orientation.w=ori_goal[3]

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
        return psm
        
    def choose_task(task):
        if task == 'wipe_face':
            self.selection_mat = np.array([1,1,1,1,1,0,0,0,0,0,0])
        elif task == 'shoulder':
            self.selection_mat = np.array([0,0,0,0,0,1,1,0,0,0,0])
        elif task == 'knee':
            self.selection_mat = np.array([0,0,0,0,0,0,0,1,1,0,0])
        elif task == 'hand':
            self.selection_mat = np.array([0,0,0,0,0,0,1,0,0,1,1])
        else:
            print 'Somehow I got a bogus task!? \n'
            return None
        return self.selection_mat
'''
        # Find a base location by testing various base locations online for IK solution
        for i in [0.,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,-.05,-.1,-.15,-.2,-.25,-.3]:#[.1]:#[0.,.1,.3,.5,.8,1,-.1,-.2,-.3]:
            for j in [0.,.03,.05,.08,-.03,-.05,-.08, -.1,-.12,-.2,-.3]:#[.2]:#[0.,.1,.3,.5,.8,-.1,-.2,-.3]:
                for k in [0]:#[-m.pi/2]:#[0.,m.pi/4,m.pi/2,-m.pi/4,-m.pi/2]:
                    #goal_pose = req.goal
                    # transform from head frame in wheelchair to desired base goal
                    wc_B_goalpr2  =   np.matrix([[m.cos(angle_base+k), -m.sin(angle_base+k),   0.,  .4+i],
                                                 [m.sin(angle_base+k),  m.cos(angle_base+k),   0., -.9+j],
                                                 [                 0.,                   0.,    1,    0.],
                                                 [                 0.,                   0.,   0.,     1]])

                    base_position = pr2_B_wc * wc_B_goalpr2
                    #print 'base position: \n',base_position
                    self.robot.SetTransform(np.array(base_position))
                    #res = self.manipprob.MoveToHandPosition(matrices=[np.array(pr2_B_goal)],seedik=10) # call motion planner with goal joint angles
                    #self.robot.WaitForController(0) # wait
                    #print 'res: \n',res

                    v = self.robot.GetActiveDOFValues()
                    for name in self.joint_names:
                        v[self.robot.GetJoint(name).GetDOFIndex()] = self.joint_angles[self.joint_names.index(name)]
                    self.robot.SetActiveDOFValues(v)

                    with self.env:
                        #print 'checking goal base location: \n' , np.array(base_position)
                        sol = self.manip.FindIKSolution(np.array(pr2_B_goal), op.IkFilterOptions.CheckEnvCollisions)
                        if sol is not None:
                            now = rospy.Time.now() + rospy.Duration(1.0)
                            self.listener.waitForTransform('/odom_combined', '/base_link', now, rospy.Duration(10))
                            (trans,rot) = self.listener.lookupTransform('/odom_combined', '/base_link', now)

                            odom_goal = createBMatrix(trans, rot) * base_position
                            pos_goal = odom_goal[:3,3]
                            ori_goal = tr.matrix_to_quaternion(odom_goal[0:3,0:3])
                            #print 'Got an iksolution! \n', sol
                            psm = PoseStamped()
                            psm.header.frame_id = '/odom_combined'
                            psm.pose.position.x=pos_goal[0]
                            psm.pose.position.y=pos_goal[1]
                            psm.pose.position.z=pos_goal[2]
                            psm.pose.orientation.x=ori_goal[0]
                            psm.pose.orientation.y=ori_goal[1]
                            psm.pose.orientation.z=ori_goal[2]
                            psm.pose.orientation.w=ori_goal[3]

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
                            print 'I found a goal location! It is at B transform: \n',base_position
                            print 'The quaternion to the goal location is: \n',psm
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
                        else:
                            print 'I found a bad goal location. Trying again!'
                            #rospy.sleep(.1)
        print 'I found nothing! My given inputs were: \n', req.goal, req.head
'''
if __name__ == "__main__":
    rospy.init_node('select_base_server')
    selector = BaseSelector()
    rospy.spin()
