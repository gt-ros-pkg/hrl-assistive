#!/usr/bin/env python

import numpy as np
import math as m
import openravepy as op

import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
import rospy, rospkg
import tf
from geometry_msgs.msg import PoseStamped

import hrl_lib.transforms as tr
from hrl_base_selection.srv import BaseMove
from visualization_msgs.msg import Marker
from helper_functions import createBMatrix


def handle_select_base(req):
    print 'My given inputs were: \n'
    print 'goal is: \n',req.goal
    print 'head is: \n', req.head
    pos_temp = [req.head.pose.position.x,req.head.pose.position.y,req.head.pose.position.z]
    ori_temp = [req.head.pose.orientation.x,req.head.pose.orientation.y,req.head.pose.orientation.z,req.head.pose.orientation.w]
    head = createBMatrix(pos_temp,ori_temp)
    print 'head from input: \n',head




    #(trans,rot) = listener.lookupTransform('/base_link', '/head_frame', rospy.Time(0))
    #pos_temp = trans
    #ori_temp = rot
    #head = createBMatrix(pos_temp,ori_temp)
    #print 'head from tf: \n',head

    pos_temp = [req.goal.pose.position.x,req.goal.pose.position.y,req.goal.pose.position.z]
    ori_temp = [req.goal.pose.orientation.x,req.goal.pose.orientation.y,req.goal.pose.orientation.z,req.goal.pose.orientation.w]
    goal = createBMatrix(pos_temp,ori_temp)
    print 'goal: \n',goal

    print 'I will move to be able to reach the mouth.'
    env = op.Environment()
    #env.SetViewer('qtcoin')
    env.Load('robots/pr2-beta-static.zae')
    robot = env.GetRobots()[0]
    v = robot.GetActiveDOFValues()
    v[robot.GetJoint('l_shoulder_pan_joint').GetDOFIndex()]= 3.14/2
    v[robot.GetJoint('r_shoulder_pan_joint').GetDOFIndex()] = -3.14/2
    v[robot.GetJoint('l_gripper_l_finger_joint').GetDOFIndex()] = .54
    v[robot.GetJoint('torso_lift_joint').GetDOFIndex()] = .3
    robot.SetActiveDOFValues(v)
    robot_start = np.matrix([[m.cos(0.),-m.sin(0.),0.,0],[m.sin(0.),m.cos(0.),0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    robot.SetTransform(np.array(robot_start))

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('hrl_base_selection')
    env.Load(''.join([pkg_path, '/models/ADA_Wheelchair.dae']))


    manip = robot.SetActiveManipulator('leftarm')
    ikmodel = op.databases.inversekinematics.InverseKinematicsModel(robot,iktype=op.IkParameterization.Type.Transform6D)
    manipprob = op.interfaces.BaseManipulation(robot) # create the interface for basic manipulation programs

    if not ikmodel.load():
        ikmodel.autogenerate()

    wheelchair = env.GetBodies()[1]
    wc_angle =  m.pi
    pr2_B_wc =   np.matrix([[     head[0,0],      head[0,1],              0.,      head[0,3]],
                            [     head[1,0],      head[1,1],              0.,      head[1,3]],
                            [            0.,             0.,              1.,             0.],
                            [            0.,             0.,              0.,             1]])

    corner_B_head = np.matrix([[m.cos(0.),-m.sin(0.),0.,.45],[m.sin(0.),m.cos(0.),0.,.34],[0.,0.,1,0.],[0.,0.,0.,1]])
    wheelchair_location = pr2_B_wc * corner_B_head.I
    wheelchair.SetTransform(np.array(wheelchair_location))

    pos_goal = [wheelchair_location[0,3],wheelchair_location[1,3],wheelchair_location[2,3]]
    ori_goal = tr.matrix_to_quaternion(wheelchair_location[0:3,0:3])
    
    marker = Marker()
    marker.header.frame_id = "/base_link"
    marker.header.stamp = rospy.Time()
    marker.ns = "base_service_wc_model"
    marker.id = 0
    #marker.type = Marker.SPHERE
    marker.type = Marker.MESH_RESOURCE;
    marker.action = Marker.ADD
    marker.pose.position.x = pos_goal[0]
    marker.pose.position.y = pos_goal[1]
    marker.pose.position.z = 0
    marker.pose.orientation.x = ori_goal[0]
    marker.pose.orientation.y = ori_goal[1]
    marker.pose.orientation.z = ori_goal[2]
    marker.pose.orientation.w = ori_goal[3]
    marker.scale.x = .025
    marker.scale.y = .025
    marker.scale.z = .025
    marker.color.a = 1.
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    #only if using a MESH_RESOURCE marker type:
    marker.mesh_resource = "package://hrl_base_selection/models/ADA_Wheelchair.dae"#~/git/gt-ros-pkg.hrl-assistive/hrl_base_selection/models/ADA_Wheelchair.dae" # ''.join([pkg_path, '/models/ADA_Wheelchair.dae']) #"package://pr2_description/meshes/base_v0/base.dae"

    for i in [0.,.1,.2,.3,-.1,-.2,-.3]:#[.1]:#[0.,.1,.3,.5,.8,1,-.1,-.2,-.3]:
        for j in [0.,.05,.1,.13,-.1,-.2,-.3]:#[.2]:#[0.,.1,.3,.5,.8,-.1,-.2,-.3]:
            for k in [0]:#[-m.pi/2]:#[0.,m.pi/4,m.pi/2,-m.pi/4,-m.pi/2]:

                #goal_pose = req.goal
                angle = m.pi
                goal_B_gripper =  np.matrix([[                 0,                  0,               1.,               .1],
                                             [                 0,                  1,               0.,               0.],
                                             [               -1.,                 0.,               0.,               0.],
                                             [                0.,                 0.,               0.,               1.]])
                #pr2_B_goal = head * head_B_goal
                pr2_B_goal = goal*goal_B_gripper
                angle_base = m.pi/2
                wc_B_goalpr2  =   np.matrix([[    m.cos(angle_base+k),     -m.sin(angle_base+k),                0.,              .4+i],
                                             [    m.sin(angle_base+k),      m.cos(angle_base+k),                0.,             -.8+j],
                                             [                     0.,                       0.,                1,                 0.],
                                             [                     0.,                       0.,                0.,                 1]])

                base_position = pr2_B_wc * wc_B_goalpr2
                #print 'base position: \n',base_position
                robot.SetTransform(np.array(base_position))

                #res = manipprob.MoveToHandPosition(matrices=[np.array(pr2_B_goal)],seedik=10) # call motion planner with goal joint angles
                #robot.WaitForController(0) # wait
                #print 'res: \n',res

                with env:
                    print 'checking goal: \n' , np.array(pr2_B_goal)
                    sol = manip.FindIKSolution(np.array(pr2_B_goal), op.IkFilterOptions.CheckEnvCollisions)
                    if sol != None:
                        #robot.SetDOFValues(sol,manip.GetArmIndices()) # set the current solution
                        #Tee = manip.GetEndEffectorTransform()
                        #env.UpdatePublishedBodies() # allow viewer to update new robot

                        print 'Got an iksolution! \n', sol
                        traj = None
                        try:
                            #res = manipprob.MoveToHandPosition(matrices=[np.array(pr2_B_goal)],seedik=10) # call motion planner with goal joint angles
                            traj=manipprob.MoveManipulator(goal=sol,outputtrajobj=True)
                            print 'Got a trajectory! \n'#,traj
                            print ''
                        except:
                            #print 'traj = \n',traj
                            traj = None
                            print 'traj failed \n'
                            pass
                        #traj =1 #This gets rid of traj
                        if (traj != None):
                            now = rospy.Time.now()
                            listener.waitForTransform('/odom_combined', '/base_link', now, rospy.Duration(10))
                            (trans,rot) = listener.lookupTransform('/odom_combined', '/base_link', now)

                            odom_goal = createBMatrix(trans,rot)*base_position
                            pos_goal = [odom_goal[0,3],odom_goal[1,3],odom_goal[2,3]]
                            ori_goal = tr.matrix_to_quaternion(odom_goal[0:3,0:3])
                            psm = PoseStamped()
                            psm.header.frame_id = '/odom_combined'
                            psm.pose.position.x=pos_goal[0]
                            psm.pose.position.y=pos_goal[1]
                            psm.pose.position.z=pos_goal[2]
                            psm.pose.orientation.x=ori_goal[0]
                            psm.pose.orientation.y=ori_goal[1]
                            psm.pose.orientation.z=ori_goal[2]
                            psm.pose.orientation.w=ori_goal[3]
                            print 'I found a goal location! It is at B transform: \n',base_position
                            print 'The quaternion to the goal location is: \n',psm
                            print 'The trajectory I found is : \n',traj
                            #srv.base_goal.
                            return psm
                        
                    else:
                        print 'I found a bad goal location. Trying again!'

    print 'I found nothing! My given inputs were: \n', req.goal, req.head
    return None

def select_base_server():
    s = rospy.Service('select_base_position', BaseMove, handle_select_base)
    print "Ready to select base."
    rospy.spin()

if __name__ == "__main__":

    rospy.init_node('select_base_server')

    vis_pub = rospy.Publisher("base_service_wc_model",Marker, latch = True)# .node_handle.advertise<visualization_msgs::Marker>( "visualization_marker", 0 );
    listener = tf.TransformListener()
    select_base_server()

















