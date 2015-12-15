#!/usr/bin/env python
import sys, optparse

import rospy, rospkg
import openravepy as op
import numpy as np
import math as m
import copy
from hrl_base_selection.srv import BaseMove_multi
import roslib; roslib.load_manifest('hrl_haptic_mpc')
import time
import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
import rospy, rospkg
import tf
roslib.load_manifest('hrl_base_selection')
import hrl_lib.transforms as tr
import tf
import rospy
from visualization_msgs.msg import Marker
import time
from hrl_msgs.msg import FloatArrayBare
from helper_functions import createBMatrix
from pr2_controllers_msgs.msg import SingleJointPositionActionGoal

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist

def select_base_client():
    angle = -m.pi/2
    pr2_B_head1  =  np.matrix([[   m.cos(angle),  -m.sin(angle),          0.,        0.],
                               [   m.sin(angle),   m.cos(angle),          0.,       2.5],
                               [             0.,             0.,          1.,       1.1546],
                               [             0.,             0.,          0.,        1.]])
    an = -m.pi/2
    pr2_B_head2 = np.matrix([[  m.cos(an),   0.,  m.sin(an),       0.],
                             [         0.,   1.,         0.,       0.], 
                             [ -m.sin(an),   0.,  m.cos(an),       0.],
                             [         0.,   0.,         0.,       1.]])
    pr2_B_head=pr2_B_head1*pr2_B_head2
    pos_goal = [pr2_B_head[0,3],pr2_B_head[1,3],pr2_B_head[2,3]]
    ori_goal = tr.matrix_to_quaternion(pr2_B_head[0:3,0:3])
    psm_head = PoseStamped()
    psm_head.header.frame_id = '/base_link'
    psm_head.pose.position.x=pos_goal[0]
    psm_head.pose.position.y=pos_goal[1]
    psm_head.pose.position.z=pos_goal[2]
    psm_head.pose.orientation.x=ori_goal[0]
    psm_head.pose.orientation.y=ori_goal[1]
    psm_head.pose.orientation.z=ori_goal[2]
    psm_head.pose.orientation.w=ori_goal[3]
    head_pub = rospy.Publisher("/haptic_mpc/head_pose", PoseStamped, latch=True)
    head_pub.publish(psm_head)

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('hrl_base_selection')
    
    
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
    vis_pub.publish( marker )
    print 'I think I just published the wc model \n'

    goal_angle = 0#m.pi/2
    pr2_B_goal  =  np.matrix([[    m.cos(goal_angle),     -m.sin(goal_angle),                0.,              2.6],
                              [    m.sin(goal_angle),      m.cos(goal_angle),                0.,              .3],
                              [                   0.,                     0.,                1.,             1.2],
                              [                   0.,                     0.,                0.,              1.]])
    pos_goal = [pr2_B_goal[0,3],pr2_B_goal[1,3],pr2_B_goal[2,3]]
    ori_goal = tr.matrix_to_quaternion(pr2_B_goal[0:3,0:3])
    psm_goal = PoseStamped()
    psm_goal.header.frame_id = '/base_link'
    psm_goal.pose.position.x=pos_goal[0]
    psm_goal.pose.position.y=pos_goal[1]
    psm_goal.pose.position.z=pos_goal[2]
    psm_goal.pose.orientation.x=ori_goal[0]
    psm_goal.pose.orientation.y=ori_goal[1]
    psm_goal.pose.orientation.z=ori_goal[2]
    psm_goal.pose.orientation.w=ori_goal[3]
    goal_pub = rospy.Publisher("/arm_reacher/goal_pose", PoseStamped, latch=True)
    goal_pub.publish(psm_goal)

    # task = 'shaving'
    task = 'yogurt'
    model = 'chair'

    start_time = time.time()
    rospy.wait_for_service('select_base_position')
    try:
        #select_base_position = rospy.ServiceProxy('select_base_position', BaseMove)
        #response = select_base_position(psm_goal, psm_head)
        select_base_position = rospy.ServiceProxy('select_base_position', BaseMove_multi)
        response = select_base_position(task, model)
        print 'response is: \n', response
        print 'Total time for service to return the response was: %fs' % (time.time()-start_time)
        return response.base_goal, response.configuration_goal#, response.ik_solution
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def publish_to_autobed(configuration_goals_list):
    autobed_pub = rospy.Publisher('/abdin0', FloatArrayBare, latch=True)
    autobed_goal = FloatArrayBare()
    autobed_goal.data = [configuration_goals_list[0][2], configuration_goals_list[0][1], 0.]
    autobed_pub.publish(autobed_goal)

def publish_to_zaxis(config):
    torso_lift_pub = rospy.Publisher('torso_controller/position_joint_action/goal',
                                     SingleJointPositionActionGoal, latch=True)
    torso_lift_msg = SingleJointPositionActionGoal()
    torso_lift_msg.goal.position = config[0][0]
    torso_lift_pub.publish(torso_lift_msg)

def usage():
    return "%s [current_loc goal head]"%sys.argv[0]

if __name__ == "__main__":
    rospy.init_node('client_node')
    vis_pub = rospy.Publisher("base_service_wc_model", Marker, latch=True)
    #if len(sys.argv) == 3:
    #    current_loc = PoseStamped(sys.argv[0])
    #    goal = PoseStamped(sys.argv[1])
    #    head = PoseStamped(sys.argv[2])
    #else:
    #    print usage()
    #    sys.exit(1)
    print "Requesting Base Goal Position"
    goal_array, config_array = select_base_client()
    base_goals = []
    configuration_goals = []
    for item in goal_array:
        base_goals.append(item)
    for item in config_array:
        configuration_goals.append(item)

    base_goals_list = []
    configuration_goals_list = []
    for i in xrange(int(len(base_goals)/7)):
        psm = PoseStamped()
        psm.header.frame_id = '/base_link'
        psm.pose.position.x = base_goals[int(0+7*i)]
        psm.pose.position.y = base_goals[int(1+7*i)]
        psm.pose.position.z = base_goals[int(2+7*i)]
        psm.pose.orientation.x = base_goals[int(3+7*i)]
        psm.pose.orientation.y = base_goals[int(4+7*i)]
        psm.pose.orientation.z = base_goals[int(5+7*i)]
        psm.pose.orientation.w = base_goals[int(6+7*i)]
        psm.header.frame_id = '/base_link'
        base_goals_list.append(copy.copy(psm))
        configuration_goals_list.append([configuration_goals[0+3*i], configuration_goals[1+3*i]+9.+10,
                                         configuration_goals[2+3*i]+40.])

    publish_to_autobed(configuration_goals_list)
    publish_to_zaxis(configuration_goals_list)

    print "Base Goal Position is:\n", base_goals_list[0]
    print "Config Goal is:\n", configuration_goals_list[0]
    rospy.spin()
    #print "ik solution is: \n", ik
