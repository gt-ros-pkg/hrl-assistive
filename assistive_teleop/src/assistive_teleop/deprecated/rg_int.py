#!/usr/bin/python

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
from geometry_msgs.msg  import PoseStamped
from std_msgs.msg import Float32, String
from object_manipulation_msgs.msg import ReactiveGraspGoal, ReactiveGraspAction
import pose_utils
import actionlib

class ReactiveGraspIntermediary:
    def __init__(self):
        rospy.Subscriber('wt_rg_left_goal', PoseStamped, self.left_goal_cb)
        rospy.Subscriber('wt_rg_right_goal', PoseStamped, self.right_goal_cb)

        self.pose_test = rospy.Publisher('test_pose', PoseStamped, latch=True)

        self.rg_action_client = [actionlib.SimpleActionClient('reactive_grasp/left', ReactiveGraspAction),
                                 actionlib.SimpleActionClient('reactive_grasp/right', ReactiveGraspAction)]
        
        rospy.loginfo("Waiting for reactive_grasp/left server")
        if self.rg_action_client[0].wait_for_server(rospy.Duration(3)):
            rospy.loginfo("Found reactive_grasp/left server")
        else:
            rospy.logwarn("CANNOT FIND reactive_grasp/left server")
        
        rospy.loginfo("Waiting for reactive_grasp/right server")
        if self.rg_action_client[0].wait_for_server(rospy.Duration(3)):
            rospy.loginfo("Found reactive_grasp/right server")
        else:
            rospy.logwarn("CANNOT FIND reactive_grasp/right server")
        

    def right_goal_cb(self, ps):
        goal_pose = self.find_approach(ps)
        self.pose_test.publish(goal_pose)
        goal = ReactiveGraspGoal()
        goal.arm_name = 'right_arm'
        goal.final_grasp_pose = goal_pose
        self.rg_action_client[1].send_goal(goal)
        print "Sent Reactive Grasp Goal to Right Arm"

    def left_goal_cb(self, ps):
        goal_pose = self.find_approach(ps)
        goal = ReactiveGraspGoal()
        goal.arm_name = 'right_arm'
        goal.final_grasp_pose = goal_pose
        self.rg_action_client[0].send_goal(goal)
        print "Sent Reactive Grasp Goal to Left Arm"
        
    def find_approach(self, ps, stdoff=0.16):
        rotd_pose = pose_utils.rot_pose(ps, 0, -90, 0)
        transd_pose = pose_utils.pose_relative_move(rotd_pose, -stdoff, 0, 0)
        return transd_pose



if __name__=='__main__':
    rospy.init_node('wt_reactive_grasp_int')
    RGI = ReactiveGraspIntermediary()
    while not rospy.is_shutdown():
        rospy.spin()
