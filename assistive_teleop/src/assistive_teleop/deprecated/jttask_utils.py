#!/usr/bin/python

import numpy as np
from copy import deepcopy
import math

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
import actionlib
from geometry_msgs.msg import PoseStamped, Point, Quaternion, WrenchStamped
from std_msgs.msg import Float64MultiArray
from tf import transformations, TransformListener

from assistive_teleop.srv import FrameUpdate, FrameUpdateRequest
from assistive_teleop.msg import FtMoveAction, FtMoveGoal, FtHoldAction, FtHoldGoal

TEST = True

class jt_task_utils():
    def __init__(self, tf=None):
        if tf is None:
            self.tf = TransformListener()
        else:
            self.tf = tf
        
        #### SERVICES ####
        rospy.loginfo("Waiting for utility_frame_services")
        try:
            rospy.wait_for_service('/l_utility_frame_update', 3.0)
            rospy.wait_for_service('/r_utility_frame_update', 3.0)
            self.update_frame = [rospy.ServiceProxy('/l_utility_frame_update', FrameUpdate),\
                                 rospy.ServiceProxy('/r_utility_frame_update', FrameUpdate)]

            rospy.loginfo("Found utility_frame_services")
        except:
            rospy.logwarn("Left or Right Utility Frame Service Not available")
        
        #### Action Clients ####
        self.ft_move_client = actionlib.SimpleActionClient('l_cart/ft_move_action', FtMoveAction)
        rospy.loginfo("Waiting for l_cart/ft_move_action server")
        if self.ft_move_client.wait_for_server(rospy.Duration(3)):
            rospy.loginfo("Found l_cart/ft_move_action server")
        else:
            rospy.logwarn("Cannot find l_cart/ft_move_action server")

        self.ft_move_r_client = actionlib.SimpleActionClient('r_cart/ft_move_action', FtMoveAction)
        rospy.loginfo("Waiting for r_cart/ft_move_action server")
        if self.ft_move_r_client.wait_for_server(rospy.Duration(3)):
            rospy.loginfo("Found r_cart/ft_move_action server")
        else:
            rospy.logwarn("Cannot find r_cart/ft_move_action server")

        self.ft_hold_client = actionlib.SimpleActionClient('ft_hold_action', FtHoldAction)
        rospy.loginfo("Waiting for ft_hold_action server")
        if self.ft_hold_client.wait_for_server(rospy.Duration(3)):
            rospy.loginfo("Found ft_hold_action server")
        else:
            rospy.logwarn("Cannot find ft_hold_action server")

        #### SUBSCRIBERS ####
        self.curr_state = [PoseStamped(), PoseStamped()]
        rospy.Subscriber('/l_cart/state/x', PoseStamped, self.get_l_state)
        rospy.Subscriber('/r_cart/state/x', PoseStamped, self.get_r_state)
        rospy.Subscriber('/wt_l_wrist_command', Point, self.rot_l_wrist)
        rospy.Subscriber('/wt_r_wrist_command', Point, self.rot_r_wrist)
        rospy.Subscriber('/wt_left_arm_pose_commands', Point, self.trans_l_hand)
        rospy.Subscriber('/wt_right_arm_pose_commands', Point, self.trans_r_hand)

     #   self.ft_wrench = WrenchStamped()
     #   self.force_stopped = False
     #   self.ft_z_thresh = -2
     #   self.ft_mag_thresh = 5
     #   rospy.Subscriber('ft_data_pm_adjusted', WrenchStamped, self.get_ft_state)
        
        #### PUBLISHERS ####
        self.goal_pub = [rospy.Publisher('l_cart/command_pose', PoseStamped),\
                         rospy.Publisher('r_cart/command_pose', PoseStamped)] 

        self.posture_pub = [rospy.Publisher('l_cart/command_posture', Float64MultiArray),\
                            rospy.Publisher('r_cart/command_posture', Float64MultiArray)]
        
        #### STATIC DATA ####
        self.postures = {
            'off': [],
            'mantis': [0, 1, 0,  -1, 3.14, -1, 3.14],
            'elbowupr': [-0.79,0,-1.6,  9999, 9999, 9999, 9999],
            'elbowupl': [0.79,0,1.6 , 9999, 9999, 9999, 9999],
            'old_elbowupr': [-0.79,0,-1.6, -0.79,3.14, -0.79,5.49],
            'old_elbowupl': [0.79,0,1.6, -0.79,3.14, -0.79,5.49],
            'elbowdownr': [-0.028262, 1.294634, -0.2578564, -1.549888, -31.27891385, -1.05276449, -1.8127318],
            'elbowdownl': [-0.008819572, 1.28348282, 0.2033844, -1.5565279, -0.09634, -1.023502, 1.799089]
        }

    def get_l_state(self, ps):#WORKING, TESTED
        self.curr_state[0] = ps

    def get_r_state(self, ps): #WORKING, TESTED
        self.curr_state[1] = ps

  #  def get_ft_state(self, ws):
  #      self.ft_wrench = ws
  #      self.ft_mag = math.sqrt(ws.wrench.force.x**2 + ws.wrench.force.y**2 + ws.wrench.force.z**2)
  #      if ws.wrench.force.z < self.ft_z_thresh:
  #          self.force_stopped = True
  #         # rospy.logwarn("Z force threshold exceeded")
  #      if self.ft_mag > self.ft_mag_thresh:
  #          self.force_stopped = True
  #          rospy.logwarn("Total force threshold exceeded")

    def rot_l_wrist(self, pt):
        out_pose = deepcopy(self.curr_state[0])
        q_r = transformations.quaternion_about_axis(pt.x, (1,0,0)) #Hand frame roll (hand roll)
        q_p = transformations.quaternion_about_axis(pt.y, (0,1,0)) #Hand frame pitch (wrist flex)
        q_h = transformations.quaternion_multiply(q_r, q_p)
        q_f = transformations.quaternion_about_axis(pt.y, (1,0,0)) #Forearm frame rot (forearm roll)
        
        if pt.x or pt.y:
            self.tf.waitForTransform(out_pose.header.frame_id, 'l_wrist_roll_link', out_pose.header.stamp, rospy.Duration(3.0))
            hand_pose = self.tf.transformPose('l_wrist_roll_link', out_pose)
            q_hand_pose = (hand_pose.pose.orientation.x, hand_pose.pose.orientation.y, hand_pose.pose.orientation.z, hand_pose.pose.orientation.w)
            q_h_rot = transformations.quaternion_multiply(q_h, hand_pose.pose.orientation)
            hand_pose.pose.orientation = Quaternion(*q_h_rot)
            out_pose = self.tf.transformPose(out_pose.header.frame_id, hand_pose)

        if pt.z:
            self.tf.waitForTransform(out_pose.header.frame_id, 'l_forearm_roll_link', out_pose.header.stamp, rospy.Duration(3.0))
            hand_pose = self.tf.transformPose('l_forearm_roll_link', out_pose)
            q_hand_pose = (hand_pose.pose.orientation.x, hand_pose.pose.orientation.y, hand_pose.pose.orientation.z, hand_pose.pose.orientation.w)
            q_f_rot = transformations.quaternion_multiply(q_f, hand_pose.pose.orientation)
            hand_pose.pose.orientation = Quaternion(*q_f_rot)
            out_pose = self.tf.transformPose(out_pose.header.frame_id, hand_pose)

        wrist_traj = self.build_trajectory(out_pose, arm=0)
        #TODO: Add Action Goal Sender to Move along trajectory!!!!!!!!!!!!!!!!

    def rot_r_wrist(self, pt):
        out_pose = deepcopy(self.curr_state[1])
        q_r = transformations.quaternion_about_axis(-pt.x, (1,0,0)) #Hand frame roll (hand roll)
        q_p = transformations.quaternion_about_axis(-pt.y, (0,1,0)) #Hand frame pitch (wrist flex)
        q_h = transformations.quaternion_multiply(q_r, q_p)
        q_f = transformations.quaternion_about_axis(-pt.y, (1,0,0)) #Forearm frame rot (forearm roll)
        
        if pt.x or pt.y:
            self.tf.waitForTransform(out_pose.header.frame_id, 'r_wrist_roll_link', out_pose.header.stamp, rospy.Duration(3.0))
            hand_pose = self.tf.transformPose('r_wrist_roll_link', out_pose)
            q_hand_pose = (hand_pose.pose.orientation.x, hand_pose.pose.orientation.y, hand_pose.pose.orientation.z, hand_pose.pose.orientation.w)
            q_h_rot = transformations.quaternion_multiply(q_h, hand_pose.pose.orientation)
            hand_pose.pose.orientation = Quaternion(*q_h_rot)
            out_pose = self.tf.transformPose(out_pose.header.frame_id, hand_pose)

        if pt.z:
            self.tf.waitForTransform(out_pose.header.frame_id, 'r_forearm_roll_link', out_pose.header.stamp, rospy.Duration(3.0))
            hand_pose = self.tf.transformPose('r_forearm_roll_link', out_pose)
            q_hand_pose = (hand_pose.pose.orientation.x, hand_pose.pose.orientation.y, hand_pose.pose.orientation.z, hand_pose.pose.orientation.w)
            q_f_rot = transformations.quaternion_multiply(q_f, hand_pose.pose.orientation)
            hand_pose.pose.orientation = Quaternion(*q_f_rot)
            out_pose = self.tf.transformPose(out_pose.header.frame_id, hand_pose)

        wrist_traj = self.build_trajectory(out_pose, arm=1)
        #TODO: Add Action Goal Sender to Move along trajectory!!!!!!!!!!!!!!!!

    def trans_l_hand(self, pt):
        print "Moving Left Hand with JT Task Controller"
        out_pose = PoseStamped()
        out_pose.header.frame_id = self.curr_state[0].header.frame_id
        out_pose.header.stamp = rospy.Time.now()
        out_pose.pose.position.x = self.curr_state[0].pose.position.x + pt.x
        out_pose.pose.position.y = self.curr_state[0].pose.position.y + pt.y
        out_pose.pose.position.z = self.curr_state[0].pose.position.z + pt.z
        out_pose.pose.orientation = self.curr_state[0].pose.orientation
        trans_traj = self.build_trajectory(out_pose, arm=0)
        self.ft_move_client.send_goal(FtMoveGoal(trans_traj,0., True))
        self.ft_move_client.wait_for_result(rospy.Duration(0.025*len(trans_traj)))

    def trans_r_hand(self, pt):
        out_pose = PoseStamped()
        out_pose.header.frame_id = self.curr_state[1].header.frame_id
        out_pose.header.stamp = rospy.Time.now()
        out_pose.pose.position.x = self.curr_state[1].pose.position.x + pt.x
        out_pose.pose.position.y = self.curr_state[1].pose.position.y + pt.y
        out_pose.pose.position.z = self.curr_state[1].pose.position.z + pt.z
        out_pose.pose.orientation = self.curr_state[1].pose.orientation
        trans_traj = self.build_trajectory(out_pose, arm=0)
        self.ft_move_client.send_goal(FtMoveGoal(trans_traj,0., True))
        self.ft_move_client.wait_for_result(rospy.Duration(0.025*len(trans_traj)))

    def send_posture(self, posture='off', arm=0 ): # WORKING, TESTED TODO: SLOW TRANSITION (if possible)
        if 'elbow' in posture:
            if arm == 0:
                posture = posture + 'l'
            elif arm == 1:
                posture = posture + 'r'
        self.posture_pub[arm].publish(Float64MultiArray(data=self.postures[posture]))

    def send_traj(self, poses, arm=0):
        send_rate = rospy.Rate(50)
        ##!!!!!!!!!!!!  MUST BALANCE SEND RATE WITH SPACING IN 'BUILD_TRAJECTORY' FOR CONTROL OF VELOCITY !!!!!!!!!!!!
        finished = False
        count = 0
        while not (rospy.is_shutdown() or finished):
            self.goal_pub[arm].publish(poses[count])
            count += 1
            send_rate.sleep()
            if count == len(poses):
                finished = True

    def send_traj_to_contact(self, poses, arm=0):
        send_rate = rospy.Rate(20)
        ##!!!!!!!!!!!!  MUST BALANCE SEND RATE WITH SPACING IN 'BUILD_TRAJECTORY' FOR CONTROL OF VELOCITY !!!!!!!!!!!!
        finished = False
        count = 0
        while not (rospy.is_shutdown() or finished or self.force_stopped):
            self.goal_pub[arm].publish(poses[count])
            count += 1
            send_rate.sleep()
            if count == len(poses):
                finished = True

    def build_trajectory(self, finish, start=None, arm=0, space = 0.001, steps=None): #WORKING, TESTED
    ##!!!!!!!!!!!!  MUST BALANCE SPACING WITH SEND RATE IN 'SEND_TRAJ' FOR CONTROL OF VELOCITY !!!!!!!!!!!!
        if start is None: # if given one pose, use current position as start
            start = self.curr_state[arm]

        dist = self.calc_dist(start,finish,arm=arm)     #Total distance to travel
        if steps is None:
            steps = int(math.ceil(dist/space))
        fracs = np.linspace(0, 1, steps)   #A list of fractional positions along course
        print "Steps: %s" %steps
        
        poses = [PoseStamped() for i in xrange(steps)]
        xs = np.linspace(start.pose.position.x, finish.pose.position.x, steps)
        ys = np.linspace(start.pose.position.y, finish.pose.position.y, steps)
        zs = np.linspace(start.pose.position.z, finish.pose.position.z, steps)
        
        qs = [start.pose.orientation.x, start.pose.orientation.y,
              start.pose.orientation.z, start.pose.orientation.w] 
        qf = [finish.pose.orientation.x, finish.pose.orientation.y,
              finish.pose.orientation.z, finish.pose.orientation.w] 
        
        for i,frac in enumerate(fracs):
            poses[i].header.stamp = rospy.Time.now()
            poses[i].header.frame_id = start.header.frame_id
            poses[i].pose.position = Point(xs[i], ys[i], zs[i])
            new_q = transformations.quaternion_slerp(qs,qf,frac)
            poses[i].pose.orientation = Quaternion(*new_q)
        #rospy.loginfo("Planning straight-line path, please wait")
        #self.wt_log_out.publish(data="Planning straight-line path, please wait")
        return poses
        
    def pose_frame_move(self, pose, x, y=0, z=0, arm=0): # FINISHED, UNTESTED
        self.update_frame[arm](pose)
        if arm == 0:
            frame = 'lh_utility_frame'
        elif arm == 1:
            frame = 'rh_utility_frame'
        pose.header.stamp = rospy.Time.now()
        self.tf.waitForTransform(pose.header.frame_id, frame , pose.header.stamp, rospy.Duration(3.0))
        framepose = self.tf.transformPose(frame, pose)
        framepose.pose.position.x += x
        framepose.pose.position.y += y
        framepose.pose.position.z += z
        self.dist = math.sqrt(x**2+y**2+z**2)
        self.tf.waitForTransform(frame, pose.header.frame_id, pose.header.stamp, rospy.Duration(3.0))
        return self.tf.transformPose(pose.header.frame_id, framepose)

    def calc_dist(self, ps1, ps2=None, arm=0): #FINISHED, UNTESTED
        if ps2 is None:
            ps2 = self.curr_pose()

        p1 = ps1.pose.position
        p2 = ps2.pose.position
        wrist_dist = math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

        self.update_frame[arm](ps2)
        ps2.header.stamp=rospy.Time(0)
        np2 = self.tf.transformPose('lh_utility_frame', ps2)
        np2.pose.position.x += 0.21
        self.tf.waitForTransform(np2.header.frame_id, 'torso_lift_link', rospy.Time.now(), rospy.Duration(3.0))
        p2 = self.tf.transformPose('torso_lift_link', np2)
        
        self.update_frame[arm](ps1)
        ps1.header.stamp=rospy.Time(0)
        np1 = self.tf.transformPose('lh_utility_frame', ps1)
        np1.pose.position.x += 0.21
        self.tf.waitForTransform(np1.header.frame_id, 'torso_lift_link', rospy.Time.now(), rospy.Duration(3.0))
        p1 = self.tf.transformPose('torso_lift_link', np1)
        
        p1 = p1.pose.position
        p2 = p2.pose.position
        finger_dist = math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)
        dist = max(wrist_dist, finger_dist)
        print 'Calculated Distance: ', dist
        return dist 
    
    def test(self):
        print "Testing..."
        rospy.sleep(1)
        #### TEST STATE GRABBING ####
        print "Left Current Pose:"
        print self.curr_state[0]
        print "Right Current Pose:"
        print self.curr_state[1]
        
        #### TEST FORCE STATE GRABBING ####
        print "Current Force Wrench:"
        print self.ft_wrench
        print "Current Force Magnitude:"
        print self.ft_mag

        #### TEST LEFT ARM GOAL SENDING ####
        l_pose = PoseStamped()
        l_pose.header.frame_id = 'torso_lift_link'
        l_pose.pose.position = Point(0.6, 0.3, 0.1)
        l_pose.pose.orientation = Quaternion(1,0,0,0)
        raw_input("send left arm goal")
        self.goal_pub[0].publish(l_pose)
        #### TEST RIGHT ARM GOAL SENDING
        #r_pose = PoseStamped()
        #r_pose.header.frame_id = 'torso_lift_link'
        #r_pose.pose.position = Point(0.6, -0.3, 0.1)
        #r_pose.pose.orientation = Quaternion(1,0,0,0)
        #raw_input("send right arm goal")
        #self.goal_pub[1].publish(r_pose)
        
        #### TEST POSE SETTING ####
       # raw_input("Left Elbow Up")
       # self.send_posture('elbowup',0)
       # raw_input("Right Elbow Up")
       # self.send_posture('elbowup',1)
       # raw_input("Left Elbow Down")
       # self.send_posture('elbowdown',0)
        #raw_input("Right Elbow Down")
        #self.send_posture('elbowdown',1)
        #raw_input("Both Postures Off")
        #self.send_posture(arm=0)
        #self.send_posture(arm=1)
        #print "Postures adjusted"

        #### TEST TRAJECTORY MOTION ####
        l_pose2 = PoseStamped()
        l_pose2.header.frame_id = 'torso_lift_link'
        l_pose2.pose.position = Point(0.8, 0.3, 0.1)
        l_pose2.pose.orientation = Quaternion(1,0,0,0)
        raw_input("Left trajectory")
        #l_pose2 = self.pose_frame_move(self.curr_state[0], -0.1, arm=0)
        traj = self.build_trajectory(l_pose2)
        self.send_traj_to_contact(traj)

        #r_pose2 = PoseStamped()
        #r_pose2.header.frame_id = 'torso_lift_link'
        #r_pose2.pose.position = Point(0.8, -0.15, -0.3)
        #r_pose2.pose.orientation = Quaternion(0,0.5,0.5,0)
        #raw_input("Right trajectory")
        #r_pose2 = self.pose_frame_move(self.curr_state[1], -0.1, arm=1)
        #traj = self.build_trajectory(r_pose2, arm=1)
        #self.send_traj(traj,1)

        #### RECONFIRM POSITION ####
        print "New Left Pose:"
        print self.curr_state[0]
        print "New Right Pose:"
        print self.curr_state[1]


if __name__ == '__main__':
    rospy.init_node('jttask_utils_test')
    jttu = jt_task_utils()
    if TEST:
        jttu.test()
    else:
        while not rospy.is_shutdown():
           rospy.spin()
