#!/usr/bin/python
import sys
import math
import numpy as np

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
import actionlib
from geometry_msgs.msg  import PoseStamped, Point
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import String, Float32, Bool
from tf import TransformListener, transformations
from pr2_controllers_msgs.msg import Pr2GripperCommand

from pixel_2_3d.srv import Pixel23d
from pr2_arms import PR2Arm_Planning
from pr2_gripper import PR2Gripper
import pose_utils as pu

class ArmIntermediary():
    def __init__(self, arm):
        self.arm = arm
        self.tfl = TransformListener()
        self.pr2_arm = PR2Arm_Planning(self.arm, self.tfl)
        self.pr2_gripper = PR2Gripper(self.arm)

        rospy.loginfo('Waiting for Pixel_2_3d Service')
        try:
            rospy.wait_for_service('/pixel_2_3d', 7.0)
            self.p23d_proxy = rospy.ServiceProxy('/pixel_2_3d', Pixel23d, True)
            rospy.loginfo("Found pixel_2_3d service")
        except:
            rospy.logwarn("Pixel_2_3d Service Not available")

        #Low-level motion requests: pass directly to pr2_arm
        rospy.Subscriber("wt_"+self.arm+"_arm_pose_commands", Point,
                         self.torso_frame_move)
        rospy.Subscriber("wt_"+self.arm+"_arm_angle_commands", JointTrajectoryPoint,
                         self.pr2_arm.send_traj_point)

        #More complex motion scripts, built here using pr2_arm functions 
        rospy.Subscriber("norm_approach_"+self.arm, PoseStamped, self.norm_approach)
        rospy.Subscriber("wt_grasp_"+self.arm+"_goal", PoseStamped, self.grasp)
        rospy.Subscriber("wt_wipe_"+self.arm+"_goals", PoseStamped, self.wipe)
        rospy.Subscriber("wt_swipe_"+self.arm+"_goals", PoseStamped, self.swipe)
        rospy.Subscriber("wt_lin_move_"+self.arm, Float32, self.hand_move)
        rospy.Subscriber("wt_adjust_elbow_"+self.arm, Float32, 
                        self.pr2_arm.adjust_elbow)
        rospy.Subscriber('wt_surf_wipe_right_points', Point, 
                        self.prep_surf_wipe)
        rospy.Subscriber("wt_poke_"+self.arm+"_point", PoseStamped, self.poke)
        rospy.Subscriber(rospy.get_name()+"/log_out", String, self.repub_log)
        rospy.Subscriber("wt_"+self.arm[0]+"_gripper_commands",
                        Pr2GripperCommand, self.gripper_pos)
        rospy.Subscriber("wt_"+self.arm[0]+"_gripper_grab_commands",
                        Bool, self.gripper_grab)
        rospy.Subscriber("wt_"+self.arm[0]+"_gripper_release_commands",
                        Bool, self.gripper_release)

        self.wt_log_out = rospy.Publisher("wt_log_out", String)

        self.wipe_started = False
        self.surf_wipe_started = False
        self.wipe_ends = [PoseStamped(), PoseStamped()]

    def repub_log(self, msg):
        self.wt_log_out.publish(msg)

    def gripper_pos(self, msg):
        self.pr2_gripper.gripper_action(msg.position, msg.max_effort)

    def gripper_grab(self, msg):
        self.pr2_gripper.grab()

    def gripper_release(self, msg):
        self.pr2_gripper.release()

    def torso_frame_move(self, msg):
        """Do linear motion relative to torso frame."""
        goal = self.pr2_arm.curr_pose()
        goal.pose.position.x += msg.x
        goal.pose.position.y += msg.y
        goal.pose.position.z += msg.z
        self.pr2_arm.blind_move(goal)

    def hand_move(self, f32):
        """Do linear motion relative to hand frame."""
        hfm_pose = self.pr2_arm.hand_frame_move(f32.data)
        self.pr2_arm.blind_move(hfm_pose)

    def norm_approach(self, pose):
        """ Safe move normal to surface pose at given distance."""
        appr = pu.find_approach(pose, 0.25)
        self.pr2_arm.move_arm_to(appr)
        
    def grasp(self, ps):
        """Do simple grasp: Normal approch, open, advance, close, retreat."""
        rospy.loginfo("Initiating Grasp Sequence")
        self.wt_log_out.publish(data="Initiating Grasp Sequence")
        approach = pose_utils.find_approach(ps, 0.15)
        rospy.loginfo("approach: \r\n %s" %approach)
        at_appr = self.pr2_arm.move_arm_to(approach)
        rospy.loginfo("arrived at approach: %s" %at_appr)
        if at_appr:
            opened = self.pr2_arm.gripper(0.09)
            if opened:
                rospy.loginfo("making linear approach")
                hfm_pose = pose_utils.find_approach(ps, -0.02) 
                self.pr2_arm.blind_move(hfm_pose)
                self.pr2_arm.wait_for_stop()
                closed = self.pr2_arm.gripper(0.0)
                if not closed:
                    rospy.loginfo("Couldn't close completely:\
                                    Grasp likely successful")
                hfm_pose = self.pr2_arm.hand_frame_move(-0.20) 
                self.pr2_arm.blind_move(hfm_pose)
        else:
            pass

    def prep_surf_wipe(self, point):
        pixel_u = point.x
        pixel_v = point.y
        test_pose = self.p23d_proxy(pixel_u, pixel_v).pixel3d
        test_pose = pose_utils.find_approach(test_pose, 0)
        (reachable, ik_goal) = self.pr2_arm.full_ik_check(test_pose)
        if reachable:
            if not self.surf_wipe_started:
                start_pose = test_pose
                self.surf_wipe_start = [pixel_u, pixel_v, start_pose]
                self.surf_wipe_started = True
                rospy.loginfo("Received valid starting position for wiping\
                                action")
                self.wt_log_out.publish(data="Received valid starting position\
                                            for wiping action")
                return #Return after 1st point, wait for second
            else:
                rospy.loginfo("Received valid ending position for wiping\
                                action")
                self.wt_log_out.publish(data="Received valid ending position\
                                            for wiping action")
                self.surf_wipe_started = False
        else:
            rospy.loginfo("Cannot reach wipe position, please try another")
            self.wt_log_out.publish(data="Cannot reach wipe position,\
                                            please try another")
            return #Return on invalid point, wait for another
        
        dist = self.pr2_arm.calc_dist(self.surf_wipe_start[2],test_pose)
        print 'dist', dist
        num_points = dist/0.02
        print 'num_points', num_points
        us = np.round(np.linspace(self.surf_wipe_start[0], pixel_u, num_points))
        vs = np.round(np.linspace(self.surf_wipe_start[1], pixel_v, num_points))
        surf_points = [PoseStamped() for i in xrange(len(us))]
        print "Surface Points", [us,vs]
        for i in xrange(len(us)):
            pose = self.p23d_proxy(us[i],vs[i]).pixel3d
            surf_points[i] = pose_utils.find_approach(pose,0)
            print i+1, '/', len(us)

        self.pr2_arm.blind_move(surf_points[0])
        self.pr2_arm.wait_for_stop()
        for pose in surf_points:
            self.pr2_arm.blind_move(pose,2.5)
            rospy.sleep(2)
        self.pr2_arm.hand_frame_move(-0.1)       

    def poke(self, ps):
        appr = pose_utils.find_approach(ps,0.15)
        prepared = self.pr2_arm.move_arm_to(appr)
        if prepared:
            pt1 = self.pr2_arm.hand_frame_move(0.155)
            self.pr2_arm.blind_move(pt1)
            self.pr2_arm.wait_for_stop()
            pt2 = self.pr2_arm.hand_frame_move(-0.155)
            self.pr2_arm.blind_move(pt2)

    def swipe(self, ps):
        traj = self.prep_wipe(ps)
        if traj is not None:
            self.wipe_move(traj, 1)

    def wipe(self, ps):
        traj = self.prep_wipe(ps)
        if traj is not None:
            self.wipe_move(traj, 4)

    def prep_wipe(self, ps):
        #print "Prep Wipe Received: %s" %pa
        print "Updating frame to: %s \r\n" %ps
        if not self.wipe_started:
            self.wipe_appr_seed = ps
            self.wipe_ends[0] = pose_utils.find_approach(ps, 0)
            print "wipe_end[0]: %s" %self.wipe_ends[0]
            (reachable, ik_goal) = self.pr2_arm.full_ik_check(self.wipe_ends[0])
            if not reachable:
                rospy.loginfo("Cannot find approach for initial wipe position,\
                                please try another")
                self.wt_log_out.publish(data="Cannot find approach for initial\
                                            wipe position, please try another")
                return None
            else:
                self.wipe_started = True
                rospy.loginfo("Received starting position for wiping action")
                self.wt_log_out.publish(data="Received starting position for\
                                                wiping action")
                return None
        else:
            self.wipe_ends[1] = pose_utils.find_approach(ps, 0)
            self.wipe_ends.reverse()
            (reachable, ik_goal) = self.pr2_arm.full_ik_check(self.wipe_ends[1])
            if not reachable:
                rospy.loginfo("Cannot find approach for final wipe position,\
                                please try another")
                self.wt_log_out.publish(data="Cannot find approach for final\
                                            wipe position, please try another")
                return None
            else:
                rospy.loginfo("Received End position for wiping action")
                self.wt_log_out.publish(data="Received End position for wiping\
                                            action")

                self.wipe_ends[1].header.stamp = rospy.Time.now()
                self.tfl.waitForTransform(self.wipe_ends[1].header.frame_id,
                                          'rh_utility_frame',
                                          rospy.Time.now(),
                                          rospy.Duration(3.0))
                fin_in_start = self.tfl.transformPose('rh_utility_frame',
                                                        self.wipe_ends[1])
                
                ang = math.atan2(-fin_in_start.pose.position.z,
                                    -fin_in_start.pose.position.y)+(math.pi/2)
                q_st_rot = transformations.quaternion_about_axis(ang, (1,0,0))
                q_st_new = transformations.quaternion_multiply(
                                    [self.wipe_ends[0].pose.orientation.x,
                                     self.wipe_ends[0].pose.orientation.y,
                                     self.wipe_ends[0].pose.orientation.z,
                                     self.wipe_ends[0].pose.orientation.w],
                                     q_st_rot)
                self.wipe_ends[0].pose.orientation = Quaternion(*q_st_new)
                self.wipe_ends[0].header.stamp = rospy.Time.now()
                self.tfl.waitForTransform(self.wipe_ends[0].header.frame_id,
                                            'rh_utility_frame',
                                            rospy.Time.now(),
                                            rospy.Duration(3.0))
                start_in_fin = self.tfl.transformPose('rh_utility_frame',
                                                        self.wipe_ends[0])
                ang = math.atan2(start_in_fin.pose.position.z,
                                start_in_fin.pose.position.y)+(math.pi/2)
                
                q_st_rot = transformations.quaternion_about_axis(ang, (1,0,0))
                q_st_new = transformations.quaternion_multiply(
                                        [self.wipe_ends[1].pose.orientation.x,
                                         self.wipe_ends[1].pose.orientation.y,
                                         self.wipe_ends[1].pose.orientation.z,
                                         self.wipe_ends[1].pose.orientation.w],
                                         q_st_rot)
                self.wipe_ends[1].pose.orientation = Quaternion(*q_st_new)
                
                appr = pose_utils.find_approach(self.wipe_appr_seed, 0.15)
                appr.pose.orientation = self.wipe_ends[1].pose.orientation
                prepared = self.pr2_arm.move_arm_to(appr)
                if prepared:
                    self.pr2_arm.blind_move(self.wipe_ends[1])
                    traj = self.pr2_arm.build_trajectory(self.wipe_ends[1],
                                                         self.wipe_ends[0])
                    wipe_traj = self.pr2_arm.build_follow_trajectory(traj)
                    self.pr2_arm.wait_for_stop()
                    self.wipe_started = False
                    return wipe_traj
                    #self.wipe(wipe_traj)
                else:
                    rospy.loginfo("Failure reaching start point,\
                                    please try again")
                    self.wt_log_out.publish(data="Failure reaching start\
                                                    point, please try again")
    
    def wipe_move(self, traj_goal, passes=4):
        times = []
        for i in range(len(traj_goal.trajectory.points)):
            times.append(traj_goal.trajectory.points[i].time_from_start)
        count = 0
        while count < passes:
            traj_goal.trajectory.points.reverse()
            for i in range(len(times)):
                traj_goal.trajectory.points[i].time_from_start = times[i]
            traj_goal.trajectory.header.stamp = rospy.Time.now()
            assert traj_goal.trajectory.points[0] != []
            self.pr2_arm.r_arm_follow_traj_client.send_goal(traj_goal)
            self.pr2_arm.r_arm_follow_traj_client.wait_for_result(
                                                    rospy.Duration(20))
            rospy.sleep(0.5)# Pause at end of swipe
            count += 1
        
        rospy.loginfo("Done Wiping")
        self.wt_log_out.publish(data="Done Wiping")
        hfm_pose = self.pr2_arm.hand_frame_move(-0.15)
        self.pr2_arm.blind_move(hfm_pose)

if __name__ == '__main__':
    arm = rospy.myargv(argv=sys.argv)[1]
    rospy.init_node('wt_'+arm+'_arm_intermediary')
    ai = ArmIntermediary(arm)
    while not rospy.is_shutdown():
        rospy.spin()
