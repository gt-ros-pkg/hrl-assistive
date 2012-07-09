#!/usr/bin/python

import roslib; roslib.load_manifest('web_teleop_trunk')
import rospy
import actionlib
import math
import numpy as np
from copy import deepcopy
from collections import deque
import l_arm_utils
from geometry_msgs.msg  import PoseStamped, WrenchStamped, Point, Quaternion
import arm_navigation_msgs.msg
import arm_navigation_msgs.msg
from kinematics_msgs.srv import GetKinematicSolverInfo, GetPositionFK, GetPositionFKRequest, GetPositionIK, GetPositionIKRequest
from pr2_controllers_msgs.msg import JointTrajectoryAction, JointTrajectoryControllerState, JointTrajectoryActionGoal, SingleJointPositionAction, SingleJointPositionGoal, Pr2GripperCommandActionGoal, JointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from std_msgs.msg import String, Float32, Bool
from tf import TransformListener, transformations
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTolerance
from web_teleop_trunk.srv import FrameUpdate
from pixel_2_3d.srv import Pixel23d


class ArmActions():

    def __init__(self):
        rospy.init_node('left_arm_actions')
        
        self.tfl = TransformListener()
        self.aul = l_arm_utils.ArmUtils(self.tfl)
        #self.fth = ft_handler.FTHandler()

        rospy.loginfo("Waiting for l_utility_frame_service")
        try:
            rospy.wait_for_service('/l_utility_frame_update', 7.0)
            self.aul.update_frame = rospy.ServiceProxy('/l_utility_frame_update', FrameUpdate)
            rospy.loginfo("Found l_utility_frame_service")
        except:
            rospy.logwarn("Left Utility Frame Service Not available")

        rospy.loginfo('Waiting for Pixel_2_3d Service')
        try:
            rospy.wait_for_service('/pixel_2_3d', 7.0)
            self.p23d_proxy = rospy.ServiceProxy('/pixel_2_3d', Pixel23d, True)
            rospy.loginfo("Found pixel_2_3d service")
        except:
            rospy.logwarn("Pixel_2_3d Service Not available")

        #Low-level motion requests: pass directly to arm_utils
        rospy.Subscriber('wt_left_arm_pose_commands', Point, self.torso_frame_move)
        rospy.Subscriber('wt_left_arm_angle_commands', JointTrajectoryPoint, self.aul.send_traj_point)

        #More complex motion scripts, defined here using arm_util functions 
        rospy.Subscriber('norm_approach_left', PoseStamped, self.norm_approach)
        rospy.Subscriber('wt_grasp_left_goal', PoseStamped, self.grasp)
        #rospy.Subscriber('wt_wipe_left_goals', PoseStamped, self.wipe)
        rospy.Subscriber('wt_wipe_left_goals', PoseStamped, self.force_wipe_agg)
        rospy.Subscriber('wt_swipe_left_goals', PoseStamped, self.swipe)
        rospy.Subscriber('wt_lin_move_left', Float32, self.hand_move)
        rospy.Subscriber('wt_adjust_elbow_left', Float32, self.aul.adjust_elbow)
        rospy.Subscriber('wt_surf_wipe_left_points', Point, self.force_surf_wipe)
        rospy.Subscriber('wt_poke_left_point', PoseStamped, self.poke)
        rospy.Subscriber('wt_contact_approach_left', PoseStamped, self.approach_to_contact)

        self.wt_log_out = rospy.Publisher('wt_log_out', String)
        self.test_pose = rospy.Publisher('test_pose', PoseStamped, latch=True) 
        self.say = rospy.Publisher('text_to_say', String) 

        self.wipe_started = False
        self.surf_wipe_started = False
        self.wipe_ends = [PoseStamped(), PoseStamped()]
        
        #FORCE_TORQUE ADDITIONS
        
        #rospy.Subscriber('pr2_netft_zeroer/wrench_zeroed', WrenchStamped, self.ft_preprocess)
        #self.rezero_wrench = rospy.Publisher('pr2_netft_zeroer/rezero_wrench', Bool)
        rospy.Subscriber('ft_data_pm_adjusted', WrenchStamped, self.ft_preprocess)
        rospy.Subscriber('wt_ft_goal', Float32, self.set_force_goal)
        
        self.wt_force_out = rospy.Publisher('wt_force_out', Float32)
        
        self.ft_rate_limit = rospy.Rate(30)

        self.ft = WrenchStamped()
        self.ft_mag = 0.
        self.ft_mag_que = deque([0]*10,10)
        self.ft_sm_mag = 0.
        self.ft_case = None
        self.force_goal_mean =3 #1.42
        self.force_goal_std= 0.625
        self.stop_maintain = False
        self.force_wipe_started = False
        self.force_wipe_start = PoseStamped()
        self.force_wipe_appr = PoseStamped()
    
    def set_force_goal(self, msg):
        self.force_goal_mean = msg.data

    def ft_preprocess(self, ft):
        self.ft = ft
        self.ft_mag = math.sqrt(ft.wrench.force.x**2 + ft.wrench.force.y**2 + ft.wrench.force.z**2)
        self.ft_mag_que.append(self.ft_mag)
        self.ft_sm_mag = np.mean(self.ft_mag_que)
        #print 'Force Magnitude: ', self.ft_mag
        self.wt_force_out.publish(self.ft_mag)

    def approach_to_contact(self, ps, overshoot=0.05):
            ps.pose.position.z += 0.02
            self.stop_maintain = True
            self.aul.update_frame(ps)
            appr = self.aul.find_approach(ps, 0.15)
            goal = self.aul.find_approach(ps, -overshoot)
            (appr_reachable, ik_goal) = self.aul.full_ik_check(appr)
            (goal_reachable, ik_goal) = self.aul.full_ik_check(goal)
            if appr_reachable and goal_reachable:
                traj = self.aul.build_trajectory(goal,appr,tot_points=200)
                #prep = self.aul.move_arm_to(appr)
                self.aul.blind_move(appr, 3)
                rospy.sleep(3)
                if True: # if prep:
                    self.adjust_forearm(traj.points[0].positions)
                    #self.rezero_wrench.publish(data=True)
                    curr_traj_point = self.advance_to_contact(traj)
                    if not curr_traj_point is None:
                        self.maintain_norm_force2(traj, curr_traj_point)
                        #self.maintain_force_position(self.aul.hand_frame_move(0.05))
                        #self.twist_wipe();  self.aul.blind_move(appr)
                    else:
                        self.aul.send_traj_point(traj.points[0], 4)
                        
            else:
                rospy.loginfo("Cannot reach desired 'move-to-contact' point")
                self.wt_log_out.publish(data="Cannot reach desired 'move-to-contact' point")

    def advance_to_contact(self, traj):
        self.stop_maintain = False
        curr_traj_point = 0
        advance_rate = rospy.Rate(10)
        while not (rospy.is_shutdown() or self.stop_maintain):
            if not (curr_traj_point >= len(traj.points)):
                self.aul.send_traj_point(traj.points[curr_traj_point], 0.3)
                curr_traj_point += 1
                advance_rate.sleep()
            else:
                rospy.loginfo("Moved past expected contact, but no contact found! Returning to start")
                self.wt_log_out.publish(data="Moved past expected contact, but no contact found! Returning to start")
                self.stop_maintain = True
                return None
            if self.ft_mag > 2.5:
                self.stop_maintain = True
                print "Contact Detected"
                return curr_traj_point
   
        
    def maintain_norm_force2(self, traj, curr_traj_point=0, mean=3, std=1):
        self.stop_maintain = False
        maintain_rate = rospy.Rate(100)
        while not (rospy.is_shutdown() or self.stop_maintain):
            if self.ft_mag > 12:
                rospy.loginfo("Force Too High, ending behavior")
                self.wt_log_out.publish(data="Force too high, ending behavior")
                break
            #print "mean: ", mean, "stds: ", std, "force: ", self.ft_mag
            if self.ft_mag < mean - std:
                curr_traj_point += 1
                curr_traj_point = np.clip(curr_traj_point, 0, len(traj.points))
                #print curr_traj_point
                print "Low"
                if not (curr_traj_point >= len(traj.points)):
                    self.aul.send_traj_point(traj.points[curr_traj_point], 0.3)
                    rospy.sleep(0.3)
                #else:
                #    rospy.loginfo("Force too low, but extent of the trajectory is reached")
                #    self.wt_log_out.publish(data="Force too low, but extent of the trajectory is reached")
                #    self.stop_maintain = True
            elif self.ft_mag > mean + std:
                print "High"
                steps = int(round((self.ft_mag/std)))
                curr_traj_point -= steps
                #print curr_traj_point
                curr_traj_point = np.clip(curr_traj_point, 0, len(traj.points))
                if curr_traj_point >= 0:
                    self.aul.send_traj_point(traj.points[curr_traj_point], 0.3)
                    rospy.sleep(0.3)
                else:
                    rospy.loginfo("Beginning of trajectory reached, cannot back away further")
                    self.wt_log_out.publish(data="Beginning of trajectory reached, cannot back away further")
                    #self.stop_maintain = True
            maintain_rate.sleep()
            mean = self.force_goal_mean
            std = self.force_goal_std
        print "Returning to start position"
        self.aul.send_traj_point(traj.points[0], 4)

    def maintain_norm_force(self, traj, curr_traj_point=0, mean=0, std=1):
        self.stop_maintain = False
        maintain_rate = rospy.Rate(100)
        while not (rospy.is_shutdown() or self.stop_maintain):
            #print "mean: ", mean, "stds: ", std, "force: ", self.ft_mag
            if self.ft_mag < mean - std:
                curr_traj_point += 1
                np.clip(curr_traj_point, 0, len(traj.points))
                if not (curr_traj_point  >= len(traj.points)):
                    print curr_traj_point
                    self.aul.send_traj_point(traj.points[curr_traj_point], 0.1)
                    rospy.sleep(0.1)
                else:
                    rospy.loginfo("Force too low, but extent of the trajectory is reached")
                    self.wt_log_out.publish(data="Force too low, but extent of the trajectory is reached")
                    stopped = True
            elif self.ft_mag > mean + std:
                curr_traj_point -= 1
                np.clip(curr_traj_point, 0, len(traj.points))
                if curr_traj_point >= 0:
                    print curr_traj_point
                    self.aul.send_traj_point(traj.points[curr_traj_point], 0.1)
                    rospy.sleep(0.1)
                else:
                    rospy.loginfo("Beginning of trajectory reached, cannot back away further")
                    self.wt_log_out.publish(data="Beginning of trajectory reached, cannot back away further")
                    stopped = True
            maintain_rate.sleep()
            mean = self.force_goal_mean
            std = self.force_goal_std

#    def maintain_net_force(self, mean=0, std=3):
#        self.stop_maintain = False
#        maintain_rate = rospy.Rate(100)
#        while not (rospy.is_shutdown() or self.stop_maintain):
#            if self.ft_mag > mean + 8:
#                curr_angs = self.aul.joint_state_act.positions
#                try:
#                    x_plus = np.subtract(self.aul.ik_pose_proxy(self.aul.form_ik_request(self.aul.hand_frame_move(0.02))).solution.joint_state.position, curr_angs)
#                    x_minus = np.subtract(self.aul.ik_pose_proxy(self.aul.form_ik_request(self.aul.hand_frame_move(-0.02))).solution.joint_state.position, curr_angs)
#                    y_plus = np.subtract(self.aul.ik_pose_proxy(self.aul.form_ik_request(self.aul.hand_frame_move(0, 0.02, 0))).solution.joint_state.position, curr_angs)
#                    y_minus = np.subtract(self.aul.ik_pose_proxy(self.aul.form_ik_request(self.aul.hand_frame_move(0, -0.02, 0))).solution.joint_state.position, curr_angs)
#                    z_plus = np.subtract(self.aul.ik_pose_proxy(self.aul.form_ik_request(self.aul.hand_frame_move(0, 0, 0.02))).solution.joint_state.position, curr_angs)
#                    z_minus = np.subtract(self.aul.ik_pose_proxy(self.aul.form_ik_request(self.aul.hand_frame_move(0, 0, -0.02))).solution.joint_state.position, curr_angs)
#                    #print 'x: ', x_plus,'\r\n', x_minus
#                    #print 'y: ', y_plus,'\r\n', y_minus
#                    #print 'z: ', z_plus,'\r\n', z_minus
#                    ft_sum = self.ft_mag
#                    parts = np.divide([self.ft.wrench.force.x, self.ft.wrench.force.y, self.ft.wrench.force.z], ft_sum)
#                    print 'parts', parts
#                    ends = [[x_plus,x_minus],[y_plus, y_minus],[z_plus,z_minus]]
#                    side = [[0]*7]*3
#                    for i, part in enumerate(parts):
#                        if part >=0:
#                            side[i] = ends[i][0]
#                        else:
#                            side[i] = ends[i][1]
#
#                    ang_out = curr_angs
#                    for i in range(3):
#                        ang_out -= np.average(side, 0, parts)
#                except:
#                    print 'Near Joint Limits!'
#                self.aul.send_joint_angles(ang_out)
#
#                #print 'x: ', x_plus, x_minus
#            maintain_rate.sleep()
#            mean = self.force_goal_mean
#            std = self.force_goal_std

    def maintain_force_position(self,pose = None, mean=3, std=1):
        self.stop_maintain = False
        if pose is None:
            goal = self.aul.curr_pose()
        else:
            goal = pose
        goal_ort = [goal.pose.orientation.x,goal.pose.orientation.y,goal.pose.orientation.z,goal.pose.orientation.w]
        error = PoseStamped()
        maintain_rate = rospy.Rate(250)
        while not (rospy.is_shutdown() or self.stop_maintain):
            current = self.aul.curr_pose()
            current_ort =  [current.pose.orientation.x, current.pose.orientation.y, current.pose.orientation.z, current.pose.orientation.w]
            error.pose.position.x = current.pose.position.x - goal.pose.position.x
            error.pose.position.y = current.pose.position.y - goal.pose.position.y
            error.pose.position.z = current.pose.position.z - goal.pose.position.z
            error_mag = math.sqrt(error.pose.position.x**2 + error.pose.position.y**2 + error.pose.position.z**2) 
            #out = deepcopy(goal)
            out = PoseStamped()
            out.header.frame_id = goal.header.frame_id
            out.header.stamp = rospy.Time.now()
            out.pose.position = Point(goal.pose.position.x, goal.pose.position.y, goal.pose.position.z)
            self.test_pose.publish(out)
            if all(np.array(self.ft_mag_que) < mean - std) and error_mag > 0.005:
                #print 'Force Too LOW'
                out.pose.position.x += 0.990*error.pose.position.x
                out.pose.position.y += 0.990*error.pose.position.y
                out.pose.position.z += 0.990*error.pose.position.z
                ori = transformations.quaternion_slerp(goal_ort, current_ort, 0.990)
                out.pose.orientation = Quaternion(*ori)
                self.aul.fast_move(out,0.0038)
            elif all(np.array(self.ft_mag_que) > mean + std):
                #print 'Moving to avoid force'
                current.pose.position.x += self.ft.wrench.force.x/9000
                current.pose.position.y += self.ft.wrench.force.y/9000
                current.pose.position.z += self.ft.wrench.force.z/9000
                self.aul.fast_move(current,0.0038)
            else:
                pass
                #print "Force in desired range"
            mean = self.force_goal_mean
            std = self.force_goal_std
            #rospy.sleep(0.001)
            maintain_rate.sleep()

    #def maintain_force_position(self,pose = None, mean=3, std=1):
    #    self.stop_maintain = False
    #    if pose is None:
    #        goal = self.aul.curr_pose()
    #    else:
    #        goal = pose
    #    self.rezero_wrench.publish(data=True)
    #    maintain_rate = rospy.Rate(500)
    #    kp = 0.07
    #    kd = 0.03
    #    ki = 0.0
    #    error = PoseStamped()
    #    old_error = PoseStamped()
    #    sum_error_x = deque([0]*10,10)
    #    sum_error_y = deque([0]*10,10)
    #    sum_error_z = deque([0]*10,10)
    #    while not (rospy.is_shutdown() or self.stop_maintain):
    #        current = self.aul.curr_pose()
    #        error.pose.position.x = current.pose.position.x - goal.pose.position.x
    #        error.pose.position.y = current.pose.position.y - goal.pose.position.y
    #        error.pose.position.z = current.pose.position.z - goal.pose.position.z
    #        sum_error_x.append(error.pose.position.x)
    #        sum_error_y.append(error.pose.position.y)
    #        sum_error_z.append(error.pose.position.z)
    #        print "Force: ",  self.ft_sm_mag, min(self.ft_mag_que), max(self.ft_mag_que)
    #        print "Error: ", error.pose.position
    #        print self.ft_mag_que, mean-std
    #        print self.ft_mag_que < mean-std
    #        break
    #        if all([self.ft_mag_que < mean - std]):
    #            print 'Force Too LOW'
    #            current.pose.position.x += kp*error.pose.position.x + kd*(error.pose.position.x - old_error.pose.position.x) + ki*np.sum(sum_error_x)
    #            current.pose.position.x += kp*error.pose.position.y + kd*(error.pose.position.y - old_error.pose.position.y) + ki*np.sum(sum_error_y)
    #            current.pose.position.x += kp*error.pose.position.z + kd*(error.pose.position.z - old_error.pose.position.z) + ki*np.sum(sum_error_z)
    #            self.aul.fast_move(current, 0.02)
    #            self.test_pose.publish(current)

    #        if all([i > mean + std for i in self.ft_mag_que]):
    #            print 'Moving to avoid force'
    #            current.pose.position.x += self.ft.wrench.force.x/10000
    #            current.pose.position.y += self.ft.wrench.force.y/10000
    #            current.pose.position.z += self.ft.wrench.force.z/10000
    #            self.aul.fast_move(current, 0.02)
    #            self.test_pose.publish(current)


    #        old_error = error
    #        mean = self.force_goal_mean
    #        std = self.force_goal_std
    #        maintain_rate.sleep()

    def maintain_net_force(self, mean=3, std=1):
        self.stop_maintain = False
        maintain_rate = rospy.Rate(500)
        #self.rezero_wrench.publish(data=True)
        while not (rospy.is_shutdown() or self.stop_maintain):
            if self.ft_mag > mean + 3:
                print 'Moving to avoid force'
                print "Force: ",  self.ft_mag
                goal = self.aul.curr_pose()
                goal.pose.position.x += np.clip(self.ft.wrench.force.x/5000, -0.001, 0.001)
                goal.pose.position.y += np.clip(self.ft.wrench.force.y/5000, -0.001, 0.001)
                goal.pose.position.z += np.clip(self.ft.wrench.force.z/5000, -0.001, 0.001)
                self.test_pose.publish(goal)

                self.aul.fast_move(goal, 0.02)

                mean = self.force_goal_mean
                std = self.force_goal_std
                maintain_rate.sleep()

    def mannequin(self):
        mannequin_rate=rospy.Rate(100)
        pose = PoseStamped()
        while not rospy.is_shutdown():
            #joints = np.add(np.array(self.aul.joint_state_act.positions), np.clip(np.array(self.aul.joint_state_err.positions), -0.05, 0.05))
            joints = self.aul.joint_state_act.positions
            print joints
            #raw_input('Review Joint Angles')
            self.aul.send_joint_angles(joints, 0.00001)
            pose.header.stamp = rospy.Time.now()
            self.test_pose.publish(pose)
            mannequin_rate.sleep()

    def force_wipe_agg(self, ps):
        ps.pose.position.z += 0.02
        self.aul.update_frame(ps)
        rospy.sleep(0.1)
        pose = self.aul.find_approach(ps, 0)
        (goal_reachable, ik_goal) = self.aul.ik_check(pose)
        if goal_reachable:
            if not self.force_wipe_started:
                appr = self.aul.find_approach(ps, 0.20)
                (appr_reachable, ik_goal) = self.aul.ik_check(appr)
                self.test_pose.publish(appr)
                if appr_reachable:
                    self.force_wipe_start = pose
                    self.force_wipe_appr = appr
                    self.force_wipe_started = True
                else:
                    rospy.loginfo("Cannot reach approach point, please choose another")
                    self.wt_log_out.publish(data="Cannot reach approach point, please choose another")
                    self.say.publish(data="I cannot get to a safe approach for there, please choose another point")
            else:
                ps1, ps2 = self.align_poses(self.force_wipe_start, pose)
                self.force_wipe_prep(ps1, ps2)
               # self.force_wipe_prep(self.force_wipe_start, pose)
                self.force_wipe_started = False
        else: 
            rospy.loginfo("Cannot reach wiping point, please choose another")
            self.wt_log_out.publish(data="Cannot reach wiping point, please choose another")
            self.say.publish(data="I cannot reach there, please choose another point")

    def force_wipe_prep(self, ps_start, ps_finish, travel = 0.05):
        ps_start.header.stamp = rospy.Time.now()
        ps_finish.header.stamp = rospy.Time.now()
        ps_start_far = self.aul.hand_frame_move(travel, 0, 0, ps_start)
        ps_start_near = self.aul.hand_frame_move(-travel, 0, 0, ps_start)
        ps_finish_far = self.aul.hand_frame_move(travel, 0, 0, ps_finish)
        ps_finish_near = self.aul.hand_frame_move(-travel, 0, 0, ps_finish)
        n_points = int(math.ceil(self.aul.calc_dist(ps_start, ps_finish)*9000))
        print 'n_points: ', n_points
        mid_traj = self.aul.build_trajectory(ps_finish, ps_start, tot_points=n_points, jagged=False)
        near_traj = self.aul.build_trajectory(ps_finish_near, ps_start_near, tot_points=n_points, jagged=False)
        far_traj = self.aul.build_trajectory(ps_finish_far, ps_start_far, tot_points=n_points, jagged=False)
        self.force_wipe(mid_traj, near_traj, far_traj)

    def force_surf_wipe(self, point):
        self.fsw_poses = self.prep_surf_wipe(point)
        if not self.fsw_poses is None:
            near_poses = far_poses = [PoseStamped() for i in xrange(len(self.fsw_poses))]
            for i, p in enumerate(self.fsw_poses):
                near_poses[i]=self.aul.pose_frame_move(p, -0.05)
                far_poses[i]=self.aul.pose_frame_move(p, 0.05)
            near_traj = self.aul.fill_ik_traj(near_poses)
            mid_traj = self.aul.fill_ik_traj(self.fsw_poses)
            far_traj = self.aul.fill_ik_traj(far_poses)
            print 'Trajectories Found'
            self.force_wipe(mid_traj, near_traj, far_traj)

    def adjust_forearm(self, in_angles):
        print 'cur angles: ', self.aul.joint_state_act.positions, 'angs: ', in_angles
        print np.abs(np.subtract(self.aul.joint_state_act.positions, in_angles))
        if np.max(np.abs(np.subtract(self.aul.joint_state_act.positions,in_angles)))>math.pi:
            self.say.publish(data="Adjusting for-arm roll")
            print "Evasive Action!"
            angles = list(self.aul.joint_state_act.positions)
            flex = in_angles[5]
            angles[5] = -0.1
            self.aul.send_joint_angles(angles, 4)
            angles[4] = in_angles[4]
            self.aul.send_joint_angles(angles,6)
            angles[5] = flex
            self.aul.send_joint_angles(angles, 4)

    def force_wipe(self, mid_traj, near_traj, far_traj):
        near_angles = [list(near_traj.points[i].positions) for i in range(len(near_traj.points))]
        mid_angles = [list(mid_traj.points[i].positions) for i in range(len(mid_traj.points))]
        far_angles = [list(far_traj.points[i].positions) for i in range(len(far_traj.points))]
        print 'lens: nmf: ', len(near_angles), len(mid_angles), len(far_angles)
        fmn_diff = np.abs(np.subtract(near_angles, far_angles))
        fmn_max = np.max(fmn_diff, axis=0)
        print 'fmn_max: ', fmn_max
        if any(fmn_max >math.pi/2):
            rospy.loginfo("TOO LARGE OF AN ANGLE CHANGE ALONG GRADIENT, IK REGION PROBLEMS!")
            self.wt_log_out.publish(data="The path requested required large movements (IK Limits cause angle wrapping)")
            self.say.publish(data="Large motion detected, cancelling. Please try again.")
            return
        for i in range(7):
            n_max =  max(np.abs(np.diff(near_angles,axis=0)[i]))
            m_max = max(np.abs(np.diff(mid_angles,axis=0)[i]))
            f_max = max(np.abs(np.diff(far_angles,axis=0)[i]))
            n_mean = 4*np.mean(np.abs(np.diff(near_angles,axis=0)[i]))
            m_mean = 4*np.mean(np.abs(np.diff(mid_angles,axis=0)[i]))
            f_mean = 4*np.mean(np.abs(np.diff(far_angles,axis=0)[i]))               
            print 'near: max: ', n_max, 'mean: ', n_mean
            print 'mid: max: ', m_max, 'mean: ', m_mean 
            print 'far: max: ', f_max, 'mean: ', f_mean
            if (n_max >= n_mean) or (m_max >= m_mean) or (f_max >= f_mean):
                rospy.logerr("TOO LARGE OF AN ANGLE CHANGE ALONG PATH, IK REGION PROBLEMS!")
                self.wt_log_out.publish(data="The path requested required large movements (IK Limits cause angle wrapping)")
                self.say.publish(data="Large motion detected, cancelling. Please try again.")
                return
        near_diff = np.subtract(near_angles, mid_angles).tolist()
        far_diff = np.subtract(far_angles, mid_angles).tolist()
        self.say.publish(data="Moving to approach point")
        appr = self.force_wipe_appr
        appr.pose.orientation = self.aul.get_fk(near_angles[0]).pose.orientation
        #prep = self.aul.move_arm_to(appr)
        self.aul.blind_move(appr)
        #rospy.sleep(3)
        if True: #prep:
            self.adjust_forearm(near_angles[0])    
            self.say.publish(data="Making Approach.")
            bias = 2
            self.aul.send_joint_angles(np.add(mid_angles[0],np.multiply(bias, near_diff[0])), 3.5)
            #self.rezero_wrench.publish(data=True)
            rospy.sleep(1)
            wipe_rate = rospy.Rate(1000)
            self.stop_maintain = False
            count = 0
            lap = 0
            max_laps = 4
            mean = self.force_goal_mean
            std = self.force_goal_std
            self.say.publish(data="Wiping")
            single_dir = False#True
            time = rospy.Time.now().to_sec()
            while not (rospy.is_shutdown() or self.stop_maintain) and (count + 1 <= len(mid_angles)) and (lap < max_laps):
                if self.ft_mag > 10:
                    angles_out = np.add(mid_angles[0], np.multiply(2, near_diff[0]))
                    self.aul.send_joint_angles(angles_out,2)
                    rospy.loginfo("Force Too High, ending behavior")
                    self.wt_log_out.publish(data="Force too high, ending behavior")
                    break
                #print "mean: ", mean, "std: ", std, "force: ", self.ft_mag
                if self.ft_mag >= mean + std:
                #    print 'Force too high!'
                    bias += (self.ft_mag/500)
                elif self.ft_mag < mean - std:
                #    print 'Force too low'
                    bias -= max(0.003,(self.ft_mag/1500))
                else:
                #    print 'Force Within Range'
                    count += 1
                bias = np.clip(bias, -1, 2)   
                if bias > 0.:
                    diff = near_diff[count]
                else:
                    diff = far_diff[count]
                angles_out = np.add(mid_angles[count], np.multiply(abs(bias), diff))
                self.aul.send_joint_angles(angles_out, 0.0025)
                #rospy.sleep(0.0000i1)
                #print "Rate: ", (1/ (rospy.Time.now().to_sec() - time))
                #time = rospy.Time.now().to_sec()
                wipe_rate.sleep()
                
                mean = self.force_goal_mean
                std = self.force_goal_std
                if count + 1>= len(mid_angles):
                    if single_dir:
                        bias = 1
                        pose = self.aul.curr_pose()
                        pose = self.aul.hand_frame_move(-0.01)
                        rot = transformations.quaternion_about_axis(math.radians(-10), (0,1,0))
                        q = transformations.quaternion_multiply([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w],rot)
                        pose.pose.orientation = Quaternion(*q)
                        self.aul.blind_move(pose)
                        #goal = self.aul.ik_pose_proxy(self.aul.form_ik_request(pose))
                        #if goal.error_code.val == 1:
                         #   self.aul.send_angles_wrap(goal.solution.joint_state.position)
                        #angles_out = list(self.aul.joint_state_act.positions)
                        #angles_out[4] += 0.05
                       # self.aul.send_joint_angles(angles_out,3)
                        angles_out = np.add(mid_angles[count], np.multiply(bias, near_diff[count]))
                        self.aul.send_joint_angles(angles_out,5)
                        angles_out = np.add(mid_angles[0], np.multiply(bias, near_diff[0]))
                        self.aul.send_joint_angles(angles_out,5)
                    else:
                        mid_angles.reverse()
                        near_diff.reverse()
                        far_diff.reverse()
                    lap += 1
                    #if lap == 3:
                    #    self.say.publish(data="Hold Still, you rascal!")
                    count = 0
                    rospy.sleep(0.5)
            self.say.publish(data="Pulling away")
            angles_out = np.add(mid_angles[0], np.multiply(2, near_diff[0]))
            self.aul.send_joint_angles(angles_out,5)
            rospy.sleep(5)
            self.say.publish(data="Finished wiping. Thank you")

    def torso_frame_move(self, msg):
        self.stop_maintain = True
        goal = self.aul.curr_pose()
        goal.pose.position.x += msg.x
        goal.pose.position.y += msg.y
        goal.pose.position.z += msg.z
        self.aul.blind_move(goal)
    
    def hand_move(self, f32):
        self.stop_maintain = True
        hfm_pose = self.aul.hand_frame_move(f32.data)
        self.aul.blind_move(hfm_pose)

    def norm_approach(self, pose):
        self.stop_maintain = True
        self.aul.update_frame(pose)
        appr = self.aul.find_approach(pose, 0.15)
        self.aul.move_arm_to(appr)
        
    def grasp(self, ps):
        self.stop_maintain = True
        rospy.loginfo("Initiating Grasp Sequence")
        self.wt_log_out.publish(data="Initiating Grasp Sequence")
        self.aul.update_frame(ps)
        approach = self.aul.find_approach(ps, 0.15)
        rospy.loginfo("approach: \r\n %s" %approach)
        at_appr = self.aul.move_arm_to(approach)
        rospy.loginfo("arrived at approach: %s" %at_appr)
        if at_appr:
            opened = self.aul.gripper(0.09)
            if opened:
                rospy.loginfo("making linear approach")
                #hfm_pose = self.aul.hand_frame_move(0.23) 
                hfm_pose = self.aul.find_approach(ps,-0.02)
                self.aul.blind_move(hfm_pose)
                self.aul.wait_for_stop(2)
                closed = self.aul.gripper(0.0)
                if not closed:
                    rospy.loginfo("Couldn't close completely: Grasp likely successful")
                hfm_pose = self.aul.hand_frame_move(-0.23) 
                self.aul.blind_move(hfm_pose)
        else:
            pass

    def prep_surf_wipe(self, point):
        pixel_u = point.x
        pixel_v = point.y
        test_pose = self.p23d_proxy(pixel_u, pixel_v).pixel3d
        self.aul.update_frame(test_pose)
        test_pose = self.aul.find_approach(test_pose, 0)
        (reachable, ik_goal) = self.aul.full_ik_check(test_pose)
        if reachable:
            if not self.surf_wipe_started:
                start_pose = test_pose
                self.surf_wipe_start = [pixel_u, pixel_v, start_pose]
                self.surf_wipe_started = True
                rospy.loginfo("Received valid starting position for wiping action")
                self.wt_log_out.publish(data="Received valid starting position for wiping action")
                return None#Return after 1st point, wait for second
            else:
                rospy.loginfo("Received valid ending position for wiping action")
                self.wt_log_out.publish(data="Received valid ending position for wiping action")
                self.surf_wipe_started = False #Continue on successful 2nd point
        else:
            rospy.loginfo("Cannot reach wipe position, please try another")
            self.wt_log_out.publish(data="Cannot reach wipe position, please try another")
            return None#Return on invalid point, wait for another
        
        dist = self.aul.calc_dist(self.surf_wipe_start[2],test_pose)
        print 'dist', dist
        num_points = dist/0.01
        print 'num_points', num_points
        us = np.round(np.linspace(self.surf_wipe_start[0], pixel_u, num_points))
        vs = np.round(np.linspace(self.surf_wipe_start[1], pixel_v, num_points))
        surf_points = [PoseStamped() for i in xrange(len(us))]
        print "Surface Points", [us,vs]
        for i in xrange(len(us)):
            pose = self.p23d_proxy(us[i],vs[i]).pixel3d
            self.aul.update_frame(pose)
            surf_points[i] = self.aul.find_approach(pose,0)
            print i+1, '/', len(us)
        return surf_points
    


        #self.aul.blind_move(surf_points[0])
        #self.aul.wait_for_stop()
        #for pose in surf_points:
        #    self.aul.blind_move(pose,2.5)
        #    rospy.sleep(2)
        #    #self.aul.wait_for_stop()
        #self.aul.hand_frame_move(-0.1)       

    def twist_wipe(self):
        angles = list(self.aul.joint_state_act.positions)
        count = 0
        while count < 3:
            angles[6] = -6.7
            self.aul.send_joint_angles(angles)
            while self.aul.joint_state_act.positions[6] > -6.6:
                rospy.sleep(0.1)
            angles[6] = 0.8
            self.aul.send_joint_angles(angles)
            while self.aul.joint_state_act.positions[6] < 0.7:
                rospy.sleep(0.1)
            count += 1
            

    def poke(self, ps):
        self.stop_maintain = True
        self.aul.update_frame(ps)
        appr = self.aul.find_approach(ps,0.15)
        touch = self.aul.find_approach(ps,0)
        prepared = self.aul.move_arm_to(appr)
        if prepared:
            self.aul.blind_move(touch)
            self.aul.wait_for_stop()
            rospy.sleep(7)
            self.aul.blind_move(appr)

    def swipe(self, ps):
        traj = self.prep_wipe(ps)
        if traj is not None:
            self.stop_maintain = True
            self.wipe_move(traj, 1)

    def wipe(self, ps):
        traj = self.prep_wipe(ps)
        if traj is not None:
            self.stop_maintain = True
            self.wipe_move(traj, 4)

    def prep_wipe(self, ps):
        #print "Prep Wipe Received: %s" %pa
        self.aul.update_frame(ps)
        print "Updating frame to: %s \r\n" %ps
        if not self.wipe_started:
            self.wipe_appr_seed = ps
            self.wipe_ends[0] = self.aul.find_approach(ps, 0)
            print "wipe_end[0]: %s" %self.wipe_ends[0]
            (reachable, ik_goal) = self.aul.full_ik_check(self.wipe_ends[0])
            if not reachable:
                rospy.loginfo("Cannot find approach for initial wipe position, please try another")
                self.wt_log_out.publish(data="Cannot find approach for initial wipe position, please try another")
                return None
            else:
                self.wipe_started = True
                rospy.loginfo("Received starting position for wiping action")
                self.wt_log_out.publish(data="Received starting position for wiping action")
                return None
        else:
            self.wipe_ends[1] = self.aul.find_approach(ps, 0)
            self.wipe_ends.reverse()
            (reachable, ik_goal) = self.aul.full_ik_check(self.wipe_ends[1])
            if not reachable:
                rospy.loginfo("Cannot find approach for final wipe position, please try another")
                self.wt_log_out.publish(data="Cannot find approach for final wipe position, please try another")
                return None
            else:
                rospy.loginfo("Received End position for wiping action")
                self.wt_log_out.publish(data="Received End position for wiping action")
                ####### REMOVED AND REPLACED WITH ALIGN FUNCTION ##############
                self.wipe_ends[0], self.wipe_ends[1] = self.align_poses(self.wipe_ends[0],self.wipe_ends[1])
    

                self.aul.update_frame(self.wipe_appr_seed)
                appr = self.aul.find_approach(self.wipe_appr_seed, 0.15)
                appr.pose.orientation = self.wipe_ends[1].pose.orientation
                prepared = self.aul.move_arm_to(appr)
                if prepared:
                    #self.aul.advance_to_contact()
                    self.aul.blind_move(self.wipe_ends[1])
                    traj = self.aul.build_trajectory(self.wipe_ends[1], self.wipe_ends[0])
                    wipe_traj = self.aul.build_follow_trajectory(traj)
                    self.aul.wait_for_stop()
                    self.wipe_started = False
                    return wipe_traj
                    #self.wipe(wipe_traj)
                else:
                    rospy.loginfo("Failure reaching start point, please try again")
                    self.wt_log_out.publish(data="Failure reaching start point, please try again")
    
    def align_poses(self, ps1, ps2):
        
                self.aul.update_frame(ps1)
                ps2.header.stamp = rospy.Time.now()
                self.tfl.waitForTransform(ps2.header.frame_id, 'lh_utility_frame', rospy.Time.now(), rospy.Duration(3.0))
                ps2_in_ps1 = self.tfl.transformPose('lh_utility_frame', ps2)
                
                ang = math.atan2(-ps2_in_ps1.pose.position.z, -ps2_in_ps1.pose.position.y)+(math.pi/2)
                q_st_rot = transformations.quaternion_about_axis(ang, (1,0,0))
                q_st_new = transformations.quaternion_multiply([ps1.pose.orientation.x, ps1.pose.orientation.y, ps1.pose.orientation.z, ps1.pose.orientation.w],q_st_rot)
                ps1.pose.orientation = Quaternion(*q_st_new)

                self.aul.update_frame(ps2)
                ps1.header.stamp = rospy.Time.now()
                self.tfl.waitForTransform(ps1.header.frame_id, 'lh_utility_frame', rospy.Time.now(), rospy.Duration(3.0))
                ps1_in_ps2 = self.tfl.transformPose('lh_utility_frame', ps1)
                ang = math.atan2(ps1_in_ps2.pose.position.z, ps1_in_ps2.pose.position.y)+(math.pi/2)
                
                q_st_rot = transformations.quaternion_about_axis(ang, (1,0,0))
                q_st_new = transformations.quaternion_multiply([ps2.pose.orientation.x, ps2.pose.orientation.y, ps2.pose.orientation.z, ps2.pose.orientation.w],q_st_rot)
                ps2.pose.orientation = Quaternion(*q_st_new)
                return ps1, ps2


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
            self.aul.l_arm_follow_traj_client.send_goal(traj_goal)
            self.aul.l_arm_follow_traj_client.wait_for_result(rospy.Duration(20))
            rospy.sleep(0.5)# Pause at end of swipe
            count += 1
        
        rospy.loginfo("Done Wiping")
        self.wt_log_out.publish(data="Done Wiping")
        hfm_pose = self.aul.hand_frame_move(-0.15)
        self.aul.blind_move(hfm_pose)


if __name__ == '__main__':
    AA = ArmActions()
    while not rospy.is_shutdown():
        #AA.maintain_force_position()
        #AA.mannequin()
        rospy.spin()
