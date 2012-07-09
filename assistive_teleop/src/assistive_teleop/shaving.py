#!/usr/bin/python

import math 

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
from geometry_msgs.msg import PoseStamped, WrenchStamped, Point, Quaternion
from std_msgs.msg import String, Int8, Bool
from tf import TransformListener
import actionlib

import jttask_utils
from assistive_teleop.msg import FtMoveAction, FtMoveGoal, FtHoldAction, FtHoldGoal
from hrl_rfh_fall_2011.srv import GetHeadPose, GetHeadPoseRequest, EllipsoidCommand, EllipsoidCommandRequest
from hrl_pr2_arms.pr2_controller_switcher import ControllerSwitcher

POSTURE = False
SHAVING_TEST = False
TOOL = '45' #'inline' #'90'

class Shaving_manager():
    def __init__(self):
        self.tf = TransformListener()
        self.jtarms = jttask_utils.jt_task_utils(self.tf)
        self.controller_switcher = ControllerSwitcher()
        
        ##### Subscribers ####
        self.disc_sub = rospy.Subscriber('wt_disc_shaving_step', Int8, self.disc_change_pose)
        self.disc_sub.impl.add_callback(self.cancel_goals, None)
        self.cont_sub = rospy.Subscriber('wt_cont_shaving_step', Point, self.cont_change_pose)
        self.cont_sub.impl.add_callback(self.cancel_goals, None)
        self.loc_sub = rospy.Subscriber('wt_shave_location', Int8, self.set_shave_pose)
        self.loc_sub.impl.add_callback(self.cancel_goals, None)
        rospy.Subscriber('wt_stop_shaving', Bool, self.stop_shaving)

        #### Services ####
        rospy.loginfo("Waiting for get_head_pose service")
        try:
            rospy.wait_for_service('/get_head_pose', 5.0)
            self.get_pose_proxy = rospy.ServiceProxy('/get_head_pose', GetHeadPose)
            rospy.loginfo("Found get_head_pose service")
        except:
            rospy.logwarn("get_head_pose service not available")

        rospy.loginfo("Waiting for ellipsoid_command service")
        try:
            rospy.wait_for_service('/ellipsoid_command', 5.0)
            self.ellipsoid_command_proxy = rospy.ServiceProxy('/ellipsoid_command', EllipsoidCommand)
            rospy.loginfo("Found ellipsoid_command service")
        except:
            rospy.logwarn("ellipsoid_command service not available")
        #### Publishers ####
        self.wt_log_out = rospy.Publisher('wt_log_out', String)
        self.test_out = rospy.Publisher('test_pose_1', PoseStamped)
        self.rezero_wrench = rospy.Publisher('/netft_gravity_zeroing/rezero_wrench', Bool)

        #### Data ####
        self.hand_offset = 0.195
        self.angle_tool_offset = 0.03
        self.inline_tool_offset = 0.085
        self.selected_pose = 0
        self.poses = {
            0 : ["near_ear",0.],
            1 : ["upper_cheek",0.],
            2 : ["middle_cheek",0.],
            3 : ["jaw_bone",0.],
            4 : ["back_neck",0.],
            5 : ["nose",0.],
            6 : ["chin",0.],
            7 : ["mouth_corner", 0.]
        }

        #self.ft_wrench = WrenchStamped()
       # self.force_unsafe = False
       # self.ft_activity_thresh = 3
       # self.ft_safety_thresh = 12
    #    rospy.Subscriber('ft_data_pm_adjusted', WrenchStamped, self.get_ft_state)
    #    rospy.Subscriber('/netft_gravity_zeroing/wrench_zeroed', WrenchStamped, self.get_ft_state)
        if POSTURE:
            rospy.sleep(3)
            print 'Sending Posture'
            self.jtarms.send_posture(posture='elbowup', arm=0) 
    
  #  def get_ft_state(self, ws):
  #      self.ft_wrench = ws
  #      self.ft_mag = math.sqrt(ws.wrench.force.x**2 + ws.wrench.force.y**2 + ws.wrench.force.z**2)
  #      if self.ft_mag > self.ft_safety_thresh:
  #          self.force_unsafe = True
  #      if self.ft_mag > self.ft_activity_thresh:
  #          self.ft_activity = rospy.Time.now()
    def stop_shaving(self, msg):
        self.cancel_goals(msg)
        rospy.loginfo("Stopping Shaving Behavior")
        self.wt_log_out.publish(data="Stopping Shaving Behavior")
        self.controller_switcher.carefree_switch('l','%s_arm_controller')

    def cancel_goals(self, msg):
        self.jtarms.ft_move_client.cancel_all_goals() #Stop what we're doing, as we're about to do something different, and stopping soon may be desired
        self.jtarms.ft_hold_client.cancel_all_goals()
        rospy.loginfo("Cancelling Active Shaving Goals")
        self.wt_log_out.publish(data="Cancelling Active Shaving Goals")

    def cont_change_pose(self, step):
        self.controller_switcher.carefree_switch('l','%s_cart')
        self.new_pose = True
        req = EllipsoidCommandRequest()
        req.change_latitude = int(-step.y)
        req.change_longitude = int(-step.x)
        req.change_height = int(step.z)
        print req
        print type(req.change_latitude)
        self.ellipsoid_command_proxy.call(req)
        rospy.loginfo("Moving Across Ellipsoid")
        self.wt_log_out.publish("Moving Across Ellipsoid")

    def disc_change_pose(self, step):
        self.new_pose = True
        self.selected_pose += step.data
        if self.selected_pose > 7:
            self.selected_pose = 7
            self.wt_log_out.publish(data="Final position in sequence, there is no next position")
            return
        elif self.selected_pose < 0:
            self.selected_pose = 0
            self.wt_log_out.publish(data="First position in sequence, there is no previous position")
            return
      
        rospy.loginfo("Moving to "+self.poses[self.selected_pose][0])
        self.wt_log_out.publish(data="Moving to "+self.poses[self.selected_pose][0])
        ellipse_pose = self.get_pose_proxy(self.poses[self.selected_pose][0], self.poses[self.selected_pose][1]).pose
        self.adjust_pose(ellipse_pose)

    def set_shave_pose(self, num):
        self.selected_pose = num.data
        rospy.loginfo("Moving to "+self.poses[self.selected_pose][0])
        self.wt_log_out.publish(data="Moving to "+self.poses[self.selected_pose][0])
        ellipse_pose = self.get_pose_proxy(self.poses[self.selected_pose][0], self.poses[self.selected_pose][1]).pose
        self.adjust_pose(ellipse_pose)

    def adjust_pose(self, ellipse_pose):
        self.controller_switcher.carefree_switch('l','%s_cart')
        self.jtarms.ft_move_client.cancel_all_goals() #Stop what we're doing, as we're about to do something different, and stopping soon may be desired
        self.jtarms.ft_hold_client.cancel_all_goals()
        #self.test_out.publish(ellipse_pose)
        print "New Position Received, should stop current action"
        self.tf.waitForTransform(ellipse_pose.header.frame_id, 'torso_lift_link', ellipse_pose.header.stamp, rospy.Duration(3.0))
        torso_pose = self.tf.transformPose('torso_lift_link', ellipse_pose)
        if TOOL == 'inline':
            goal_pose = self.jtarms.pose_frame_move(torso_pose, -(self.hand_offset + self.inline_tool_offset), arm=0)
            approach_pose = self.jtarms.pose_frame_move(goal_pose, -0.15, arm=0)
        elif TOOL == '90':
            goal_pose = self.jtarms.pose_frame_move(torso_pose, 0.015)
        elif TOOL == '45':
            goal_pose = torso_pose
            approach_pose = self.jtarms.pose_frame_move(goal_pose, -0.15, arm=0)

        traj_to_approach = self.jtarms.build_trajectory(approach_pose, arm=0)
        traj_appr_to_goal = self.jtarms.build_trajectory(goal_pose, approach_pose, arm=0)
        self.ft_move(traj_to_approach)
        self.rezero_wrench.publish(data=True)
        self.ft_move(traj_appr_to_goal, ignore_ft=False)
        retreat_traj = self.jtarms.build_trajectory(approach_pose, arm=0)
        escape_traj = self.jtarms.build_trajectory(approach_pose, arm=0, steps=30)
        self.jtarms.ft_hold_client.send_goal(FtHoldGoal(2.,8,10,rospy.Duration(10)))#activity_thresh, z_unsafe, mag_unsafe, timeout
        print "Waiting for hold result"
        finished = self.jtarms.ft_hold_client.wait_for_result(rospy.Duration(0))
        print "Finished before Timeout: %s" %finished
        print "Done Waiting"
        hold_result = self.jtarms.ft_hold_client.get_result()
        print "Hold Result:"
        print hold_result
        if hold_result.unsafe:
            self.ft_move(escape_traj)
        elif hold_result.timeout:
            self.ft_move(retreat_traj)
        
    def ft_move(self, traj, ft_thresh=2, ignore_ft=True):
        self.jtarms.ft_move_client.send_goal(FtMoveGoal(traj, ft_thresh, ignore_ft),feedback_cb=self.ft_move_feedback)
        finished_within_timeout = self.jtarms.ft_move_client.wait_for_result(rospy.Duration(0.25*len(traj)))
        if finished_within_timeout:
            result = self.jtarms.ft_move_client.get_result()
            if result.all_sent:
                rospy.loginfo("Full Trajectory Completed")
                self.wt_log_out.publish(data="Full Trajectory Completed")
            elif result.contact:
                rospy.loginfo("Stopped Trajectory upon contact")
                self.wt_log_out.publish(data="Stopped Trajectory upon contact")
        else:
            self.jtarms.ft_move_client.cancel_goal()
            rospy.loginfo("Failed to complete movement within timeout period.")
            self.wt_log_out.publish(data="Failed to complete movement within timeout period.")


    def ft_move_feedback(self, fdbk):
        pct = 100.*float(fdbk.current)/float(fdbk.total)
        #rospy.loginfo("Movement is %2.0f percent complete." %pct)
        self.wt_log_out.publish(data="Movement to %s is %2.0f percent complete." 
                                        %(self.poses[self.selected_pose][0], pct))


    #def hold_ft_aware(self, escape_pose, arm=0):
    #    self.jtarms.force_stopped = False
    #    self.force_unsafe = False
    #    escape_traj = self.jtarms.build_trajectory(escape_pose)
    #    time_since_activity = rospy.Duration(0)
    #    self.ft_activity = rospy.Time.now()
    #    while not (rospy.is_shutdown() or self.force_unsafe or time_since_activity>rospy.Duration(10) or self.new_pose):
    #        time_since_activity = rospy.Time.now()-self.ft_activity
    #        rospy.sleep(0.01)

    #    if self.force_unsafe:
    #        rospy.loginfo("Unsafe High Forces, moving away!")
    #        self.wt_log_out.publish(data="Unsafe High Forces, moving away!")    
    #        self.jtarms.goal_pub[arm].publish(escape_pose)
    #        return

    #    if time_since_activity>rospy.Duration(10):
    #        rospy.loginfo("10s of inactivity, moving away")
    #        self.wt_log_out.publish(data="10s of inactivity, moving away")    

    #    if self.new_pose:
    #        rospy.loginfo("New Pose Received")
    #        self.wt_log_out.publish(data="New Pose Received")    
    #        self.new_pose = False

    #    self.jtarms.send_traj(escape_traj)


    def test(self):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'torso_lift_link'
        goal_pose.header.stamp = rospy.Time.now()
        goal_pose.pose.position = Point(0.8, 0.3, 0.0)
        goal_pose.pose.orientation = Quaternion(1,0,0,0)
        approach_pose = self.jtarms.pose_frame_move(goal_pose, -0.15, arm=0)
        traj_to_approach = self.jtarms.build_trajectory(approach_pose, arm=0)
        traj_appr_to_goal = self.jtarms.build_trajectory(goal_pose, approach_pose, arm=0)
        raw_input("Move to approach pose")
        self.jtarms.send_traj(traj_to_approach)
        self.jtarms.force_stopped = False
        self.force_unsafe = False
        raw_input("Move to contact pose")
        self.jtarms.force_stopped = False
        self.force_unsafe = False
        self.jtarms.send_traj_to_contact(traj_appr_to_goal)
        raw_input("Hold under force constraints")
        self.jtarms.force_stopped = False
        self.force_unsafe = False
        self.hold_ft_aware(approach_pose, arm=0)
        rospy.loginfo("Finished Testing")
            
    
if __name__=='__main__':
    rospy.init_node('shaving_manager')
    sm = Shaving_manager()
    if SHAVING_TEST:
        sm.test()
    else:
        while not rospy.is_shutdown():
            rospy.spin()
        #sm.controller_switcher.carefree_switch('l','%s_arm_controller')






        



