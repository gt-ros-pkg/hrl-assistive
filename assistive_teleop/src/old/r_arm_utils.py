#!/usr/bin/python

from threading import RLock
import math
import numpy as np

import roslib; roslib.load_manifest('web_teleop_trunk')
import rospy
import actionlib
from geometry_msgs.msg  import PoseStamped, WrenchStamped, Quaternion, Point
import arm_navigation_msgs.msg
from arm_navigation_msgs.msg import RobotState, MultiDOFJointState
from sensor_msgs.msg import JointState
from kinematics_msgs.srv import GetKinematicSolverInfo, GetPositionFK, GetPositionFKRequest, GetPositionIK, \
                                GetPositionIKRequest
from pr2_controllers_msgs.msg import JointTrajectoryAction, JointTrajectoryControllerState, JointTrajectoryActionGoal,\
                                     SingleJointPositionAction, SingleJointPositionGoal, Pr2GripperCommandAction,\
                                     Pr2GripperCommandGoal, JointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from std_msgs.msg import String, Float32
from tf import TransformListener, transformations, TransformBroadcaster
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTolerance
from web_teleop_trunk.srv import FrameUpdate, FrameUpdateRequest
from arm_navigation_msgs.srv import SetPlanningSceneDiff, SetPlanningSceneDiffRequest

class ArmUtils():
  
    wipe_started = False 
    standoff = 0.368 #0.2 + 0.168 (dist from wrist to fingertips)
    torso_min = 0.0115 
    torso_max = 0.295 
    dist = 0. 

    move_arm_error_dict = {
         -1 : "PLANNING_FAILED: Could not plan a clear path to goal.", 
          0 : "Move Arm Action Aborted on Internal Failure.", 
          1 : "SUCCEEDED", 
         -2 : "TIMED_OUT",
         -3 : "START_STATE_IN_COLLISION: Try freeing the arms manually.",
         -4 : "START_STATE_VIOLATES_PATH_CONSTRAINTS",
         -5 : "GOAL_IN_COLLISION",
         -6 : "GOAL_VIOLATES_PATH_CONSTRAINTS",
         -7 : "INVALID_ROBOT_STATE",
         -8 : "INCOMPLETE_ROBOT_STATE",
         -9 : "INVALID_PLANNER_ID",
         -10 : "INVALID_NUM_PLANNING_ATTEMPTS",
         -11 : "INVALID_ALLOWED_PLANNING_TIME",
         -12 : "INVALID_GROUP_NAME",
         -13 : "INVALID_GOAL_JOINT_CONSTRAINTS",
         -14 : "INVALID_GOAL_POSITION_CONSTRAINTS",
         -15 : "INVALID_GOAL_ORIENTATION_CONSTRAINTS",
         -16 : "INVALID_PATH_JOINT_CONSTRAINTS",
         -17 : "INVALID_PATH_POSITION_CONSTRAINTS",
         -18 : "INVALID_PATH_ORIENTATION_CONSTRAINTS",
         -19 : "INVALID_TRAJECTORY",
         -20 : "INVALID_INDEX",
         -21 : "JOINT_LIMITS_VIOLATED",
         -22 : "PATH_CONSTRAINTS_VIOLATED",
         -23 : "COLLISION_CONSTRAINTS_VIOLATED",
         -24 : "GOAL_CONSTRAINTS_VIOLATED",
         -25 : "JOINTS_NOT_MOVING",
         -26 : "TRAJECTORY_CONTROLLER_FAILED",
         -27 : "FRAME_TRANSFORM_FAILURE",
         -28 : "COLLISION_CHECKING_UNAVAILABLE",
         -29 : "ROBOT_STATE_STALE",
         -30 : "SENSOR_INFO_STALE",
         -31 : "NO_IK_SOLUTION: Cannot reach goal configuration.",
         -32 : "INVALID_LINK_NAME",
         -33 : "IK_LINK_IN_COLLISION: Cannot reach goal configuration without contact.",
         -34 : "NO_FK_SOLUTION",
         -35 : "KINEMATICS_STATE_IN_COLLISION",
         -36 : "INVALID_TIMEOUT"
         }

    def __init__(self, tfListener=None):
        self.move_right_arm_client = actionlib.SimpleActionClient('move_right_arm', arm_navigation_msgs.msg.MoveArmAction)
        rospy.loginfo("Waiting for move_right_arm server")
        if self.move_right_arm_client.wait_for_server(rospy.Duration(50)):
            rospy.loginfo("Found move_right_arm server")
        else:
            rospy.logwarn("Cannot find move_right_arm server")

        self.r_arm_traj_client = actionlib.SimpleActionClient('r_arm_controller/joint_trajectory_action', JointTrajectoryAction)
        rospy.loginfo("Waiting for r_arm_controller/joint_trajectory_action server")
        if self.r_arm_traj_client.wait_for_server(rospy.Duration(5)):
            rospy.loginfo("Found r_arm_controller/joint_trajectory_action server")
        else:
            rospy.logwarn("Cannot find r_arm_controller/joint_trajectory_action server")

        self.r_arm_follow_traj_client = actionlib.SimpleActionClient('r_arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for r_arm_controller/follow_joint_trajectory server")
        if self.r_arm_follow_traj_client.wait_for_server(rospy.Duration(5)):
            rospy.loginfo("Found r_arm_controller/follow_joint_trajectory server")
        else:
            rospy.logwarn("Cannot find r_arm_controller/follow_joint_trajectory server")

        self.torso_client = actionlib.SimpleActionClient('torso_controller/position_joint_action', SingleJointPositionAction)
        rospy.loginfo("Waiting for  torso_controller/position_joint_action server")
        if self.torso_client.wait_for_server(rospy.Duration(5)):
            rospy.loginfo("Found torso_controller/position_joint_action server")
        else:
            rospy.logwarn("Cannot find torso_controller/position_joint_action server")

        self.r_gripper_client = actionlib.SimpleActionClient('r_gripper_controller/gripper_action', Pr2GripperCommandAction)
        rospy.loginfo("Waiting for r_gripper_controller/gripper_action server")
        if self.r_gripper_client.wait_for_server(rospy.Duration(5)):
            rospy.loginfo("Found r_gripper_controller/gripper_action server")
        else:
            rospy.logwarn("Cannot find r_gripper_controller/gripper_action server")
        
        rospy.loginfo("Waiting for r_utility_frame_service")
        try:
            rospy.wait_for_service('/r_utility_frame_update', 7.0)
            self.update_frame = rospy.ServiceProxy('/r_utility_frame_update', FrameUpdate)
            rospy.loginfo("Found r_utility_frame_service")
        except:
            rospy.logwarn("Right Utility Frame Service Not available")
        
        #self.joint_state_lock = RLock()
        rospy.Subscriber('r_arm_controller/state', JointTrajectoryControllerState, self.update_joint_state)
        self.torso_state_lock = RLock()
        rospy.Subscriber('torso_controller/state', JointTrajectoryControllerState, self.update_torso_state)
        
        #self.r_arm_command = rospy.Publisher('/r_arm_controller/command', JointTrajectory )
        self.wt_log_out = rospy.Publisher('wt_log_out', String)

        if tfListener is None:
            self.tf = TransformListener()
        else:
            self.tf = tfListener
        self.tfb = TransformBroadcaster()
        
        rospy.loginfo("Waiting for FK/IK Solver services")
        try:
            rospy.wait_for_service('/pr2_right_arm_kinematics/get_fk')
            rospy.wait_for_service('/pr2_right_arm_kinematics/get_fk_solver_info')
            rospy.wait_for_service('/pr2_right_arm_kinematics/get_ik')
            rospy.wait_for_service('/pr2_right_arm_kinematics/get_ik_solver_info')
            self.fk_info_proxy = rospy.ServiceProxy('/pr2_right_arm_kinematics/get_fk_solver_info',
                                                    GetKinematicSolverInfo)
            self.fk_info = self.fk_info_proxy()
            self.fk_pose_proxy = rospy.ServiceProxy('/pr2_right_arm_kinematics/get_fk', GetPositionFK, True)    
            self.ik_info_proxy = rospy.ServiceProxy('/pr2_right_arm_kinematics/get_ik_solver_info',
                                                    GetKinematicSolverInfo)
            self.ik_info = self.ik_info_proxy()
            self.ik_pose_proxy = rospy.ServiceProxy('/pr2_right_arm_kinematics/get_ik', GetPositionIK, True)    
            rospy.loginfo("Found FK/IK Solver services")
        except:
            rospy.logerr("Could not find FK/IK Solver services")
        
        self.set_planning_scene_diff_name = '/environment_server/set_planning_scene_diff'
        rospy.wait_for_service('/environment_server/set_planning_scene_diff')
        self.set_planning_scene_diff = rospy.ServiceProxy('/environment_server/set_planning_scene_diff', SetPlanningSceneDiff)
        self.set_planning_scene_diff(SetPlanningSceneDiffRequest())
    
    def update_joint_state(self, jtcs):
        #self.joint_state_lock.acquire()
        self.joint_state_time = jtcs.header.stamp
        self.joint_state_act = jtcs.actual
        self.joint_state_des = jtcs.desired
        #self.joint_state_lock.release()

    def update_torso_state(self, jtcs):
        self.torso_state_lock.acquire()
        self.torso_state = jtcs
        self.torso_state_lock.release()
        
    def curr_pose(self):
        #print 'Requesting Current Pose'
        (trans,rot) = self.tf.lookupTransform('/torso_lift_link', '/r_wrist_roll_link', rospy.Time(0))
        cp = PoseStamped()
        cp.header.stamp = rospy.Time.now()
        cp.header.frame_id = '/torso_lift_link'
        cp.pose.position = Point(*trans)
        cp.pose.orientation = Quaternion(*rot)
        #print 'Current Pose:', cp
        return cp
   
    def wait_for_stop(self, wait_time=1, timeout=60):
        rospy.sleep(wait_time)
        end_time = rospy.Time.now()+rospy.Duration(timeout)
        while not self.is_stopped():
            if rospy.Time.now()<end_time:
                rospy.sleep(wait_time)
            else:
                raise

    def is_stopped(self):
        #self.joint_state_lock.acquire()
        time = self.joint_state_time
        vels = self.joint_state_act.velocities
        #self.joint_state_lock.release()
        if np.allclose(vels, np.zeros(7)) and (rospy.Time.now()-time)<rospy.Duration(0.1):
            return True
        else:
            return False
    
    def get_fk(self, angles, frame='torso_lift_link', link=-1): # get FK pose of a list of joint angles for the arm
        fk_request = GetPositionFKRequest()
        fk_request.header.frame_id = frame
        fk_request.header.stamp = rospy.Time.now()
        fk_request.fk_link_names =  self.fk_info.kinematic_solver_info.link_names
        fk_request.robot_state.joint_state.header = fk_request.header
        fk_request.robot_state.joint_state.position = angles #self.joint_state_act.positions
        fk_request.robot_state.joint_state.name = self.fk_info.kinematic_solver_info.joint_names
        print 'fk_request:', fk_request
        try:
            return self.fk_pose_proxy(fk_request).pose_stamped[link]
        except rospy.ServiceException, e:
            rospy.loginfo("FK service did not process request: %s" %str(e))
    
    def adjust_elbow(self, f32):
        request = self.form_ik_request(self.curr_pose())
        angs =  list(request.ik_request.ik_seed_state.joint_state.position)
        if f32.data == 1:
            angs[2] -= 0.25 #Angle direction different between arms
        else:
            angs[2] += 0.25
        request.ik_request.ik_seed_state.joint_state.position = angs 
        ik_goal = self.ik_pose_proxy(request)
        if ik_goal.error_code.val == 1:
            self.send_joint_angles(ik_goal.solution.joint_state.position)
        else:
            rospy.loginfo("Cannot adjust elbow position further")
            self.wt_log_out.publish(data="Cannot adjust elbow position further")

    def gripper(self, position, effort=30):
        pos = np.clip(position,0.0, 0.09)
        goal = Pr2GripperCommandGoal()
        goal.command.position = pos
        goal.command.max_effort = effort
        self.r_gripper_client.send_goal(goal)
        finished_within_time = self.r_gripper_client.wait_for_result(rospy.Duration(15))
        if not (finished_within_time):
            self.r_gripper_client.cancel_goal()
            rospy.loginfo("Timed out moving right gripper")
            return False
        else:
            state = self.r_gripper_client.get_state()
            result = self.r_gripper_client.get_result()
            if (state == 3): #3 == SUCCEEDED
                rospy.loginfo("Gripper Action Succeeded")
                return True
            else:
                rospy.loginfo("Gripper Action Failed")
                rospy.loginfo("Failure Result: %s" %result)
                return False

    def find_approach(self, msg, stdoff=0.20):
        stdoff += 0.17 #0.185 #adjust fingertip standoffs for distance to wrist link
        self.pose_in = msg
        self.tf.waitForTransform('rh_utility_frame','base_footprint', rospy.Time(0), rospy.Duration(3.0))
        self.tfb.sendTransform((self.pose_in.pose.position.x, self.pose_in.pose.position.y,
                                self.pose_in.pose.position.z),
                                (self.pose_in.pose.orientation.x, self.pose_in.pose.orientation.y,
                                self.pose_in.pose.orientation.z, self.pose_in.pose.orientation.w),
                                rospy.Time.now(),
                                "rh_utility_frame",
                                self.pose_in.header.frame_id)
        self.tf.waitForTransform('rh_utility_frame','r_wrist_roll_link', rospy.Time.now(), rospy.Duration(3.0))
        goal = PoseStamped()
        goal.header.frame_id = 'rh_utility_frame'
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.z = stdoff
        goal.pose.orientation.x = 0
        goal.pose.orientation.y = 0.5*math.sqrt(2)
        goal.pose.orientation.z = 0
        goal.pose.orientation.w = 0.5*math.sqrt(2)
        self.tf.waitForTransform(goal.header.frame_id, 'torso_lift_link', rospy.Time.now(), rospy.Duration(3.0))
        appr = self.tf.transformPose('torso_lift_link', goal)
        return appr

    def hand_frame_move(self, x, y=0, z=0, duration=None):
        print "Right Hand Frame Move"
        cp = self.curr_pose()
        self.tf.waitForTransform(cp.header.frame_id, '/r_wrist_roll_link', cp.header.stamp, rospy.Duration(3.0))
        newpose = self.tf.transformPose('/r_wrist_roll_link', cp)
        newpose.pose.position.x += x
        newpose.pose.position.y += y
        newpose.pose.position.z += z
        self.dist = math.sqrt(x**2+y**2+z**2)
        return  self.tf.transformPose(cp.header.frame_id, newpose)

    def build_trajectory(self, finish, start=None, ik_space = 0.005, duration = None, tot_points=None):
        if start == None: # if given one pose, use current position as start, else, assume (start, finish)
            start = self.curr_pose()
       
        #TODO: USE TF TO BASE DISTANCE ON TOOL_FRAME MOVEMENT DISTANCE, NOT WRIST_ROLL_LINK

        # Based upon distance to travel, determine the number of intermediate steps required
        # find linear increments of position.

        dist = self.calc_dist(start,finish)     #Total distance to travel
        ik_steps = math.ceil(dist/ik_space)     
        print "IK Steps: %s" %ik_steps
        if tot_points is None:
           tot_points = 1500*dist
        if duration is None:
            duration = dist*120
        ik_fracs = np.linspace(0, 1, ik_steps)   #A list of fractional positions along course to evaluate ik
        ang_fracs = np.linspace(0,1, tot_points)  

        x_gap = finish.pose.position.x - start.pose.position.x
        y_gap = finish.pose.position.y - start.pose.position.y
        z_gap = finish.pose.position.z - start.pose.position.z

        qs = [start.pose.orientation.x, start.pose.orientation.y,
              start.pose.orientation.z, start.pose.orientation.w] 
        qf = [finish.pose.orientation.x, finish.pose.orientation.y,
              finish.pose.orientation.z, finish.pose.orientation.w] 

        #For each step between start and finish, find a complete pose (linear position changes, and quaternion slerp)
        steps = [PoseStamped() for i in range(len(ik_fracs))]
        for i,frac in enumerate(ik_fracs):
            steps[i].header.stamp = rospy.Time.now()
            steps[i].header.frame_id = start.header.frame_id
            steps[i].pose.position.x = start.pose.position.x + x_gap*frac
            steps[i].pose.position.y = start.pose.position.y + y_gap*frac
            steps[i].pose.position.z = start.pose.position.z + z_gap*frac
            new_q = transformations.quaternion_slerp(qs,qf,frac)
            steps[i].pose.orientation.x = new_q[0]
            steps[i].pose.orientation.y = new_q[1]
            steps[i].pose.orientation.z = new_q[2]
            steps[i].pose.orientation.w = new_q[3]
        rospy.loginfo("Planning straight-line path, please wait")
        self.wt_log_out.publish(data="Planning straight-line path, please wait")
       
        #For each pose in trajectory, find ik angles
        #Find initial ik for seeding
        req = self.form_ik_request(steps[0])
        ik_goal = self.ik_pose_proxy(req)
        seed = ik_goal.solution.joint_state.position

        ik_points = [[0]*7 for i in range(len(ik_fracs))]
        for i, p in enumerate(steps):
            request = self.form_ik_request(p)
            request.ik_request.ik_seed_state.joint_state.position = seed
            ik_goal = self.ik_pose_proxy(request)
            ik_points[i] = ik_goal.solution.joint_state.position
            seed = ik_goal.solution.joint_state.position # seed the next ik w/previous points results
        ik_points = np.array(ik_points)
        # Linearly interpolate angles 10 times between ik-defined points (non-linear in cartesian space, but this is reduced from dense ik sampling along linear path.  Used to maintain large number of trajectory points without running IK on every one.    
        angle_points = np.zeros((7, tot_points))
        for i in xrange(7):
            angle_points[i,:] = np.interp(ang_fracs, ik_fracs, ik_points[:,i])

        #Build Joint Trajectory from angle positions
        traj = JointTrajectory()
        traj.header.frame_id = steps[0].header.frame_id
        traj.joint_names = self.ik_info.kinematic_solver_info.joint_names
        #print 'angle_points', len(angle_points[0]), angle_points
        for i in range(len(angle_points[0])):
            traj.points.append(JointTrajectoryPoint())
            traj.points[i].positions = angle_points[:,i]
            traj.points[i].velocities = [0]*7
            traj.points[i].time_from_start = rospy.Duration(ang_fracs[i]*duration)
        return traj

    def build_follow_trajectory(self, traj):
        # Build 'Follow Joint Trajectory' goal from trajectory (includes tolerances for error)
        traj_goal = FollowJointTrajectoryGoal()
        traj_goal.trajectory = traj
        tolerance = JointTolerance()
        tolerance.position = 0.05
        tolerance.velocity = 0.1
        traj_goal.path_tolerance = [tolerance for i in range(len(traj.points))]
        traj_goal.goal_tolerance = [tolerance for i in range(len(traj.points))]
        traj_goal.goal_time_tolerance = rospy.Duration(3)
        return traj_goal

    def move_torso(self, pos):
        rospy.loginfo("Moving Torso to reach arm goal")
        goal_out = SingleJointPositionGoal()
        goal_out.position = pos
        
        finished_within_time = False
        self.torso_client.send_goal(goal_out)
        finished_within_time = self.torso_client.wait_for_result(rospy.Duration(45))
        if not (finished_within_time):
            self.torso_client.cancel_goal()
            self.wt_log_out.publish(data="Timed out moving torso")
            rospy.loginfo("Timed out moving torso")
            return False
        else:
            state = self.torso_client.get_state()
            result = self.torso_client.get_result()
            if (state == 3): #3 == SUCCEEDED
                rospy.loginfo("Torso Action Succeeded")
                self.wt_log_out.publish(data="Move Torso Succeeded")
                return True
            else:
                rospy.loginfo("Move Torso Failed")
                rospy.loginfo("Failure Result: %s" %result)
                self.wt_log_out.publish(data="Move Torso Failed")
                return False

    def check_torso(self, request):
        rospy.loginfo("Checking Torso")
        goal_z = request.ik_request.pose_stamped.pose.position.z
        #print "Goal z: %s" %goal_z
        self.torso_state_lock.acquire()
        torso_pos = self.torso_state.actual.positions[0]
        self.torso_state_lock.release()
        spine_range = [self.torso_min - torso_pos, self.torso_max - torso_pos]
        rospy.loginfo("Spine Range: %s" %spine_range)
        
        #Check for exact solution if goal can be made level with spine (possible known best case first)
        if goal_z >= spine_range[0] and goal_z <= spine_range[1]: 
            rospy.loginfo("Goal within spine movement range")
            request.ik_request.pose_stamped.pose.position.z = 0;
            streach_goal = goal_z
        elif goal_z > spine_range[1]:#Goal is above possible shoulder height
            rospy.loginfo("Goal above spine movement range")
            request.ik_request.pose_stamped.pose.position.z -= spine_range[1]
            streach_goal = spine_range[1]
        else:#Goal is below possible shoulder height
            rospy.loginfo("Goal below spine movement range")
            request.ik_request.pose_stamped.pose.position.z -= spine_range[0]
            streach_goal = spine_range[0]
        
        #print "Checking optimal position, which gives: \r\n %s" %request.ik_request.pose_stamped.pose.position
        ik_goal = self.ik_pose_proxy(request)
        if ik_goal.error_code.val ==1:
            rospy.loginfo("Goal can be reached by moving spine")
            self.wt_log_out.publish(data="Goal can be reached by moving spine")
            #print "Streach Goal: %s" %streach_goal
        else:
            rospy.loginfo("Goal cannot be reached, even using spine movement")
            return [False, request.ik_request.pose_stamped]

        #Find nearest working solution (so you don't wait for full-range spine motions all the time, but move incrementally
        trial = 1
        while True:
            request.ik_request.pose_stamped.pose.position.z = goal_z - 0.1*trial*streach_goal
            ik_goal = self.ik_pose_proxy(request)
            if ik_goal.error_code.val == 1:
                self.wt_log_out.publish(data="Using torso to reach goal")
                rospy.loginfo("Using Torso to reach goal")
                #print "Acceptable modified Pose: \r\n %s" %request.ik_request.pose_stamped.pose.position
                streached = self.move_torso(torso_pos + 0.1*trial*streach_goal)
                return [True, request.ik_request.pose_stamped]
            else:
                if trial < 10:
                    trial += 1
                    print "Trial %s" %trial
                else:
                    return [False, request.ik_request.pose_stamped]
   
    def fast_move(self, ps, time=0.):
        ik_goal = self.ik_pose_proxy(self.form_ik_request(ps))
        if ik_goal.error_code.val == 1:
            point = JointTrajectoryPoint()
            point.positions = ik_goal.solution.joint_state.position
            self.send_traj_point(point)
        else:
            rospy.logerr("FAST MOVE IK FAILED!")

    def blind_move(self, ps, duration = None):
        (reachable, ik_goal) = self.full_ik_check(ps)
        if reachable:
            self.send_joint_angles(ik_goal.solution.joint_state.position, duration)

    def full_ik_check(self, ps):
        #print "Blind Move Command Received:\r\n %s" %ps.pose.position
        req = self.form_ik_request(ps)
        ik_goal = self.ik_pose_proxy(req)
        if ik_goal.error_code.val == 1:
            self.dist = self.calc_dist(ps)
            #print "Initial IK Good, sending goal: %s" %ik_goal
            return (True, ik_goal)
        else:
         #   print "Initial IK Bad, finding better solution"
            (torso_succeeded, pos) = self.check_torso(req)
            if torso_succeeded:
          #      print "Successful with Torso, sending new position"
           #     print "Fully reachable with torso adjustment: New Pose:\r\n %s" %pos
                self.dist = self.calc_dist(pos)
                ik_goal = self.ik_pose_proxy(self.form_ik_request(pos))# From pose, get request, get ik
                return (True, ik_goal)
            else:
                rospy.loginfo("Incrementing Reach")
                percent = 0.9
                while percent > 0.01:
                    print "Percent: %s" %percent
                    #print "Trying to move %s percent of the way" %percent
                    goal_pos = req.ik_request.pose_stamped.pose.position
                    #print "goal_pos: \r\n %s" %goal_pos
                    curr_pos = self.curr_pose()
                    curr_pos = curr_pos.pose.position
                    #print "curr_pos: \r\n %s" %curr_pos
                    req.ik_request.pose_stamped.pose.position.x = curr_pos.x + percent*(goal_pos.x-curr_pos.x)
                    req.ik_request.pose_stamped.pose.position.y = curr_pos.y + percent*(goal_pos.y-curr_pos.y)
                    req.ik_request.pose_stamped.pose.position.z = curr_pos.z + percent*(goal_pos.z-curr_pos.z)
                    #print "Check torso for part-way pose:\r\n %s" %req.ik_request.pose_stamped.pose.position
                    (torso_succeeded, pos) = self.check_torso(req)
                    #print "Pos: %s" %pos
                    if torso_succeeded:
                        print "Successful with Torso, sending new position"
                        self.dist = self.calc_dist(pos)
                        #print "new pose from torso movement: \r\n %s" %pos.pose.position
                        ik_goal = self.ik_pose_proxy(self.form_ik_request(pos))# From pose, get request, get ik
                        return (True, ik_goal)
                    else:
                        rospy.loginfo("Torso Could not reach goal")
                        ik_goal = self.ik_pose_proxy(req)
                        if ik_goal.error_code.val == 1:
                            rospy.loginfo("Initial Goal Out of Reach, Moving as Far as Possible")
                            self.wt_log_out.publish(data="Initial Goal Out of Reach, Moving as Far as Possible")
                            self.dist = self.calc_dist(req.ik_request.pose_stamped)
                            return (True, ik_goal)
                        else:
                            percent -= 0.1
                    
                rospy.loginfo("IK Failed: Error Code %s" %str(ik_goal.error_code))
                self.wt_log_out.publish(data="Inverse Kinematics Failed: Goal Out of Reach.")    
                return (False, ik_goal)

    def calc_dist(self, ps1, ps2=None):
        if ps2 is None:
            print "Calculating distance from Current Pose"
            ps2 = self.curr_pose()

        p1 = ps1.pose.position
        p2 = ps2.pose.position
        wrist_dist = math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

        self.update_frame(ps2)
        ps2.header.stamp=rospy.Time(0)
        np2 = self.tf.transformPose('rh_utility_frame', ps2)
        np2.pose.position.x += 0.168
        self.tf.waitForTransform(np2.header.frame_id, 'torso_lift_link', rospy.Time.now(), rospy.Duration(3.0))
        p2 = self.tf.transformPose('torso_lift_link', np2)
        
        self.update_frame(ps1)
        ps1.header.stamp=rospy.Time(0)
        np1 = self.tf.transformPose('rh_utility_frame', ps1)
        np1.pose.position.x += 0.168
        self.tf.waitForTransform(np1.header.frame_id, 'torso_lift_link', rospy.Time.now(), rospy.Duration(3.0))
        p1 = self.tf.transformPose('torso_lift_link', np1)
        
        p1 = p1.pose.position
        p2 = p2.pose.position
        finger_dist = math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)
        dist = max(wrist_dist, finger_dist)
        print 'Calculated Distance: ', dist
        return dist 

    def form_ik_request(self, ps):
        #print "forming IK request for :%s" %ps
        req = GetPositionIKRequest()
        req.timeout = rospy.Duration(5)
        req.ik_request.pose_stamped = ps 
        req.ik_request.ik_link_name = self.ik_info.kinematic_solver_info.link_names[-1]
        req.ik_request.ik_seed_state.joint_state.name =  self.ik_info.kinematic_solver_info.joint_names
        #self.joint_state_lock.acquire()
        req.ik_request.ik_seed_state.joint_state.position =  self.joint_state_act.positions
        #self.joint_state_lock.release()
        return req
    
    def send_joint_angles(self, angles, duration = None):
        point = JointTrajectoryPoint()
        point.positions = angles
        self.send_traj_point(point, duration)

    def send_traj_point(self, point, time=None):
        if time is None: 
            point.time_from_start = rospy.Duration(max(20*self.dist, 4))
        else:
            point.time_from_start = rospy.Duration(time)
        
        joint_traj = JointTrajectory()
        joint_traj.header.stamp = rospy.Time.now()
        joint_traj.header.frame_id = '/torso_lift_link'
        joint_traj.joint_names = self.ik_info.kinematic_solver_info.joint_names
        joint_traj.points.append(point)
        joint_traj.points[0].velocities = [0,0,0,0,0,0,0]
        #print joint_traj
        
        rarm_goal = JointTrajectoryGoal()
        rarm_goal.trajectory = joint_traj
        self.r_arm_traj_client.send_goal(rarm_goal)

        #self.r_arm_command.publish(joint_traj)

    def move_arm_to(self, goal_in):
        rospy.loginfo("composing move_right_arm goal")

        goal_out = arm_navigation_msgs.msg.MoveArmGoal()

        goal_out.motion_plan_request.group_name = "right_arm"
        goal_out.motion_plan_request.num_planning_attempts = 1
        goal_out.motion_plan_request.planner_id = ""
        goal_out.planner_service_name = "ompl_planning/plan_kinematic_path"
        goal_out.motion_plan_request.allowed_planning_time = rospy.Duration(5.0)
        
        pos = arm_navigation_msgs.msg.PositionConstraint()
        pos.header.frame_id = goal_in.header.frame_id 
        pos.link_name="r_wrist_roll_link"
        pos.position = goal_in.pose.position

        pos.constraint_region_shape.type = 0 
        pos.constraint_region_shape.dimensions=[0.01]

        pos.constraint_region_orientation = Quaternion(0,0,0,1)
        pos.weight = 1

        goal_out.motion_plan_request.goal_constraints.position_constraints.append(pos)
    
        ort = arm_navigation_msgs.msg.OrientationConstraint()    
        ort.header.frame_id=goal_in.header.frame_id
        ort.link_name="r_wrist_roll_link"
        ort.orientation = goal_in.pose.orientation
        
        ort.absolute_roll_tolerance = ort.absolute_pitch_tolerance = ort.absolute_yaw_tolerance = 0.02
        ort.weight = 0.5
        goal_out.motion_plan_request.goal_constraints.orientation_constraints.append(ort)
        rospy.loginfo("sending composed move_right_arm goal")

        finished_within_time = False
        self.move_right_arm_client.send_goal(goal_out)
        finished_within_time = self.move_right_arm_client.wait_for_result(rospy.Duration(60))
        if not (finished_within_time):
            self.move_right_arm_client.cancel_goal()
            self.wt_log_out.publish(data="Timed out achieving right arm goal pose")
            rospy.loginfo("Timed out achieving right arm goal pose")
            return False
        else:
            state = self.move_right_arm_client.get_state()
            result = self.move_right_arm_client.get_result()
            if (state == 3): #3 == SUCCEEDED
                rospy.loginfo("Action Finished: %s" %state)
                self.wt_log_out.publish(data="Move Right Arm with Motion Planning: %s" %self.move_arm_error_dict[result.error_code.val])
                return True
            else:
                if result.error_code.val == 1:
                    rospy.loginfo("Move_right_arm_action failed: Unable to plan a path to goal")
                    self.wt_log_out.publish(data="Move Right Arm with Motion Planning: Failed: Unable to plan a path to the goal")
                elif result.error_code.val == -31:
                    (torso_succeeded, pos) = self.check_torso(self.form_ik_request(goal_in))
                    if torso_succeeded:
                        rospy.loginfo("IK Failed in Current Position. Adjusting Height and Retrying")
                        self.wt_log_out.publish(data="IK Failed in Current Position. Adjusting Height and Retrying")
                        self.move_arm_to(pos)
                    else:
                        rospy.loginfo("Move_right_arm action failed: %s" %state)
                        rospy.loginfo("Failure Result: %s" %result)
                        self.wt_log_out.publish(data="Move Right Arm with Motion Planning and Torso Check Failed: %s" %self.move_arm_error_dict[result.error_code.val])
                else:
                    rospy.loginfo("Move_right_arm action failed: %s" %state)
                    rospy.loginfo("Failure Result: %s" %result)
                    self.wt_log_out.publish(data="Move Right Arm with Motion Planning: Failed: %s" %self.move_arm_error_dict[result.error_code.val])
            return False

if __name__ == '__main__':
    rospy.init_node('arm_utils_right')
    AUL = ArmUtils()
    rospy.spin()
    #r = rospy.Rate(1000)
   # while not rospy.is_shutdown():
       #t1 = rospy.get_time()
       #AUL.curr_pose()
       #t2 = rospy.get_time()
       #print 1./(t2-t1)
       #r.sleep()
