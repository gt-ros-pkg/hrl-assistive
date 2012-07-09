#!/usr/bin/python

import roslib; roslib.load_manifest('web_teleop_trunk')
import rospy
import actionlib
import math
import numpy as np
from geometry_msgs.msg  import PoseStamped, WrenchStamped
import arm_navigation_msgs.msg
import arm_navigation_msgs.msg
from kinematics_msgs.srv import GetKinematicSolverInfo, GetPositionFK, GetPositionFKRequest, GetPositionIK, GetPositionIKRequest
from pr2_controllers_msgs.msg import JointTrajectoryAction, JointTrajectoryControllerState, JointTrajectoryActionGoal, SingleJointPositionAction, SingleJointPositionGoal, Pr2GripperCommandActionGoal, JointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from std_msgs.msg import String, Float32
from tf import TransformListener, transformations, TransformBroadcaster
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTolerance

class MoveArmIntermediary():
  
    wipe_started = False 
    standoff = 0.368 #0.2 + 0.168 (dist from wrist to fingertips)
    frame = 'base_footprint'
    px = py = pz = 0;
    qx = qy = qz = 0;
    qw = 1;

    move_arm_error_dict = {
         -1 : "PLANNING_FAILED: Could not plan a clear path to goal.", 
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

    def __init__(self):
        rospy.init_node('move_right_arm_intermediary')
        self.move_right_arm_client = actionlib.SimpleActionClient('move_right_arm', arm_navigation_msgs.msg.MoveArmAction)
        self.r_arm_traj_client = actionlib.SimpleActionClient('r_arm_controller/joint_trajectory_action', JointTrajectoryAction)
        self.r_arm_follow_traj_client = actionlib.SimpleActionClient('r_arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.torso_client = actionlib.SimpleActionClient('torso_controller/position_joint_action', SingleJointPositionAction)
        

        self.torso_min = 0.0115 
        self.torso_max = 0.295 
        self.dist = 0. 
        self.wipe_ends = [PoseStamped(), PoseStamped()]

        rospy.Subscriber('r_arm_controller/state', JointTrajectoryControllerState , self.set_joint_state) 
        rospy.Subscriber('torso_controller/state', JointTrajectoryControllerState , self.set_torso_state) 
        rospy.Subscriber('norm_approach_right', PoseStamped, self.norm_approach)
        rospy.Subscriber('wt_lin_move_right',Float32, self.linear_move)
        rospy.Subscriber('wt_right_arm_pose_commands', PoseStamped, self.blind_move)
        rospy.Subscriber('wt_right_arm_angle_commands', JointTrajectoryPoint, self.send_traj_point)
        rospy.Subscriber('wt_move_right_arm_goals', PoseStamped, self.move_arm_to)
        rospy.Subscriber('wt_grasp_right_goal', PoseStamped, self.grasp)
        rospy.Subscriber('wt_wipe_right_goals', PoseStamped, self.prep_wipe)
    
        self.pose_out = rospy.Publisher('r_hand_pose', PoseStamped)
        self.wt_log_out = rospy.Publisher('wt_log_out', String )
        self.r_gripper_out = rospy.Publisher('/r_gripper_controller/gripper_action/goal', Pr2GripperCommandActionGoal )
        self.r_arm_command = rospy.Publisher('/r_arm_controller/command', JointTrajectory )

        self.tf = TransformListener()
        self.tfb = TransformBroadcaster()

        rospy.loginfo("Waiting for move_right_arm server")
        self.move_right_arm_client.wait_for_server()
        rospy.loginfo("Move_right_arm server found")

        rospy.loginfo("Waiting for FK/IK Solver services")
        rospy.wait_for_service('/pr2_right_arm_kinematics/get_fk')
        rospy.wait_for_service('/pr2_right_arm_kinematics/get_fk_solver_info')
        rospy.wait_for_service('/pr2_right_arm_kinematics/get_ik')
        rospy.wait_for_service('/pr2_right_arm_kinematics/get_ik_solver_info')
        self.fk_info_proxy = rospy.ServiceProxy('/pr2_right_arm_kinematics/get_fk_solver_info', GetKinematicSolverInfo)
        self.fk_pose_proxy = rospy.ServiceProxy('/pr2_right_arm_kinematics/get_fk', GetPositionFK)    
        self.ik_info_proxy = rospy.ServiceProxy('/pr2_right_arm_kinematics/get_ik_solver_info', GetKinematicSolverInfo)
        self.ik_pose_proxy = rospy.ServiceProxy('/pr2_right_arm_kinematics/get_ik', GetPositionIK)    
        rospy.loginfo("Service Proxies Established")
      
    def set_joint_state(self,msg):
        self.joint_state_act = msg.actual
        self.joint_state_des = msg.desired
        #print "Joint State: %s,%s,%s,%s,%s,%s,%s" %(self.joint_state_act.positions)

    def set_torso_state(self,msg):
        self.torso_state = msg;

    def get_kin_info(self):
        print "getting ik info"
        try:
            self.ik_info = self.ik_info_proxy();
        except rospy.ServiceException, e:
            rospy.loginfo("IK Service did no process request: %s" %str(e))
        #print "IK Info: %s" %self.ik_info
        
        print "getting fk info"
        try:
            self.fk_info = self.fk_info_proxy();
        except rospy.ServiceException, e:
            rospy.loginfo("FK Service did no process request: %s" %str(e))
        #print "FK Info: %s" %self.fk_info

    def get_fk(self, msg, frame='/torso_lift_link'): # get FK pose of a list of joint angles for the arm
        fk_request = GetPositionFKRequest()
        fk_request.header.frame_id = frame
        #fk_request.header.stamp = rospy.Time.now()
        fk_request.fk_link_names =  self.fk_info.kinematic_solver_info.link_names
        fk_request.robot_state.joint_state.position = self.joint_state_act.positions
        fk_request.robot_state.joint_state.name = self.fk_info.kinematic_solver_info.joint_names
        #print "FK Request: %s " %fk_request

        try:
            fk_response = self.fk_pose_proxy(fk_request)
            return fk_response.pose_stamped[-1]
        except rospy.ServiceException, e:
            rospy.loginfo("FK service did not process request: %s" %str(e))
    
    def grasp(self, ps):
        rospy.loginfo("Initiating Grasp Sequence")
        self.wt_log_out.publish(data="Initiating Grasp Sequence")
        self.update_frame(ps)
        approach = self.find_approach(ps)
        rospy.loginfo("approach: \r\n %s" %approach)
        at_appr = self.move_arm_to(approach)
        rospy.loginfo("arrived at approach: %s" %at_appr)
        if at_appr:
            ope = Pr2GripperCommandActionGoal()
            ope.goal.command.position = 0.08
            ope.goal.command.max_effort = -1
            self.r_gripper_out.publish(ope)
            rospy.sleep(3)
            rospy.loginfo("making linear approach")
            lin_mov_goal = Float32()
            lin_mov_goal.data = self.standoff - 0.13
            self.linear_move(lin_mov_goal)
            rospy.sleep(7)
            close = Pr2GripperCommandActionGoal()
            close.goal.command.position = 0
            close.goal.command.max_effort = -1
            self.r_gripper_out.publish(close)
        else:
            pass

    def norm_approach(self, pose):
        self.update_frame(pose)
        appr = self.find_approach(pose)
        self.move_arm_to(appr)

    def update_frame(self, pose):
        self.standoff = 0.368
        self.frame = pose.header.frame_id
        self.px = pose.pose.position.x    
        self.py = pose.pose.position.y    
        self.pz = pose.pose.position.z    
        self.qx = pose.pose.orientation.x
        self.qy = pose.pose.orientation.y
        self.qz = pose.pose.orientation.z
        self.qw = pose.pose.orientation.w

        self.tfb.sendTransform((self.px,self.py,self.pz),(self.qx,self.qy,self.qz,self.qw), rospy.Time.now(), "rh_utility_frame", self.frame)
        #self.find_approach(pose)

    def find_approach(self, msg, stdoff=0.2):
        stdoff += 0.202 #adjust fingertip standoffs for distance to wrist link
        self.pose_in = msg
        self.tf.waitForTransform('rh_utility_frame','base_footprint', rospy.Time(0), rospy.Duration(3.0))
        self.tfb.sendTransform((self.pose_in.pose.position.x, self.pose_in.pose.position.y, self.pose_in.pose.position.z),(self.pose_in.pose.orientation.x, self.pose_in.pose.orientation.y, self.pose_in.pose.orientation.z, self.pose_in.pose.orientation.w), rospy.Time.now(), "rh_utility_frame", self.pose_in.header.frame_id)
        self.tf.waitForTransform('rh_utility_frame','r_wrist_roll_link', rospy.Time.now(), rospy.Duration(3.0))
        goal = PoseStamped()
        goal.header.frame_id = 'rh_utility_frame'
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.z = stdoff
        goal.pose.orientation.x = 0
        goal.pose.orientation.y = 0.5*math.sqrt(2)
        goal.pose.orientation.z = 0
        goal.pose.orientation.w = 0.5*math.sqrt(2)
        #print "Goal:\r\n %s" %goal    
        self.tf.waitForTransform(goal.header.frame_id, 'torso_lift_link', rospy.Time.now(), rospy.Duration(3.0))
        appr = self.tf.transformPose('torso_lift_link', goal)
        #rospy.loginfo("Appr: \r\n %s" %appr)   
        self.wt_log_out.publish(data="Normal Approach with right hand: Trying to move WITH motion planning")
        return appr
        #self.move_arm_out.publish(appr)

    def linear_move(self, msg):
        print "Linear Move: Right Arm: %s m Step" %msg.data
        cp = self.curr_pose
        #cp = self.get_fk(self.joint_state_des.positions)#self.curr_pose
        self.tf.waitForTransform(cp.header.frame_id, '/r_wrist_roll_link', cp.header.stamp, rospy.Duration(3.0))
        newpose = self.tf.transformPose('/r_wrist_roll_link', cp)
        newpose.pose.position.x += msg.data
        step_goal = self.tf.transformPose(cp.header.frame_id, newpose)
        self.dist = msg.data
        self.blind_move(step_goal)
#        return step_goal

    def prep_wipe(self, ps):
        #print "Prep Wipe Received: %s" %pa
        self.update_frame(ps)
        print "Updating frame to: %s \r\n" %ps
        if not self.wipe_started:
            self.wipe_appr_seed = ps
            self.wipe_ends[0] = self.find_approach(ps, 0)
            print "wipe_end[0]: %s" %self.wipe_ends[0]
            (reachable, ik_goal) = self.full_ik_check(self.wipe_ends[0])
            if not reachable:
                rospy.loginfo("Cannot find approach for initial wipe position, please try another")
                self.wt_log_out.publish(data="Cannot find approach for initial wipe position, please try another")
                return
            else:
                self.wipe_started = True
                rospy.loginfo("Received starting position for wiping action")
                self.wt_log_out.publish(data="Received starting position for wiping action")
        else:
            self.wipe_ends[1] = self.find_approach(ps, 0)
            self.wipe_ends.reverse()
            (reachable, ik_goal) = self.full_ik_check(self.wipe_ends[1])
            if not reachable:
                rospy.loginfo("Cannot find approach for final wipe position, please try another")
                self.wt_log_out.publish(data="Cannot find approach for final wipe position, please try another")
                return
            else:
                rospy.loginfo("Received End position for wiping action")
                self.wt_log_out.publish(data="Received End position for wiping action")
                self.update_frame(self.wipe_ends[0])

                self.wipe_ends[1].header.stamp = rospy.Time.now()
                self.tf.waitForTransform(self.wipe_ends[1].header.frame_id, 'rh_utility_frame', rospy.Time.now(), rospy.Duration(3.0))
                fin_in_start = self.tf.transformPose('rh_utility_frame', self.wipe_ends[1])
                
                ang = math.atan2(-fin_in_start.pose.position.z, -fin_in_start.pose.position.y)+(math.pi/2)
                q_st_rot = transformations.quaternion_about_axis(ang, (1,0,0))
                q_st_new = transformations.quaternion_multiply([self.wipe_ends[0].pose.orientation.x, self.wipe_ends[0].pose.orientation.y, self.wipe_ends[0].pose.orientation.z, self.wipe_ends[0].pose.orientation.w],q_st_rot)
                self.wipe_ends[0].pose.orientation.x = q_st_new[0]
                self.wipe_ends[0].pose.orientation.y = q_st_new[1]
                self.wipe_ends[0].pose.orientation.z = q_st_new[2]
                self.wipe_ends[0].pose.orientation.w = q_st_new[3]

                self.update_frame(self.wipe_ends[1])
                self.wipe_ends[0].header.stamp = rospy.Time.now()
                self.tf.waitForTransform(self.wipe_ends[0].header.frame_id, 'rh_utility_frame', rospy.Time.now(), rospy.Duration(3.0))
                start_in_fin = self.tf.transformPose('rh_utility_frame', self.wipe_ends[0])
                ang = math.atan2(start_in_fin.pose.position.z, start_in_fin.pose.position.y)+(math.pi/2)
                
                q_st_rot = transformations.quaternion_about_axis(ang, (1,0,0))
                q_st_new = transformations.quaternion_multiply([self.wipe_ends[1].pose.orientation.x, self.wipe_ends[1].pose.orientation.y, self.wipe_ends[1].pose.orientation.z, self.wipe_ends[1].pose.orientation.w],q_st_rot)
                self.wipe_ends[1].pose.orientation.x = q_st_new[0]
                self.wipe_ends[1].pose.orientation.y = q_st_new[1]
                self.wipe_ends[1].pose.orientation.z = q_st_new[2]
                self.wipe_ends[1].pose.orientation.w = q_st_new[3]
                self.wipe_started = False

                self.update_frame(self.wipe_appr_seed)
                appr = self.find_approach(self.wipe_appr_seed, 0.15)
                appr.pose.orientation = self.wipe_ends[0].pose.orientation
                prepared = self.move_arm_to(appr)
                if prepared:
                    self.wipe(self.wipe_ends[0], self.wipe_ends[1])
                else:
                    rospy.loginfo("Cannot reach start point, please choose another")
                    self.wt_log_out.publish(data="Cannot reach start point, please choose another")

    def wipe(self, start, finish):
            dist = int(round(200*self.calc_dist(start, finish))) # 1 point per cm of separation
            print "Steps: %s" %dist
            x_step = finish.pose.position.x - start.pose.position.x
            y_step = finish.pose.position.y - start.pose.position.y
            z_step = finish.pose.position.z - start.pose.position.z
            #print "Increments: %s,%s,%s" %(x_step, y_step, z_step)

            qs = [start.pose.orientation.x, start.pose.orientation.y, start.pose.orientation.z, start.pose.orientation.w] 
            qf = [finish.pose.orientation.x, finish.pose.orientation.y, finish.pose.orientation.z, finish.pose.orientation.w] 
            steps = []
            #print "Start: %s" %start
            #print "Finish: %s" %finish
            for i in range(dist):
                frac = float(i)/float(dist)
                steps.append(PoseStamped())
                steps[i].header.stamp = rospy.Time.now()
                steps[i].header.frame_id = start.header.frame_id
                steps[i].pose.position.x = start.pose.position.x + x_step*frac
                steps[i].pose.position.y = start.pose.position.y + y_step*frac
                steps[i].pose.position.z = start.pose.position.z + z_step*frac
                new_q = transformations.quaternion_slerp(qs,qf,frac)
                steps[i].pose.orientation.x = new_q[0]
                steps[i].pose.orientation.y = new_q[1]
                steps[i].pose.orientation.z = new_q[2]
                steps[i].pose.orientation.w = new_q[3]
            steps.append(finish)
            #print "Steps:"
            #print steps
            #raw_input("Press Enter to continue")
            rospy.loginfo("Planning straight-line path, please wait")
            self.wt_log_out.publish(data="Planning straight-line path, please wait")
           
            rospy.loginfo("Initiating wipe action")
            self.blind_move(finish)
            self.r_arm_traj_client.wait_for_result(rospy.Duration(20))
            rospy.loginfo("At beginning of path")
            pts = []
            
            for i, p in enumerate(steps):
                frac = float(i)/float(len(steps))
                request = self.form_ik_request(p)
                if not i == 0:
                    request.ik_request.ik_seed_state.joint_state.position = seed
                ik_goal = self.ik_pose_proxy(request)
                pts.append(ik_goal.solution.joint_state.position)
                seed = pts[i]

                
            points = []    
            for i in range(len(pts)-1):
                angs1 = pts[i]
                angs2 = pts[i+1]
                increm = np.subtract(angs2, angs1) 
                for j in range(10):
                    points.append(np.add(angs1, np.multiply(0.1*j, increm)))
            
            #points = np.unwrap(points,1)
            p1= points
            traj = JointTrajectory()
            traj.header.frame_id = steps[0].header.frame_id
            traj.joint_names = self.ik_info.kinematic_solver_info.joint_names
            times = []
            for i in range(len(points)):
                frac = float(i)/float(len(points))
                traj.points.append(JointTrajectoryPoint())
                traj.points[i].positions = points[i]
                traj.points[i].velocities = [0]*7
                times.append(rospy.Duration(frac*dist*0.2))

            traj_goal = FollowJointTrajectoryGoal()
            traj_goal.trajectory = traj
            tolerance = JointTolerance()
            tolerance.position = 0.05
            tolerance.velocity = 0.1
            traj_goal.path_tolerance = [tolerance for i in range(len(traj.points))]
            traj_goal.goal_tolerance = [tolerance for i in range(len(traj.points))]
            traj_goal.goal_time_tolerance = rospy.Duration(3)

            #print "Steps: %s" %steps
            count = 0

            while count < 6:
                traj_goal.trajectory.points.reverse()
                for i in range(len(times)):
                    traj_goal.trajectory.points[i].time_from_start = times[i]
                if count == 0:
                    print traj_goal.trajectory.points
                    raw_input("Review Trajectory Goal")
                traj_goal.trajectory.header.stamp = rospy.Time.now()
                self.r_arm_follow_traj_client.send_goal(traj_goal)
                self.r_arm_follow_traj_client.wait_for_result(rospy.Duration(20))
                rospy.sleep(0.5)
                count += 1
            
            
            #traj_goal = JointTrajectoryGoal()
            #traj_goal.trajectory = traj
            #print "Steps: %s" %steps
            #count = 0
#
            #while count < 6:
                #traj_goal.trajectory.points.reverse()
                #for i in range(len(times)):
                    #traj_goal.trajectory.points[i].time_from_start = times[i]
                #print traj_goal
                #raw_input("Review Trajectory Goal")
                ##print "Traj goal start:"
                ##print traj_goal.trajectory.points[0].positions
                ##print "Traj goal end:"
                ##print traj_goal.trajectory.points[-1].positions
                #traj_goal.trajectory.header.stamp = rospy.Time.now()
                #self.r_arm_traj_client.send_goal(traj_goal)
                #self.r_arm_traj_client.wait_for_result(rospy.Duration(20))
                #rospy.sleep(0.5)
                #count += 1
            rospy.loginfo("Done Wiping")
            self.wt_log_out.publish(data="Done Wiping")
            self.linear_move(Float32(-0.15))
    
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
        spine_range = [self.torso_min - self.torso_state.actual.positions[0], self.torso_max - self.torso_state.actual.positions[0]]
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
                streached = self.move_torso(self.torso_state.actual.positions[0]+0.1*trial*streach_goal)
                return [True, request.ik_request.pose_stamped]
            else:
                if trial < 10:
                    trial += 1
                    print "Trial %s" %trial
                else:
                    return [False, request.ik_request.pose_stamped]
    
    def blind_move(self, ps):
        (reachable, ik_goal) = self.full_ik_check(ps)
        if reachable:
            self.send_joint_angles(ik_goal.solution.joint_state.position)

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
                    curr_pos = self.curr_pose.pose.position
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

    def calc_dist(self, ps1, ps2=False):
        if not ps2:
            p2 = self.curr_pose.pose.position
        else:
            p2 = ps2.pose.position
        p1 = ps1.pose.position
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

    def form_ik_request(self, ps):
        #print "forming IK request for :%s" %ps
        req = GetPositionIKRequest()
        req.timeout = rospy.Duration(5)
        req.ik_request.pose_stamped = ps 
        req.ik_request.ik_link_name = self.ik_info.kinematic_solver_info.link_names[-1]
        req.ik_request.ik_seed_state.joint_state.name =  self.ik_info.kinematic_solver_info.joint_names
        req.ik_request.ik_seed_state.joint_state.position =  self.joint_state_act.positions
        return req
    
    def send_joint_angles(self, angles):
        point = JointTrajectoryPoint()
        point.positions = angles
        self.send_traj_point(point)

    def send_traj_point(self, point):
        point.time_from_start = rospy.Duration(max(20*self.dist, 4))
        #point.time_from_start += rospy.Duration(2.5*abs(self.joint_state_act.positions[4]-point.positions[4])+ 2.5*abs(self.joint_state_act.positions[5]-point.positions[5]) + 2*abs(self.joint_state_act.positions[6]-point.positions[6]))
        
        joint_traj = JointTrajectory()
        joint_traj.header.stamp = rospy.Time.now()
        joint_traj.header.frame_id = '/torso_lift_link'
        joint_traj.joint_names = self.ik_info.kinematic_solver_info.joint_names
        joint_traj.points.append(point)
        joint_traj.points[0].velocities = [0,0,0,0,0,0,0]
        print joint_traj
        
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
        pos.position.x = goal_in.pose.position.x 
        pos.position.y = goal_in.pose.position.y
        pos.position.z = goal_in.pose.position.z

        pos.constraint_region_shape.type = 0 
        pos.constraint_region_shape.dimensions=[0.05]

        pos.constraint_region_orientation.x = 0
        pos.constraint_region_orientation.y = 0
        pos.constraint_region_orientation.z = 0
        pos.constraint_region_orientation.w = 1
        pos.weight = 1

        goal_out.motion_plan_request.goal_constraints.position_constraints.append(pos)
    
        ort = arm_navigation_msgs.msg.OrientationConstraint()    
        ort.header.frame_id=goal_in.header.frame_id
        ort.link_name="r_wrist_roll_link"
        ort.orientation.x = goal_in.pose.orientation.x
        ort.orientation.y = goal_in.pose.orientation.y
        ort.orientation.z = goal_in.pose.orientation.z
        ort.orientation.w = goal_in.pose.orientation.w
        
        ort.absolute_roll_tolerance = 0.04
        ort.absolute_pitch_tolerance = 0.04
        ort.absolute_yaw_tolerance = 0.04
        ort.weight = 0.5

        goal_out.motion_plan_request.goal_constraints.orientation_constraints.append(ort)
        rospy.loginfo("sending composed move_right_arm goal")

        finished_within_time = False
        self.move_right_arm_client.send_goal(goal_out)
        finished_within_time = self.move_right_arm_client.wait_for_result(rospy.Duration(30))
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
    MAI = MoveArmIntermediary()
    MAI.get_kin_info()

    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        curpose = MAI.get_fk(MAI.joint_state_act.positions)
        MAI.curr_pose = curpose
        MAI.pose_out.publish(curpose)
        MAI.tfb.sendTransform((MAI.px,MAI.py,MAI.pz),(MAI.qx,MAI.qy,MAI.qz,MAI.qw), rospy.Time.now(), "rh_utility_frame", MAI.frame)
        r.sleep()
