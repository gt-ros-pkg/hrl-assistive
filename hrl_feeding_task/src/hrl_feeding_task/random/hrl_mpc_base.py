#!/usr/bin/env python

# System library
import numpy as np, math
import time
import threading, copy
import sys

# ROS library
import roslib; roslib.load_manifest('sandbox_dpark_darpa_m3')
import rospy
import tf
from tf import transformations
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PoseArray#PointStamped,
from std_msgs.msg import Bool, Empty, Int32
import hrl_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import std_msgs.msg
from pykdl_utils.kdl_kinematics import create_kdl_kin

# For action lib
from kinematics_msgs.srv import GetKinematicSolverInfo, GetPositionFK, GetPositionFKRequest, GetPositionIK, GetPositionIKRequest
import trajectory_msgs.msg
import actionlib
import pr2_controllers_msgs.msg


# HRL library
from hrl_haptic_manipulation_in_clutter_srvs.srv import HapticMPCLogAndMonitorInfo, HapticMPCLogAndMonitorInfoRequest, EnableService
import hrl_haptic_manipulation_in_clutter_msgs.msg as haptic_msgs #CONTAINS ROBOTIC STATE MESSAGE TYPE
import hrl_lib.quaternion as qt
import hrl_lib.circular_buffer as cb

class mpcBaseAction():
    def __init__(self, d_robot, controller, arm='l'):
        rospy.loginfo("mpcBaseAction is initialized.")

        self.d_robot = d_robot
        self.controller = controller
        self.arm   = arm

        if d_robot == "pr2" or d_robot == "pr2sim":
            self.robot_path = "/pr2"
        elif d_robot == "darci_sim":
            self.robot_path = "/darci_sim"
        elif d_robot == "darci":
            self.robot_path = "/darci"
        elif d_robot == "sim3" or d_robot == "sim3_nolim":
            self.robot_path = "/sim3"

        if self.arm == 'l':
            self.arm_name = 'left'
        else:
            self.arm_name = 'right'

        # Common variables
        self.joint_angles = []
        self.jstate_lock = threading.RLock() ## joint state lock

        self.stop_mpc = False
        self.stuck_mpc = False

        # For moving motion check
        self.hist_size = 20000
        self.n_jts = 7
        self.q_buf = cb.CircularBuffer(self.hist_size, (self.n_jts,))
        self.jts_gaussian_mn_buf_50 = cb.CircularBuffer(self.hist_size, (self.n_jts,))
        self.jts_gaussian_cov_buf_50 = cb.CircularBuffer(self.hist_size, (self.n_jts,self.n_jts))
        self.mean_motion_buf = cb.CircularBuffer(self.hist_size, ())


        if self.controller == "static" or self.controller == "dynamic":

            # define lock
            self.state_lock = threading.RLock() ## Haptic state lock
            self.traj_lock = threading.RLock() ## Haptic state lock
            self.tracking_lock = threading.RLock() ## Haptic state lock

            # Haptic State parameters
            #self.msg = None ## Haptic state message
            self.last_msg_time = None

            self.desired_joint_angles = []
            self.joint_velocities = []
            self.joint_stiffness = []
            self.joint_damping = []

            self.end_effector_pos = None
            self.end_effector_orient_quat = None


            self.current_pose_trajectory_msg = None
            self.current_joint_trajectory_msg = None
            self.track_err = 0

            # planner
            self.plan_count = 0

        elif self.controller == 'actionlib':
            self.armClient = actionlib.SimpleActionClient("/"+self.arm+"_arm_controller"+ \
                                                          "/joint_trajectory_action", \
                                                          pr2_controllers_msgs.msg.JointTrajectoryAction)
            self.armClient.wait_for_server()
            rospy.logout('Connected to joint control server')

            msg=  rospy.wait_for_message("/"+self.arm+"_arm_controller"+ "/state", \
                                         pr2_controllers_msgs.msg.JointTrajectoryControllerState)
            self.joint_names_list = msg.joint_names
            rospy.logout('Connected to controller state server')



        # get params
        self.getParams()

        # set communications
        self.initComms()

        if self.controller is not 'actionlib' and self.d_robot.find('sim') >= 0:
            self.arm_kdl = create_kdl_kin(self.torso_frame, self.ee_frame)
            # joint limit setting!!

        pass

    def getParams(self):

        self.ee_motion_threshold = rospy.get_param('haptic_mpc'+self.robot_path+'/ee_motion_threshold')
        self.ee_orient_motion_threshold = rospy.get_param('haptic_mpc'+self.robot_path+'/ee_orient_motion_threshold')
        self.jts_motion_threshold = rospy.get_param('haptic_mpc'+self.robot_path+'/jts_motion_threshold')

        if self.d_robot == 'darci':
            self.torso_frame = 'torso_lift_link'
        else:
            self.torso_frame = rospy.get_param('haptic_mpc'+self.robot_path+'/torso_frame' )
        self.ee_frame = rospy.get_param('haptic_mpc'+self.robot_path+'/end_effector_frame' )

        groups = rospy.get_param('haptic_mpc/groups' )
        for group in groups:
            if group['name'] == 'left_arm_joints' and self.arm == 'l':
                self.joint_names_list = group['joints']
            elif group['name'] == 'right_arm_joints' and self.arm == 'r':
                self.joint_names_list = group['joints']

        rospy.logout('Loaded parameters')


    ## Initialise the ROS communications - init node, subscribe to the robot state and goal pos, publish JEPs
    def initComms(self, mPos=None):
        #rospy.init_node(node_name)

        if self.controller == "static" or self.controller == "dynamic":

            # MPC Command Publisher
            self.goal_traj_pub = rospy.Publisher("haptic_mpc/goal_pose_array", PoseArray)
            self.goal_traj_pub.publish(PoseArray())
            self.goal_pos_pub  = rospy.Publisher("haptic_mpc/goal_pose", PoseStamped, latch=True)

            # Planner Command Publisher
            self.planner_goal_pub      = rospy.Publisher('hrl_planner/goal_pose', PoseStamped, latch=True)
            self.enable_joint_plan_pub = rospy.Publisher('hrl_planner/joint_space/enable', Bool)
            self.enable_task_plan_pub  = rospy.Publisher('hrl_planner/task_space/enable', Bool)
            self.joint_traj_pub = rospy.Publisher('haptic_mpc/joint_trajectory', JointTrajectory, latch=True)
            self.pose_traj_pub  = rospy.Publisher('haptic_mpc/pose_trajectory', PoseArray, latch=True)
            self.clear_buf_pub  = rospy.Publisher('haptic_mpc/log_and_monitor_clear', Bool)

            # Subscriber
            rospy.Subscriber('hrl_planner/joint_trajectory', JointTrajectory, self.jointTrajectoryCallback)
            rospy.Subscriber('hrl_planner/pose_trajectory', PoseArray, self.poseTrajectoryCallback)
            rospy.Subscriber('haptic_mpc/track_err', std_msgs.msg.Float64, self.trackingErrorCallback)
            rospy.Subscriber("haptic_mpc/mpc_action/stop", Bool, self.stopMPCCallback)
            rospy.Subscriber("haptic_mpc/mpc_action/stuck_info", Bool, self.stuckMPCCallback)

        else:
            self.fk_info_proxy = rospy.ServiceProxy('/pr2_'+self.arm_name+'_arm_kinematics/get_fk_solver_info', GetKinematicSolverInfo)
            rospy.wait_for_service('/pr2_'+self.arm_name+'_arm_kinematics/get_fk_solver_info')

            self.fk_pose_proxy = rospy.ServiceProxy('/pr2_'+self.arm_name+'_arm_kinematics/get_fk', GetPositionFK)
            rospy.wait_for_service('/pr2_'+self.arm_name+'_arm_kinematics/get_fk')

            self.ik_info_proxy = rospy.ServiceProxy('/pr2_'+self.arm_name+'_arm_kinematics/get_ik_solver_info', \
                                                    GetKinematicSolverInfo)
            rospy.wait_for_service('/pr2_'+self.arm_name+'_arm_kinematics/get_ik_solver_info')

            self.ik_pose_proxy = rospy.ServiceProxy('/pr2_'+self.arm_name+'_arm_kinematics/get_ik', GetPositionIK)
            rospy.wait_for_service('/pr2_'+self.arm_name+'_arm_kinematics/get_ik')

            self.fk_info = self.fk_info_proxy();


        ## Check-1: current state
        if self.controller == "static":
            self.goal_posture_pub = rospy.Publisher("haptic_mpc/goal_posture", hrl_msgs.msg.FloatArray)
            rospy.Subscriber("haptic_mpc/robot_state", haptic_msgs.RobotHapticState, self.robotStateCallback)
            self.mpc_weights_pub = rospy.Publisher("haptic_mpc/weights", haptic_msgs.HapticMpcWeights)
        elif self.controller == "dynamic":
            self.goal_posture_pub = rospy.Publisher("haptic_mpc/goal_posture", hrl_msgs.msg.FloatArrayBare)
            rospy.Subscriber("/joint_states", JointState, self.jointStatesCallback)
            self.mpc_weights_pub = rospy.Publisher("haptic_mpc/goal_weight", haptic_msgs.HapticMpcWeights)

        else:
            rospy.Subscriber('/'+self.arm+'_arm_controller/state', \
                             pr2_controllers_msgs.msg.JointTrajectoryControllerState , \
                             self.controllerJointStateCallback)

        ## Check-2: current state
        if not (self.d_robot.find("sim") >= 0):
            try:
                self.tf_lstnr = tf.TransformListener()
            except rospy.ServiceException, e:
                rospy.loginfo("ServiceException caught while instantiating a TF listener. Seems to be normal")
                pass
        else:
            print "No listner is implemented"
            try:
                self.tf_lstnr = tf.TransformListener()
            except rospy.ServiceException, e:
                rospy.loginfo("ServiceException caught while instantiating a TF listener. Seems to be normal")
                pass

        # ETC
        self.close_pub = rospy.Publisher('close_gripper', Empty)
        if mPos!=None:
            self.setStopPos(mPos)

        rospy.logout('Initialized communications')


    def clear_buf(self):
        self.q_buf.clear()
        self.jts_gaussian_mn_buf_50.clear()
        self.jts_gaussian_cov_buf_50.clear()
        self.mean_motion_buf.clear()


    # -----------------------------------------------------------------------------------------

    ## Store the robot haptic state.
    # @param msg RobotHapticState message object
    def robotStateCallback(self, msg):
        with self.state_lock:
            self.last_msg_time = rospy.Time.now() # timeout for the controller

            #self.msg = msg
            self.joint_angles = list(msg.joint_angles)
            self.q_buf.append(self.joint_angles)

            ## self.desired_joint_angles = list(msg.desired_joint_angles)
            self.joint_velocities= list(msg.joint_velocities)
            ## self.joint_stiffness = list(msg.joint_stiffness)
            ## self.joint_damping = list(msg.joint_damping)

            self.end_effector_pos = np.matrix([[msg.hand_pose.position.x], [msg.hand_pose.position.y], [msg.hand_pose.position.z]])
            self.end_effector_orient_quat = [msg.hand_pose.orientation.x, msg.hand_pose.orientation.y, msg.hand_pose.orientation.z, msg.hand_pose.orientation.w]

            #self.skin_data = msg.skins
            #self.Je = self.ma_to_m.multiArrayToMatrixList(msg.end_effector_jacobian)
            #self.Jc = self.ma_to_m.multiArrayToMatrixList(msg.contact_jacobians)

    ##
    # Callback for /joint_states topic. Updates current joint
    # angles and efforts for the arms constantly
    # @param data JointState message recieved from the /joint_states topic
    def jointStatesCallback(self, data):
        joint_angles = []
        ## joint_efforts = []
        joint_vel = []
        jt_idx_list = [0]*len(self.joint_names_list)
        for i, jt_nm in enumerate(self.joint_names_list):
            jt_idx_list[i] = data.name.index(jt_nm)

        for i, idx in enumerate(jt_idx_list):
            if data.name[idx] != self.joint_names_list[i]:
                raise RuntimeError('joint angle name does not match.')
            joint_angles.append(data.position[idx])
            ## joint_efforts.append(data.effort[idx])
            joint_vel.append(data.velocity[idx])

        with self.jstate_lock:
            self.joint_angles  = joint_angles
            ## self.joint_efforts = joint_efforts
            self.joint_velocities = joint_vel
            self.q_buf.append(self.joint_angles)


    def controllerJointStateCallback(self, msg):
        with self.jstate_lock:
            self.joint_angles = msg.actual.positions;
            self.q_buf.append(self.joint_angles)

    # -----------------------------------------------------------------------------------------

    def setPositionControl(self):
        self.weights_msg = haptic_msgs.HapticMpcWeights()
        self.weights_msg.header.stamp = rospy.Time.now()
        self.weights_msg.position_weight = 5.0
        self.weights_msg.orient_weight   = 0.0
        self.weights_msg.posture_weight  = 0.0
        self.mpc_weights_pub.publish(self.weights_msg) # Enable position tracking only - disable orientation by setting the weight to 0

    def setOrientationControl(self):
        self.weights_msg = haptic_msgs.HapticMpcWeights()
        self.weights_msg.header.stamp = rospy.Time.now()
        self.weights_msg.position_weight = 5.0
        self.weights_msg.orient_weight   = 5.0
        self.weights_msg.posture_weight  = 0.0
        self.mpc_weights_pub.publish(self.weights_msg) # Enable position and orientation tracking

    def setPostureControl(self):
        self.weights_msg = haptic_msgs.HapticMpcWeights()
        self.weights_msg.header.stamp = rospy.Time.now()
        self.weights_msg.position_weight = 0.0
        self.weights_msg.orient_weight   = 0.0
        self.weights_msg.posture_weight  = 5.0
        self.mpc_weights_pub.publish(self.weights_msg) # Enable position and orientation tracking

    # -----------------------------------------------------------------------------------------

    def setPositionGoal(self, pos, orient_quat, timeout, frame_id = '/torso_lift_link'):
        if self.controller == 'static' or self.controller == 'dynamic':

            self.setPositionControl()

            ps = PoseStamped()
            ps.header.frame_id  = frame_id
            ps.pose.position    = pos
            ps.pose.orientation = orient_quat
            ps = self.tf_lstnr.transformPose('/torso_lift_link', ps)
            self.current_goal   = ps

            # Send a goal
            self.goal_pos_pub.publish(self.current_goal)
            return self.checkMovement(0.001, timeout, goal_type='position')

        else:
            print "under construction"

    def setOrientGoal(self, pos, orient_quat, timeout, frame_id = '/torso_lift_link'):

        ps = PoseStamped()
        ps.header.frame_id  = frame_id
        ps.pose.position    = pos
        ps.pose.orientation = orient_quat
        ps = self.tf_lstnr.transformPose('/torso_lift_link', ps)
        self.current_goal   = ps

        if self.controller == 'static' or self.controller == 'dynamic':
            self.setOrientationControl()
            # Send a goal
            self.goal_pos_pub.publish(self.current_goal)
        else:
            self.get_ik(self.current_goal, timeout)

        return self.checkMovement(0.001, timeout, goal_type='orient')


    def setPostureGoal(self, lJoint, timeout):
        # Send a goal
        if self.controller == 'static' or self.controller == 'dynamic':
            self.setPostureControl()

            ps = hrl_msgs.msg.FloatArray()
            ps.data = lJoint
            self.current_goal = ps

            self.goal_posture_pub.publish(ps)
        else:
            trajMsg = pr2_controllers_msgs.msg.JointTrajectoryGoal()
            trajPoint = trajectory_msgs.msg.JointTrajectoryPoint()
            trajPoint.positions       = lJoint
            trajPoint.velocities      = [0 for i in range(0, len(self.joint_names_list))]
            trajPoint.accelerations   = [0 for i in range(0, len(self.joint_names_list))]
            trajPoint.time_from_start = rospy.Duration(timeout)

            trajMsg.trajectory.joint_names = self.joint_names_list
            trajMsg.trajectory.points.extend([trajPoint])

            self.armClient.send_goal(trajMsg)

            ps = hrl_msgs.msg.FloatArray()
            ps.data = lJoint
            self.current_goal = ps

        return self.checkMovement(0.001, timeout, goal_type='posture')


    def setPlannerGoal(self, pos, quat, timeout):

        ps = PoseStamped()
        ps.header.frame_id  = '/torso_lift_link'
        ps.pose.position    = pos
        ps.pose.orientation = quat
        self.current_goal   = ps

        # Send a goal to planner
        self.planner_goal_pub.publish(ps)
        ## return self.checkMovement(0.001, timeout, True)
        return '', None

    # -----------------------------------------------------------------------------------------

    def getPositionPlan(self, goal_pose):
        self.current_pose_trajectory_msg = None
        self.enable_task_plan_pub.publish(True)

        pos  = goal_pose.pose.position
        quat = goal_pose.pose.orientation
        single_reach_timeout = 15.0

        stop, ea = self.setPlannerGoal(pos, quat, single_reach_timeout*3)

        # Check elapsed time
        task_last_time = rospy.Time.now()

        # Get a trajectory
        rate = rospy.Rate(100) # 25Hz, nominally.
        while not rospy.is_shutdown():
            if self.current_pose_trajectory_msg != None:
                break
            else:
                current_time = rospy.Time.now()
                delta_secs = current_time.secs - task_last_time.secs
                if delta_secs > single_reach_timeout*20:
                    print "Waiting trajectory time out"
                    break

            rate.sleep()

        # Finish planner
        self.enable_task_plan_pub.publish(False)

        return self.current_pose_trajectory_msg

    def getOrientPlan(self, goal_pose):
        print "Under construction"

    def getJointPlan(self, goal_pose):
        # Start planner
        self.current_joint_trajectory_msg = None
        self.enable_joint_plan_pub.publish(True)

        pos  = goal_pose.pose.position
        quat = goal_pose.pose.orientation
        single_reach_timeout = 15.0

        stop, ea = self.setPlannerGoal(pos, quat, single_reach_timeout*3)

        # Check elapsed time
        task_last_time = rospy.Time.now()

        # Get a trajectory
        rate = rospy.Rate(100) # 25Hz, nominally.
        while not rospy.is_shutdown():
            if self.current_joint_trajectory_msg != None:
                break
            else:
                current_time = rospy.Time.now()
                delta_secs = current_time.secs - task_last_time.secs
                if delta_secs > single_reach_timeout*20:
                    print "Waiting trajectory time out"
                    break

            rate.sleep()

        # Finish planner
        self.enable_joint_plan_pub.publish(False)

        return self.current_joint_trajectory_msg


    # -----------------------------------------------------------------------------------------

    def goRePlannerMovement(self, goal_pose, timeout, single_task_reach_timeout, single_joint_reach_timeout, track_pos_err_limit, track_jnt_err_limit, plantype):
        rospy.loginfo("Start Re Planner Movement.")

        pos  = goal_pose.pose.position
        quat = goal_pose.pose.orientation
        self.plan_count = 1

        # Start to move
        last_time = rospy.Time.now()

        while True and not rospy.is_shutdown():

            # Finish condition
            current_time = rospy.Time.now()
            delta_secs = current_time.secs - last_time.secs
            if delta_secs > timeout:
                stop = 'stop by over time: ' , timeout
                break

            if plantype.find('taskplan') >= 0 or plantype.find('combiplan') >= 0:
                self.current_pose_trajectory_msg = None
                self.enable_task_plan_pub.publish(True)

                print "--------------------------------"
                print "Start task plan!!"

                stop, ea = self.setPlannerGoal(pos, quat, single_task_reach_timeout)
                self.setPositionControl()

                # Get a trajectory
                rate      = rospy.Rate(100) # 25Hz, nominally.
                task_last_time = rospy.Time.now()
                stop      = ''

                self.clear_buf_pub.publish(True)
                while not rospy.is_shutdown() and stop == '':

                    # Check elapsed time
                    current_time = rospy.Time.now()
                    delta_secs = current_time.secs - task_last_time.secs
                    if delta_secs > single_task_reach_timeout:
                        print "single reach time out"
                        break

                    if self.current_pose_trajectory_msg != None:

                        # Send the trajectory to MPC (waypoint_generator)
                        ## self.pose_traj_pub.publish(self.current_trajectory_msg)
                        goal_pose_traj_msg = self.current_pose_trajectory_msg

                        # If the start of trajectory is too different with current position, continue
                        cur_pos = copy.copy(self.getEndeffectorPose()[0])
                        if cur_pos[0,0] != None:

                            start_pose = self.current_pose_trajectory_msg.poses[0]
                            start_pos = np.matrix([start_pose.position.x,
                                                   start_pose.position.y,
                                                   start_pose.position.z]).T

                            err     = start_pos - cur_pos
                            err_mag = np.linalg.norm(err)
                            if err_mag > 0.08:
                                ## print "provided trajectory is too far from current"
                                continue

                        # get trajectory
                        goal_pose_traj_msg.poses = self.current_pose_trajectory_msg.poses
                        self.pose_traj_pub.publish(goal_pose_traj_msg)

                        ## self.goal_pos_pub.publish(ps)
                        self.current_pose_trajectory_msg = None
                    else:
                        continue

                    stop, ea = self.checkMovement(0.001, single_task_reach_timeout, track_pos_err_limit, \
                                                  current_time, True, False)

                    if stop == 'stop by stuck mpc from log and monitor':
                        stop = ''

                    rate.sleep()

                # Finish planner
                self.enable_task_plan_pub.publish(False)

                # Check movement status.
                ## stop, ea = self.checkMovement(0.001, single_reach_timeout, track_pos_err_limit, True)
                print stop
                if stop == 'Reached': break

            if plantype.find('jointplan') >= 0 or plantype.find('combiplan') >= 0:
                self.current_joint_trajectory_msg = None
                self.enable_joint_plan_pub.publish(True)

                print "--------------------------------"
                print "Start joint plan!!"

                stop, ea = self.setPlannerGoal(pos, quat, single_joint_reach_timeout)
                self.setPostureControl()

                # Check elapsed time
                task_last_time = rospy.Time.now()

                # Get a trajectory
                rate = rospy.Rate(100) # 25Hz, nominally.
                while not rospy.is_shutdown():
                    if self.current_joint_trajectory_msg != None:
                        break
                    else:
                        current_time = rospy.Time.now()
                        delta_secs = current_time.secs - task_last_time.secs
                        if delta_secs > single_joint_reach_timeout:
                        ## if delta_secs > single_reach_timeout*20:
                            print "Waiting trajectory time out"
                            break

                    rate.sleep()

                if delta_secs > single_joint_reach_timeout:
                    continue

                # Send the trajectory to MPC (waypoint_generator)
                self.clear_buf_pub.publish(True)
                self.clear_buf_pub.publish(True)

                ps = hrl_msgs.msg.FloatArray()
                ps.data = list(self.current_joint_trajectory_msg.points[-1].positions)
                self.current_goal = ps

                self.joint_traj_pub.publish(self.current_joint_trajectory_msg)
                self.plan_count = self.plan_count + 1

                # Finish planner
                self.current_joint_trajectory_msg = None
                self.enable_joint_plan_pub.publish(False)

                # Check movement status.
                joint_last_time = rospy.Time.now()
                stop, ea = self.checkMovement(0.001, single_joint_reach_timeout, track_jnt_err_limit, \
                                              joint_last_time, planner=True, goal_type='posture')
                print stop

                if stop == 'Reached': break


        current_time = rospy.Time.now()
        delta_secs = current_time.secs - task_last_time.secs
        print "======================================================"
        print "Elapsed time: ", delta_secs
        print "======================================================"

        return stop, ea





    def setJointTraj(self, pos, traj, timeout):

        ps = PoseStamped()
        ps.header.frame_id  = '/torso_lift_link'
        ps.pose.position    = pos
        self.current_goal   = ps

        trajectory = JointTrajectory()

        for i in xrange(len(traj)):
            jtp = JointTrajectoryPoint()
            jtp.positions = traj[i]
            jtp.time_from_start = rospy.Duration(1.0/25.0) # Not necessary?
            trajectory.points.append(jtp)

        # Send a traj
        self.setPostureControl()
        self.joint_traj_pub.publish(trajectory)
        return self.checkMovement(0.001, timeout, goal_type='posture')

    def checkMovement(self, time_step, timeout, track_err_limit = 100, last_time = -1, planner=False, loop=True, goal_type='position'):
        rt = rospy.Rate(1/time_step)
        stop = ''
        ea = None
        if planner: self.clear_buf()

        if last_time == -1:
            last_time = rospy.Time.now()

        while stop == '':

            if stop == '' and rospy.is_shutdown():
                stop = 'rospy shutdown'

            if stop == '' and self.stop_mpc:
                stop = 'stop_command_over_ROS'

            #if self.pause_mpc:
            #    rospy.sleep(0.1)
            #    continue

            # check timeout
            if stop == '' and timeout > 0.0:
                current_time = rospy.Time.now()
                delta_secs = current_time.secs - last_time.secs
                if delta_secs > timeout:
                    stop = 'stop by timeout'

            # check state space error for planner
            if planner == True:

                # fit Gaussian to last 50 locations of the end effector.
                self.update_jts_gaussian_buf(self.jts_gaussian_mn_buf_50,
                                             self.jts_gaussian_cov_buf_50, 50)
                self.update_mean_motion(self.jts_gaussian_mn_buf_50, step=200)

                # get current error
                ## if stop == '' and abs(self.track_err) > track_err_limit:
                ##     print self.track_err, " " , track_err_limit
                ##     stop = 'stop by over track pos limit'
                ## if self.stuck_mpc:
                if not(self.is_jts_moving(0.00051)):
                    current_time = rospy.Time.now()
                    delta_secs = current_time.secs - last_time.secs
                    if delta_secs < 6.0:
                        self.clear_buf_pub.publish(True)
                        self.clear_buf()
                        ## print "reset buf"
                    else:
                        stop = 'stop by stuck'

            if stop == '':
                stop = self.checkInGoal(goal_type)

            if loop == False:
                break

            rt.sleep()

        if planner == False and loop == True and self.controller != 'actionlib': self.setStop(goal_type)
        rospy.loginfo(stop)
        return stop, ea


    def checkInGoal(self, goal_type):

        pos_err_mag = 0.0
        quat_err_mag = 0.0
        posture_err_mag = 0.0

        if goal_type == 'position' or goal_type == 'orient':
            des_pos = np.matrix([self.current_goal.pose.position.x,
                                 self.current_goal.pose.position.y,
                                 self.current_goal.pose.position.z]).T

            cur_pos = self.getEndeffectorPose()[0]

            if cur_pos[0,0] == None:
                print "checkInPosition: No cur_pos!!!!!"
                stop = ''
                return stop

            pos_err_mag = np.linalg.norm(des_pos - cur_pos)

        ## Orientation check should be fixed...
        if goal_type == 'orient':
            des_quat = np.array([self.current_goal.pose.orientation.x,
                                 self.current_goal.pose.orientation.y,
                                 self.current_goal.pose.orientation.z,
                                 self.current_goal.pose.orientation.w])
            des_rot  = tf.transformations.quaternion_matrix(des_quat)

            cur_quat = self.getEndeffectorPose()[1]
            cur_rot  = tf.transformations.quaternion_matrix(cur_quat)

            ## print des_rot[:3,0], " --- ", cur_pos[:3,0]
            ## print qt.quat_angle(des_quat, cur_quat), np.linalg.norm(des_rot[:3,0]-cur_rot[:3,0])

            quat_err_mag  = np.linalg.norm(des_rot[:3,0]-cur_rot[:3,0])
            ## quat_err_mag = qt.quat_angle(des_quat, cur_quat)

        elif goal_type == 'posture': #JUST ADDED THIS SMALL THING!!!
            try:
                des_joint = np.array(self.current_joint_trajectory_msg.points[-1].positions)
            except:
                des_joint = np.array(self.current_goal.data)
            cur_joint = np.array(self.getJointAngles())

            joint_err_mag = np.linalg.norm(des_joint-cur_joint)

        ## print pos_err_mag, quat_err_mag

        # InPosition Amount
        if goal_type == 'position' or goal_type == 'orient':
            if pos_err_mag < self.ee_motion_threshold and quat_err_mag < self.ee_orient_motion_threshold:
                stop = 'Reached'
                rospy.loginfo("Reached a specified goal")
            else:
                stop = ''

        elif goal_type == 'posture':
            if joint_err_mag < self.jts_motion_threshold:
                stop = 'Reached'
                rospy.loginfo("Reached a specified goal")
            else:
                stop = ''
        else:
            stop = ''

        return stop

    def setStop(self, goal_type='position'):
        self.goal_traj_pub.publish(PoseArray())
        self.joint_traj_pub.publish(JointTrajectory())

        while not rospy.is_shutdown():
            [mPos, lQuat] = self.getEndeffectorPose()
            if mPos is not None:
                break
            rospy.sleep(0.2)

        if goal_type != 'posture':
            self.setStopPos(mPos, lQuat)
            rospy.loginfo("Stopping MPC")

    def setStopPos(self, mPos, lQuat=None):

        # get cur
        pos  = Point()
        quat = Quaternion()

        pos.x = mPos[0,0]
        pos.y = mPos[1,0]
        pos.z = mPos[2,0]

        if lQuat!=None:
            quat.x = lQuat[0]
            quat.y = lQuat[1]
            quat.z = lQuat[2]
            quat.w = lQuat[3]

        ps = PoseStamped()
        ps.header.frame_id  = self.torso_frame #correct?
        ps.pose.position    = pos
        ps.pose.orientation = quat

        self.goal_pos_pub.publish(ps)

        ## print "Stopping pos: ", mPos[0,0], mPos[1,0], mPos[2,0]
        rospy.loginfo("Stopping MPC with position or pose")


    def setAnnounceMovement(self, pos, quat, orient_flag, timeout, logging_name):
        rospy.loginfo("Start AnnouceMovement.")

        # Start to log
        self.log_and_monitor_run(True)
        self.close_pub.publish(Empty()) # for pr2 only

        # Provide base-logging information
        try:
            req = HapticMPCLogAndMonitorInfoRequest()
        except rospy.ServiceException, e:
            print "Service did not answer!!: %s"%str(e)
            return 'error',False

        req.logging_name = logging_name
        #req.torso_pose   = #?
        req.local_goal   = copy.copy(pos)
        self.log_and_monitor_info(req)

        # Start to move
        if orient_flag:
            stop, ea = self.setOrientGoal(pos, quat, timeout)
        else:
            stop, ea = self.setPositionGoal(pos, quat, timeout)

        # Stop to log
        self.log_and_monitor_run(False)

        return stop, ea

    def setAnnouncePlannerMovement(self, pos, quat, orient_flag, timeout, single_reach_timeout, track_pos_err_limit, track_jnt_err_limit, logging_name, plantype, video=False):
        rospy.loginfo("Start Traj AnnouceMovement.")

        # Start to log
        self.log_and_monitor_run(True)
        ## self.close_pub.publish(Empty()) # for pr2 only

        # Provide base-logging information
        try:
            req = HapticMPCLogAndMonitorInfoRequest()
        except rospy.ServiceException, e:
            print "Service did not answer!!: %s"%str(e)
            return 'error',False

        req.logging_name = logging_name
        req.local_goal   = copy.copy(pos)
        self.log_and_monitor_info(req)
        self.plan_count = 1

        # temporal
        ## at_waypoint_threshold = rospy.get_param('haptic_mpc/controller/offset')
        ## at_waypoint_index = int(at_waypoint_threshold / 0.02)
        ## print "waypoint threshold: ", at_waypoint_threshold, " " , at_waypoint_index

        # Start to move
        last_time = rospy.Time.now()

        if video==True:
            ## self.start_time_pub = rospy.Publisher('hrl_planner/start_time', std_msgs.msg.Float64, latch=True)
            ## self.start_time_pub.publish(last_time)
            print last_time
            rospy.set_param('hrl_planner/start_time', float(last_time.secs))

        while True and not rospy.is_shutdown():

            # Finish condition
            current_time = rospy.Time.now()
            delta_secs = current_time.secs - last_time.secs
            if delta_secs > timeout:
                stop = 'stop by over time: ' , timeout
                break

            if plantype.find('taskplan') > 0 or plantype.find('combiplan') > 0:
                self.enable_task_plan_pub.publish(True)

                print "--------------------------------"
                print "Start task plan!!"

                self.setPositionControl()
                stop, ea = self.setPlannerGoal(pos, quat, single_reach_timeout)

                # Get a trajectory
                rate      = rospy.Rate(100) # 25Hz, nominally.
                task_last_time = rospy.Time.now()
                stop      = ''

                self.clear_buf_pub.publish(True)
                while not rospy.is_shutdown() and stop == '':

                    # Check elapsed time
                    current_time = rospy.Time.now()
                    delta_secs = current_time.secs - task_last_time.secs
                    if delta_secs > single_reach_timeout:
                        print "single reach time out"
                        break

                    if self.current_pose_trajectory_msg != None:

                        # Send the trajectory to MPC (waypoint_generator)
                        ## self.pose_traj_pub.publish(self.current_trajectory_msg)
                        goal_pose_traj_msg = self.current_pose_trajectory_msg

                        ps = PoseStamped()
                        ps.header.frame_id  = '/torso_lift_link'

                        # If the start of trajectory is too different with current position, continue
                        cur_pos = copy.copy(self.getEndeffectorPose()[0])
                        if cur_pos[0,0] != None:

                            start_pose = self.current_pose_trajectory_msg.poses[0]
                            start_pos = np.matrix([start_pose.position.x,
                                                   start_pose.position.y,
                                                   start_pose.position.z]).T

                            err     = start_pos - cur_pos
                            err_mag = np.linalg.norm(err)
                            if err_mag > 0.08:
                                ## print "provided trajectory is too far from current"
                                continue

                        # get a waypoint
                        ## if len(self.current_pose_trajectory_msg.poses) > at_waypoint_index:
                        ##     ## ps.pose = self.current_pose_trajectory_msg.poses[3]
                        ##     goal_pose_traj_msg.poses = [self.current_pose_trajectory_msg.poses[at_waypoint_index]]
                        ##     self.pose_traj_pub.publish(goal_pose_traj_msg)
                        ## else:
                        ##     index = len(self.current_pose_trajectory_msg.poses)
                        ##     ## ps.pose = self.current_pose_trajectory_msg.poses[index-1]
                        ##     goal_pose_traj_msg.poses = [self.current_pose_trajectory_msg.poses[index-1]]
                        ##     self.pose_traj_pub.publish(goal_pose_traj_msg)

                        # get trajectory
                        goal_pose_traj_msg.poses = self.current_pose_trajectory_msg.poses
                        self.pose_traj_pub.publish(goal_pose_traj_msg)

                        ## self.goal_pos_pub.publish(ps)
                        self.current_pose_trajectory_msg = None
                    else:
                        continue

                    stop, ea = self.checkMovement(0.001, single_reach_timeout, track_pos_err_limit, \
                                                  current_time, True, False)

                    if stop == 'stop by stuck mpc from log and monitor':
                        stop = ''

                    rate.sleep()

                # Finish planner
                self.enable_task_plan_pub.publish(False)

                # Check movement status.
                ## stop, ea = self.checkMovement(0.001, single_reach_timeout, track_pos_err_limit, True)
                print stop
                if stop == 'Reached': break

            if plantype.find('jointplan') > 0 or plantype.find('combiplan') > 0:

                self.current_joint_trajectory_msg = None
                self.enable_joint_plan_pub.publish(True)

                print "--------------------------------"
                print "Start joint plan!!"

                stop, ea = self.setPlannerGoal(pos, quat, single_reach_timeout*3)
                self.setPostureControl()

                # Check elapsed time
                task_last_time = rospy.Time.now()

                # Get a trajectory
                rate = rospy.Rate(100) # 25Hz, nominally.
                while not rospy.is_shutdown():
                    if self.current_joint_trajectory_msg != None:
                        break
                    else:
                        current_time = rospy.Time.now()
                        delta_secs = current_time.secs - task_last_time.secs
                        if delta_secs > single_reach_timeout*20:
                            print "Waiting trajectory time out"
                            break

                    rate.sleep()

                if delta_secs > single_reach_timeout*20:
                    continue

                # Send the trajectory to MPC (waypoint_generator)
                self.clear_buf_pub.publish(True)
                self.clear_buf_pub.publish(True)
                self.joint_traj_pub.publish(self.current_joint_trajectory_msg)
                self.plan_count = self.plan_count + 1

                # Finish planner
                self.current_joint_trajectory_msg = None
                self.enable_joint_plan_pub.publish(False)

                # Check movement status.
                joint_last_time = rospy.Time.now()
                stop, ea = self.checkMovement(0.001, timeout, track_jnt_err_limit, joint_last_time, True)
                print stop

                if stop == 'Reached': break

        # Stop to log
        self.log_and_monitor_run(False)

        return stop, ea


    def setNonAnnounceMovement(self,pos, quat, orient_flag, timeout):
        rospy.loginfo("Start NonAnnouceMovement.")

        self.close_pub.publish(Empty())

        if orient_flag:
            stop, ea = self.setOrientGoal(pos, quat, timeout)
        else:
            stop, ea = self.setPositionGoal(pos, quat, timeout)

        return stop, ea

    def setNonAnnounceTrajMovement(self, pos, traj, pose_flag, timeout):
        rospy.loginfo("Start NonAnnouceTrajMovement.")

        self.close_pub.publish(Empty())

        if pose_flag:
            print "Under construction"
        else:
            stop, ea = self.setJointTraj(pos, traj, timeout)

        return stop, ea




    def stopMPCCallback(self, msg):
        self.stop_mpc = msg.data

    def stuckMPCCallback(self, msg):
        self.stuck_mpc = msg.data

    ## Store a trajectory of poses in the deque. Converts it to the 'torso_frame' if required.
    # @param msg A geometry_msgs.msg.PoseArray object
    def poseTrajectoryCallback(self, msg):
        with self.traj_lock:
            # if we have an empty array, clear the deque and do nothing else.
            if len(msg.poses) == 0:
                rospy.logwarn("Received empty pose array. Clearing trajectory buffer")
                return
            self.current_pose_trajectory_msg = copy.copy(msg)

    ## Store a joint angle trajectory in the deque. Performs forward kinematics to convert it to end effector poses in the torso frame.
    # @param msg A trajectory_msgs.msg.JointTrajectory object.
    def jointTrajectoryCallback(self, msg):
        with self.traj_lock:
            # if we have an empty array, clear the deque and do nothing else.
            if len(msg.points) == 0:
                rospy.logwarn("Received empty joint array. Clearing trajectory buffer")
                return
            self.current_joint_trajectory_msg = msg

    def trackingErrorCallback(self, msg):
        with self.tracking_lock:
            self.track_err = msg.data

    def getEndeffectorPose(self):

        if self.controller == "actionlib":
            if (self.d_robot == 'pr2' or self.d_robot.find("sim") >= 0) is not True:
                rospy.logerr("actionlib is not available for darci or darci_sim")
                sys.exit()

            self.get_fk()
            curr_pos = self.curr_pose.pose_stamped[-1].pose.position
            curr_orient = self.curr_pose.pose_stamped[-1].pose.orientation

            self.end_effector_pos = np.matrix([[curr_pos.x], [curr_pos.y], [curr_pos.z]])
            self.end_effector_orient_quat = [curr_orient.x, curr_orient.y, curr_orient.z, curr_orient.w]

        else:

            if not (self.d_robot.find("sim") >= 0):
                try:
                    self.tf_lstnr.waitForTransform(self.torso_frame, self.ee_frame, rospy.Time(0), \
                                                       rospy.Duration(5.0))
                except:
                    self.tf_lstnr.waitForTransform(self.torso_frame, self.ee_frame, rospy.Time(0), \
                                                       rospy.Duration(5.0))

                [self.end_effector_pos, self.end_effector_orient_quat] = \
                    self.tf_lstnr.lookupTransform(self.torso_frame, self.ee_frame, rospy.Time(0))
            else:
                mPose  = self.arm_kdl.forward(self.joint_angles)
                self.end_effector_pos = [float(mPose[0,3]),float(mPose[1,3]),float(mPose[2,3])]
                self.end_effector_orient_quat = transformations.quaternion_from_matrix(mPose)


        if not (self.d_robot.find("sim3") >= 0):
            self.end_effector_pos = np.matrix(self.end_effector_pos).T

        return [self.end_effector_pos, self.end_effector_orient_quat]


    def getJointAngles(self):
        joint_angles = copy.copy(self.joint_angles)
        return joint_angles


    def get_fk(self):
        #print "get_fk of %s" %str(msg)
        if (self.joint_angles):
            fk_request = GetPositionFKRequest()
            fk_request.header.frame_id = '/torso_lift_link'
            fk_request.fk_link_names = self.fk_info.kinematic_solver_info.link_names
            fk_request.robot_state.joint_state.position = self.joint_angles
            fk_request.robot_state.joint_state.name = self.fk_info.kinematic_solver_info.joint_names
        else:
            rospy.loginfo("No Joint States Available Yet")

        try:
            self.curr_pose = self.fk_pose_proxy(fk_request)
        except rospy.ServiceException, e:
            rospy.loginfo("FK service did not process request: %s" %str(e))


    def get_ik(self, msg, timeout):
        #print "get_ik of %s" %str(msg)
        if (self.joint_angles):
            ik_request = GetPositionIKRequest()
            ik_request.timeout = rospy.Duration(5)
            ik_request.ik_request.pose_stamped = msg
            ik_request.ik_request.ik_link_name = self.ee_frame
            ik_request.ik_request.ik_seed_state.joint_state.name =  self.joint_names_list
            ik_request.ik_request.ik_seed_state.joint_state.position =  self.joint_angles
        else:
            rospy.loginfo("No Joint States Available Yet")
            #print "IK Request: %s" %str(ik_request)
        try:
            ik_goal = self.ik_pose_proxy(ik_request)
            ## print "IK Goal: %s" %str(ik_goal)
            if ik_goal.error_code.val == 1:
                self.try_per = 1

                trajMsg = pr2_controllers_msgs.msg.JointTrajectoryGoal()
                trajPoint = trajectory_msgs.msg.JointTrajectoryPoint()
                trajPoint.positions       = ik_goal.solution.joint_state.position
                trajPoint.velocities      = [0 for i in range(0, len(self.joint_names_list))]
                trajPoint.accelerations   = [0 for i in range(0, len(self.joint_names_list))]
                trajPoint.time_from_start = rospy.Duration(timeout)

                trajMsg.trajectory.joint_names = self.joint_names_list
                trajMsg.trajectory.points.extend([trajPoint])
                self.armClient.send_goal(trajMsg)
            else:
                self.try_per -= 0.02
                if (self.try_per >= 0.01):
                    self.get_fk()
                    goal_pos = msg.pose.position
                    curr_pos = self.curr_pose.pose_stamped[-1].pose.position
                    goal_pos.x = curr_pos.x + self.try_per*(goal_pos.x-curr_pos.x)
                    goal_pos.y = curr_pos.y + self.try_per*(goal_pos.y-curr_pos.y)
                    goal_pos.z = curr_pos.z + self.try_per*(goal_pos.z-curr_pos.z)
                    msg.pose.position = goal_pos
                    self.get_ik(msg, timeout)
                else:
                    rospy.loginfo("IK Failed: Error Code %s" %str(ik_goal.error_code))
        except rospy.ServiceException, e:
            rospy.loginfo("IK service did not process request: %s" %str(e))




    def update_jts_gaussian_buf(self, mn_buf, cov_buf, hist_size):
        if len(self.q_buf) < hist_size:
            return
        jts_hist = self.q_buf.get_last(hist_size) # Take last 50 joint angles
        jts_hist = np.matrix(jts_hist).T
        mn_buf.append(np.mean(jts_hist, 1).A1) # Store a mean_pos from 50 joint_angles
        ## cov_buf.append(np.cov(jts_hist))

    def update_mean_motion(self, mn_buf, step):
        if len(mn_buf) < step:
            return
        #d = np.linalg.norm((mn_buf[-1] - mn_buf[-step])[0:2]) # Get difference between mean_poses
        d = np.linalg.norm((mn_buf[-1] - mn_buf[-step])[0:self.n_jts]) # Get difference between mean_poses

        self.mean_motion_buf.append(d) # Store the difference

    def is_jts_moving(self, angle_thresh):

        if len(self.mean_motion_buf) > 0 and \
           self.mean_motion_buf[-1] < angle_thresh:
            #n = min(len(self.mean_motion_buf), 5)
            rospy.loginfo('Mean is not moving anymore: %s'%(str(self.mean_motion_buf[-1])))
            return False

        ## if len(self.mean_motion_buf) > 0:
        ##     print self.mean_motion_buf[-1], angle_thresh
        return True



    ## def pose_constraint_to_position_orientation_constraints(self, pose_constraint):
    ##     position_constraint = PositionConstraint()
    ##     orientation_constraint = OrientationConstraint()
    ##     position_constraint.header = pose_constraint.header
    ##     position_constraint.link_name = pose_constraint.link_name
    ##     position_constraint.position = pose_constraint.pose.position

    ##     position_constraint.constraint_region_shape.type = geometric_shapes_msgs.msg.Shape.BOX
    ##     position_constraint.constraint_region_shape.dimensions.append(2*pose_constraint.absolute_position_tolerance.x)
    ##     position_constraint.constraint_region_shape.dimensions.append(2*pose_constraint.absolute_position_tolerance.y)
    ##     position_constraint.constraint_region_shape.dimensions.append(2*pose_constraint.absolute_position_tolerance.z)

    ##     position_constraint.constraint_region_orientation.x = 0.0
    ##     position_constraint.constraint_region_orientation.y = 0.0
    ##     position_constraint.constraint_region_orientation.z = 0.0
    ##     position_constraint.constraint_region_orientation.w = 1.0

    ##     position_constraint.weight = 1.0

    ##     orientation_constraint.header = pose_constraint.header
    ##     orientation_constraint.link_name = pose_constraint.link_name
    ##     orientation_constraint.orientation = pose_constraint.pose.orientation
    ##     orientation_constraint.type = pose_constraint.orientation_constraint_type

    ##     orientation_constraint.absolute_roll_tolerance = pose_constraint.absolute_roll_tolerance
    ##     orientation_constraint.absolute_pitch_tolerance = pose_constraint.absolute_pitch_tolerance
    ##     orientation_constraint.absolute_yaw_tolerance = pose_constraint.absolute_yaw_tolerance
    ##     orientation_constraint.weight = 1.0

    ##     return position_constraint, orientation_constraint






            ## self.desired_pose.pose.position = pos
            ## self.desired_pose.pose.orientation = orient_quat
            ## self.add_goal_constraint_to_move_arm_goal(self.desired_pose, self.goalA)
            ## self.move_arm.send_goal(self.goalA)

            ## rospy.logout('Run move arm')
            ## finished_within_time = self.move_arm.wait_for_result(rospy.Duration(timeout))

            ## if not finished_within_time:
            ##     self.move_arm.cancel_goal()
            ##     rospy.logout('Timed out achieving goal A')
            ## else:
            ##     state = self.move_arm.get_state()
            ##     if state == GoalStatus.SUCCEEDED:
            ##         rospy.logout('Action finished with SUCCESS')
            ##     else:
            ##         rospy.logout('Action failed')



    ## def initVars(self):
    ##     if self.controller == 'actionlib':

    ##         self.goalA = MoveArmGoal()
    ##         self.goalA.motion_plan_request.group_name = self.arm_name+'_arm'
    ##         self.goalA.motion_plan_request.num_planning_attempts = 1
    ##         self.goalA.motion_plan_request.planner_id = ''
    ##         self.goalA.planner_service_name = 'ompl_planning/plan_kinematic_path'
    ##         self.goalA.motion_plan_request.allowed_planning_time = rospy.Duration(5.)

    ##         self.desired_pose = SimplePoseConstraint()
    ##         self.desired_pose.header.frame_id = 'torso_lift_link'
    ##         self.desired_pose.link_name = self.ee_frame
    ##         self.desired_pose.absolute_position_tolerance.x = self.ee_motion_threshold
    ##         self.desired_pose.absolute_position_tolerance.y = self.ee_motion_threshold
    ##         self.desired_pose.absolute_position_tolerance.z = self.ee_motion_threshold
    ##         self.desired_pose.absolute_position_tolerance.x = self.ee_orient_motion_threshold
    ##         self.desired_pose.absolute_position_tolerance.y = self.ee_orient_motion_threshold
    ##         self.desired_pose.absolute_position_tolerance.z = self.ee_orient_motion_threshold



            ## self.move_arm = actionlib.SimpleActionClient('/move_left_arm',MoveArmAction)
            ## self.move_arm.wait_for_server()
            ## rospy.logout('Connected to move arm server')

            ## self.move_arm = actionlib.SimpleActionClient('/move_'+self.arm_name+'_arm',MoveArmAction)
            ## self.move_arm.wait_for_server()
            ## rospy.logout('Connected to move arm server')
