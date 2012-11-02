#! /usr/bin/python
        
import numpy as np
import sys
import cPickle as pkl
from threading import Lock

import roslib
roslib.load_manifest('hrl_face_adls')

import rospy
from std_msgs.msg import Int8, String, Bool
from std_srvs.srv import Empty
from geometry_msgs.msg import WrenchStamped, PoseStamped, PoseArray
import rosparam
import roslaunch.substitution_args
from tf import TransformListener, ExtrapolationException, LookupException, ConnectivityException

from hrl_geom.pose_converter import PoseConv
import hrl_geom.transformations as trans
#from hrl_pr2_arms.pr2_arm_jt_task import create_ep_arm, PR2ArmJTransposeTask
from hrl_pr2_arms.pr2_controller_switcher import ControllerSwitcher
from hrl_ellipsoidal_control.ellipsoid_space import EllipsoidSpace
from hrl_ellipsoidal_control.ellipsoidal_parameters import *
from hrl_face_adls.face_adls_parameters import *
from hrl_face_adls.msg import StringArray
from hrl_face_adls.srv import EnableFaceController, EnableFaceControllerResponse
from hrl_face_adls.srv import RequestRegistration

##
# Returns a function which will call the callback immediately in
# a different thread.
def async_call(cb):
    def cb_outer(*args, **kwargs):
        def cb_inner(te):
            cb(*args, **kwargs)
        rospy.Timer(rospy.Duration(0.00001), cb_inner, oneshot=True)
    return cb_outer

class ForceCollisionMonitor(object):
    def __init__(self):
        self.last_activity_time = rospy.get_time()
        self.last_reading = rospy.get_time()
        self.last_dangerous_cb_time = 0.0
        self.last_timeout_cb_time = 0.0
        self.last_contact_cb_time = 0.0
        self.in_action = False
        self.lock = Lock()
        self.dangerous_force_thresh = rospy.get_param("~dangerous_force_thresh", 10.0)
        self.activity_force_thresh = rospy.get_param("~activity_force_thresh", 3.0)
        self.contact_force_thresh = rospy.get_param("~contact_force_thresh", 3.0)
        self.timeout_time = rospy.get_param("~timeout_time", 30.0)
        rospy.Subscriber('/netft_gravity_zeroing/wrench_zeroed', WrenchStamped, 
                         async_call(self.force_cb), queue_size=1)
        def check_readings(te):
            time_diff = rospy.get_time() - self.last_reading
            if time_diff > 3.:
                rospy.logerr("Force monitor hasn't recieved force readings for %.1f seconds!"
                              % time_diff)
        rospy.Timer(rospy.Duration(3), check_readings)

    def force_cb(self, msg):
        self.last_reading = rospy.get_time()
        if self.lock.acquire(False):
            force_mag = np.linalg.norm([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
            if (force_mag > self.dangerous_force_thresh and
                    rospy.get_time() - self.last_dangerous_cb_time > DANGEROUS_CB_COOLDOWN):
                self.dangerous_cb(self.dangerous_force_thresh)
                self.last_dangerous_cb_time = rospy.get_time()
            elif (force_mag > self.contact_force_thresh and 
                    rospy.get_time() - self.last_contact_cb_time > CONTACT_CB_COOLDOWN):
                self.contact_cb(self.contact_force_thresh)
                self.last_contact_cb_time = rospy.get_time()
            elif force_mag > self.activity_force_thresh:
                self.update_activity()
            elif (self.is_inactive() and 
                    rospy.get_time() - self.last_timeout_cb_time > TIMEOUT_CB_COOLDOWN):
                self.timeout_cb(self.timeout_time)
                self.last_timeout_cb_time = rospy.get_time()
            self.lock.release()

    def is_inactive(self):
        return not self.in_action and rospy.get_time() - self.last_activity_time > self.timeout_time

    def register_contact_cb(self, cb=lambda x:None):
        self.contact_cb = async_call(cb)

    def register_dangerous_cb(self, cb=lambda x:None):
        self.dangerous_cb = async_call(cb)

    def register_timeout_cb(self, cb=lambda x:None):
        self.timeout_cb = async_call(cb)

    def update_activity(self):
        self.last_activity_time = rospy.get_time()

    def start_activity(self):
        self.in_action = True

    def stop_activity(self):
        self.last_activity_time = rospy.get_time()
        self.in_action = False

class FaceADLsManager(object):
    def __init__(self):

        self.ellipsoid = EllipsoidSpace()
        self.ctrl_switcher = ControllerSwitcher()
        self.ee_frame = '/l_gripper_shaver45_frame'
        self.tfl = TransformListener()
        self.pose_traj_pub = rospy.Publisher('/face_adls/goal_poses', PoseArray)

        self.global_input_sub = rospy.Subscriber("/face_adls/global_move", String, 
                                                 async_call(self.global_input_cb))
        self.local_input_sub = rospy.Subscriber("/face_adls/local_move", String, 
                                                async_call(self.local_input_cb))
        self.state_pub = rospy.Publisher('/face_adls/controller_state', Int8, latch=True)
        self.feedback_pub = rospy.Publisher('/face_adls/feedback', String)
        self.global_move_poses_pub = rospy.Publisher('/face_adls/global_move_poses', StringArray, 
                                                     latch=True)
        self.controller_enabled_pub = rospy.Publisher('/face_adls/controller_enabled', Bool, latch=True)
        self.enable_controller_srv = rospy.Service("/face_adls/enable_controller", 
                                                   EnableFaceController, self.enable_controller_cb)
        self.request_registration = rospy.ServiceProxy("/request_registration", RequestRegistration)

        def stop_move_cb(msg):
            self.stop_moving()
        self.stop_move_sub = rospy.Subscriber("/face_adls/stop_move", Bool, stop_move_cb, queue_size=1)

    def stop_moving(self):
        """Send empty PoseArray to skin controller to stop movement"""
        self.pose_traj_pub.publish(PoseArray()) #Send empty msg to skin controller

    def current_ee_pose(self, link, frame='/torso_lift_link'):
        """Get current end effector pose from tf"""
        try:
            now = rospy.Time.now()
            self.tfl.waitForTransform(frame, link, now, rospy.Duration(8.0))
            pos, rot = self.tfl.lookupTransform(link, frame, now)
        except (LookupException, ConnectivityException, ExtrapolationException, Exception) as e:
            rospy.logwarn("[face_adls_manager] TF Failure getting current end-effector pose: %s" %e)
            return None
        return pos, rot
        #cur_pose = PoseStamped()
        #cur_pose.header.stamp = rospy.Time.now()
        #cur_pose.header.frame_id = frame
        #cur_pose.pose.position.x = pos[0]
        #cur_pose.pose.position.y = pos[1]
        #cur_pose.pose.position.z = pos[2]
        #cur_pose.pose.orientation.x = rot[0]
        #cur_pose.pose.orientation.y = rot[1]
        #cur_pose.pose.orientation.z = rot[2]
        #cur_pose.pose.orientation.w = rot[3]
        #rospy.loginfo("[face_adls_manager] Got current pose: \r\n%s" %cur_pose)
        #return cur_pose

    def publish_feedback(self, message=None, transition_id=None):
        if message is not None:
            rospy.loginfo("[face_adls_manager] %s" % message)
            self.feedback_pub.publish(message)
        if transition_id is not None:
            self.state_pub.publish(Int8(transition_id))

    def enable_controller_cb(self, req):
        if req.enable:
            face_adls_modes = rospy.get_param('/face_adls_modes', None) 
            params = face_adls_modes[req.mode]
            self.shaving_side = rospy.get_param('/shaving_side', 'r') # TODO Make more general
            self.trim_retreat = req.mode == "shaving"
            self.flip_gripper = self.shaving_side == 'r' and req.mode == "shaving"
            bounds = params['%s_bounds' % self.shaving_side]
            #self.ellipsoid.set_bounds(bounds['lat'], bounds['lon'], bounds['height'])
            self.ellipsoid.set_bounds((-np.pi,np.pi), (-np.pi,np.pi), (0,100))

            reg_resp = self.request_registration(req.mode, self.shaving_side)
            if not reg_resp.success:
                self.publish_feedback(Messages.NO_PARAMS_LOADED)
                return EnableFaceControllerResponse(False)

            self.ellipsoid.load_ell_params(reg_resp.e_params.e_frame,
                                           reg_resp.e_params.E,
                                           reg_resp.e_params.is_oblate,
                                           reg_resp.e_params.height)
            global_poses_dir = rospy.get_param("~global_poses_dir", "")
            global_poses_file = params["%s_ell_poses_file" % self.shaving_side]
            global_poses_resolved = roslaunch.substitution_args.resolve_args(
                                            global_poses_dir + "/" + global_poses_file)
            self.global_poses = rosparam.load_file(global_poses_resolved)[0][0]
            self.global_move_poses_pub.publish(sorted(self.global_poses.keys()))

            # check if arm is near head
            cart_pos, cart_quat = self.current_ee_pose(self.ee_frame)
            ell_pos, ell_quat = self.ellipsoid.pose_to_ellipsoidal(cart_pos, cart_quat)
            equals = ell_pos == self.ellipsoid.enforce_bounds(ell_pos)
            print ell_pos, equals
            if self.ellipsoid._lon_bounds[0] >= 0 and ell_pos[1] >= 0:
                arm_in_bounds =  np.all(equals)
            else:
                ell_lon_1 = np.mod(ell_pos[1], 2 * np.pi)
                min_lon = np.mod(self.ellipsoid._lon_bounds[0], 2 * np.pi)
                arm_in_bounds = (equals[0] and
                        equals[2] and 
                        (ell_lon_1 >= min_lon or ell_lon_1 <= self.ellipsoid._lon_bounds[1]))

            if not arm_in_bounds:
                self.publish_feedback(Messages.ARM_AWAY_FROM_HEAD)
                return False

            self.publish_feedback(Messages.ENABLE_CONTROLLER)
            self.setup_force_monitor()

            if self.flip_gripper:
                self.gripper_rot = trans.quaternion_from_euler(np.pi, 0, 0)
            else:
                self.gripper_rot = trans.quaternion_from_euler(0, 0, 0)
            self.controller_enabled_pub.publish(Bool(True))
            success = True
        else:
            self.stop_moving()
            self.controller_enabled_pub.publish(Bool(False))
            rospy.loginfo(Messages.DISABLE_CONTROLLER)
            success =  True
        return EnableFaceControllerResponse(success)

    def setup_force_monitor(self):
        self.force_monitor = ForceCollisionMonitor()
        # registering force monitor callbacks
        def dangerous_cb(force):
            self.stop_moving()
            ell_pos, _ = self.ellipsoid.pose_to_ellipsoidal(self.current_ee_pose())
            if ell_pos[2] < SAFETY_RETREAT_HEIGHT * 0.9:
                self.publish_feedback(Messages.DANGEROUS_FORCE %force)
                self.retreat_move(SAFETY_RETREAT_HEIGHT, SAFETY_RETREAT_VELOCITY, forced=True)
        self.force_monitor.register_dangerous_cb(dangerous_cb)
        def timeout_cb(time):
            start_time = rospy.get_time()
            ell_pos, _ = self.ellipsoid.pose_to_ellipsoidal(self.current_ee_pose())
            rospy.loginfo("ELL POS TIME: %.3f " % (rospy.get_time() - start_time) + str(ell_pos) 
                          + " times: %.3f %.3f" % (self.force_monitor.last_activity_time, rospy.get_time()))
            if ell_pos[2] < RETREAT_HEIGHT * 0.9 and self.force_monitor.is_inactive():
                self.publish_feedback(Messages.TIMEOUT_RETREAT % time)
                self.retreat_move(RETREAT_HEIGHT, LOCAL_VELOCITY)
        self.force_monitor.register_timeout_cb(timeout_cb)
        def contact_cb(force):
            self.force_monitor.update_activity()
            if not self.is_forced_retreat:
                self.stop_moving()
                self.publish_feedback(Messages.CONTACT_FORCE % force)
                rospy.loginfo("Arm stopped due to making contact.")
        self.force_monitor.register_contact_cb(contact_cb)
        self.force_monitor.update_activity()
        self.is_forced_retreat = False

    def retreat_move(self, height, velocity, forced=False):
        self.force_monitor.start_activity()
        if forced:
            self.is_forced_retreat = True
        ell_pos, ell_quat = self.ellipsoid.pose_to_ellipsoidal(self.current_ee_pose())
        print "Retreat EP:", ell_pos
        latitude = ell_pos[0]
        if self.trim_retreat:
            latitude = min(latitude, TRIM_RETREAT_LATITUDE)
        rospy.loginfo("[face_adls_manager] Retreating from current location.")

        retreat_pos = (latitude, ell_pos[1], height)
        retreat_quat = (0,0,0,1)
        self.ellipsoid.create_ellipsoidal_path(retreat_pos, retreat_quat,
                                               ell_pos, ell_quat,
                                               self.gripper_rot,
                                               velocity=velocity)
        self.is_forced_retreat = False #TODO: Find way to confirm duration of retreat
        rospy.loginfo("[face_adls_manager] Finished retreat?")

    def global_input_cb(self, msg):
        if self.is_forced_retreat:
            return
        rospy.loginfo("[face_adls_manager] Performing global move.")
        self.force_monitor.start_activity()
        goal_pose = self.global_poses[msg.data]
        self.publish_feedback(Messages.GLOBAL_START %msg.data)
        curr_cart_pose = self.current_ee_pose(self.ee_frame, '/torso_lift_link')
        curr_ell_pos, curr_ell_quat = self.ellipsoid.pose_to_ellipsoidal(curr_cart_pose)
        retreat_ell_pos = [curr_ell_pos[0], curr_ell_pos[1], RETREAT_HEIGHT]
        retreat_ell_rot = np.mat(np.eye(3))
        approach_ell_pos = [goal_pose[0][0], goal_pose[0][0], RETREAT_HEIGHT]
        approach_ell_quat = goal_pose[1]
        final_ell_pos = [goal_pose[0][0], goal_pose[0][1], goal_pose[0][2] - HEIGHT_CLOSER_ADJUST]
        final_ell_quat = goal_pose[1]
        
        retreat_ell_traj = self.ellipsoid.create_ellipsoidal_path(retreat_ell_pos, retreat_ell_rot,
                                                              curr_ell_pos, curr_ell_rot)
        transition_ell_traj = self.ellipsoid.create_ellipsoidal_path(approach_ell_pos, trans.quaternion_matrix(approach_ell_quat),
                                                                 retreat_ell_pos, retreat_ell_rot)
        approach_ell_traj = self.ellipsoid.create_ellipsoidal_path(final_ell_pos, trans.quaternion_matrix(final_ell_quat),
                                                               approach_ell_pos, trans.quaternion_matrix(approach_ell_quat))
        
        full_ell_traj = retreat_traj + transition_traj + approach_traj
        full_cart_traj = [PoseConv.to_pose_stamped_msg(pose) for pose in full_ell_traj]
        self.skin_goals_pub.publish(full_cart_traj)

            
    def global_input_cb_old(self, msg):
        if self.is_forced_retreat:
            return
        if self.stop_move():
            rospy.loginfo("[face_adls_manager] Preempting other movement for global move.")
            #self.publish_feedback(Messages.GLOBAL_PREEMPT_OTHER)
        self.force_monitor.start_activity()
        goal_pose = self.global_poses[msg.data]
        goal_pose_name = msg.data
        self.publish_feedback(Messages.GLOBAL_START % goal_pose_name)
        try:
            if not self.ell_ctrl.execute_ell_move(((0, 0, RETREAT_HEIGHT), np.mat(np.eye(3))), 
                                                  ((0, 0, 1), 1), 
                                                  self.gripper_rot,
                                                  APPROACH_VELOCITY,
                                                  blocking=True):
                raise Exception
            if not self.ell_ctrl.execute_ell_move(((goal_pose[0][0], goal_pose[0][1], RETREAT_HEIGHT), (0, 0, 0, 1)), 
                                                  ((1, 1, 1), 0), 
                                                  self.gripper_rot,
                                                  GLOBAL_VELOCITY,
                                                  blocking=True):
                raise Exception
            final_goal = [goal_pose[0][0], goal_pose[0][1], goal_pose[0][2] - HEIGHT_CLOSER_ADJUST]
            if not self.ell_ctrl.execute_ell_move((final_goal, (0, 0, 0, 1)),
                                                  ((1, 1, 1), 0), 
                                                  self.gripper_rot,
                                                  GLOBAL_VELOCITY,
                                                  blocking=True):
                raise Exception
        except:
            self.publish_feedback(Messages.GLOBAL_PREEMPT % goal_pose_name)
            self.force_monitor.stop_activity()
            return
        self.publish_feedback(Messages.GLOBAL_SUCCESS % goal_pose_name)
        self.force_monitor.stop_activity()

    def local_input_cb(self, msg): #TODO: Change this
        if self.is_forced_retreat:
            return
        if self.stop_move():
            rospy.loginfo("[face_adls_manager] Preempting other movement for local move.")
            #self.publish_feedback(Messages.LOCAL_PREEMPT_OTHER)
        self.force_monitor.start_activity()
        button_press = msg.data 
        if button_press in ell_trans_params:
            self.publish_feedback(Messages.LOCAL_START % button_names_dict[button_press])
            change_trans_ep = ell_trans_params[button_press]
            success = self.ell_ctrl.execute_ell_move((change_trans_ep, (0, 0, 0, 1)),
                                                     ((0, 0, 0), 0), 
                                                    self.gripper_rot,
                                                    ELL_LOCAL_VEL,
                                                    blocking=True)
        elif button_press in ell_rot_params:
            self.publish_feedback(Messages.LOCAL_START % button_names_dict[button_press])
            change_rot_ep = ell_rot_params[button_press]
            rot_quat = trans.quaternion_from_euler(*change_rot_ep)
            success = self.ell_ctrl.execute_ell_move(((0, 0, 0), rot_quat),
                                                     ((0, 0, 0), 0), 
                                                     self.gripper_rot,
                                                     ELL_ROT_VEL,
                                                     blocking=True)
        elif button_press == "reset_rotation":
            self.publish_feedback(Messages.ROT_RESET_START)
            success = self.ell_ctrl.execute_ell_move(((0, 0, 0), np.mat(np.eye(3))),
                                                     ((0, 0, 0), 1), 
                                                     self.gripper_rot,
                                                     ELL_ROT_VEL,
                                                     blocking=True)
        else:
            rospy.logerr("[face_adls_manager] Unknown ellipsoidal local command")

        if success:
            self.publish_feedback(Messages.LOCAL_SUCCESS % button_names_dict[button_press])
        else:
            self.publish_feedback(Messages.LOCAL_PREEMPT % button_names_dict[button_press])
        self.force_monitor.stop_activity()

def main():
    rospy.init_node("face_adls_manager")

    #from pr2_traj_playback.arm_pose_move_controller import ArmPoseMoveBehavior, TrajectoryLoader
    #from pr2_traj_playback.arm_pose_move_controller import CTRL_NAME_LOW, PARAMS_FILE_LOW
    #r_apm = ArmPoseMoveBehavior('r', ctrl_name=CTRL_NAME_LOW,
    #                            param_file=PARAMS_FILE_LOW)
    #l_apm = ArmPoseMoveBehavior('l', ctrl_name=CTRL_NAME_LOW,
    #                            param_file=PARAMS_FILE_LOW)
    #traj_loader = TrajectoryLoader(r_apm, l_apm)
    #if False:
    #    traj_loader.move_to_setup_from_file("$(find hrl_face_adls)/data/l_arm_shaving_setup_r.pkl",
    #                                        velocity=0.1, reverse=False, blocking=True)
    #    traj_loader.exec_traj_from_file("$(find hrl_face_adls)/data/l_arm_shaving_setup_r.pkl",
    #                                    rate_mult=0.8, reverse=False, blocking=True)
    #if False:
    #    traj_loader.move_to_setup_from_file("$(find hrl_face_adls)/data/l_arm_shaving_setup_r.pkl",
    #                                        velocity=0.3, reverse=True, blocking=True)

    fam = FaceADLsManager()
    #fam.enable_controller()
    rospy.spin()

if __name__ == "__main__":
    main()
