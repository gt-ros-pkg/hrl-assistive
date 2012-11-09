#! /usr/bin/python
        
import numpy as np
import sys
import cPickle as pkl
from threading import Lock

import roslib
roslib.load_manifest('hrl_face_adls')

import rospy
from std_msgs.msg import Int8, String, Bool, Header
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
        #return not self.in_action and rospy.get_time() - self.last_activity_time > self.timeout_time
        return rospy.get_time() - self.last_activity_time > self.timeout_time

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
        self.controller_switcher = ControllerSwitcher()
        self.ee_frame = '/l_gripper_shaver45_frame'
        self.tfl = TransformListener()
        self.is_forced_retreat = False
        self.pose_traj_pub = rospy.Publisher('/haptic_mpc/goal_pose_array', PoseArray)

        self.global_input_sub = rospy.Subscriber("/face_adls/global_move", String, 
                                                 async_call(self.global_input_cb))
        self.local_input_sub = rospy.Subscriber("/face_adls/local_move", String, 
                                                async_call(self.local_input_cb))
        self.clicked_input_sub = rospy.Subscriber("/face_adls/clicked_move", PoseStamped, 
                                                 async_call(self.global_input_cb))
        self.feedback_pub = rospy.Publisher('/face_adls/feedback', String)
        self.global_move_poses_pub = rospy.Publisher('/face_adls/global_move_poses',
                                                     StringArray, latch=True)
        self.controller_enabled_pub = rospy.Publisher('/face_adls/controller_enabled',
                                                      Bool, latch=True)
        self.enable_controller_srv = rospy.Service("/face_adls/enable_controller", 
                                                   EnableFaceController, self.enable_controller_cb)
        self.request_registration = rospy.ServiceProxy("/request_registration", RequestRegistration)

        self.test_pose = rospy.Publisher('face_adls/test_pose', PoseStamped, latch=True)
        self.test_pose1 = rospy.Publisher('face_adls/test_pose1', PoseStamped, latch=True)
        self.test_pose2 = rospy.Publisher('face_adls/test_pose2', PoseStamped, latch=True)
        self.test_pose3 = rospy.Publisher('face_adls/test_pose3', PoseStamped, latch=True)

        def stop_move_cb(msg):
            self.stop_moving()
        self.stop_move_sub = rospy.Subscriber("/face_adls/stop_move", Bool, stop_move_cb, queue_size=1)

    def stop_moving(self):
        """Send empty PoseArray to skin controller to stop movement"""
        self.pose_traj_pub.publish(PoseArray()) #Send empty msg to skin controller

    def current_ee_pose(self, link, frame='/torso_lift_link'):
        """Get current end effector pose from tf"""
#        print "Getting pose of %s in frame: %s" %(link, frame)
        try:
            now = rospy.Time.now()
            self.tfl.waitForTransform(frame, link, now, rospy.Duration(8.0))
            pos, quat = self.tfl.lookupTransform(frame, link, now)
        except (LookupException, ConnectivityException, ExtrapolationException, Exception) as e:
            rospy.logwarn("[face_adls_manager] TF Failure getting current end-effector pose: %s" %e)
            return None
        return pos, quat 

    def publish_feedback(self, message=None):
        if message is not None:
            rospy.loginfo("[face_adls_manager] %s" % message)
            self.feedback_pub.publish(message)

    def enable_controller_cb(self, req):
        if req.enable:
            face_adls_modes = rospy.get_param('/face_adls_modes', None) 
            params = face_adls_modes[req.mode]
            self.shaving_side = rospy.get_param('/shaving_side', 'r') # TODO Make more general
            self.trim_retreat = req.mode == "shaving"
            self.flip_gripper = self.shaving_side == 'r' and req.mode == "shaving"
            bounds = params['%s_bounds' % self.shaving_side]
            #self.ellipsoid.set_bounds(bounds['lat'], bounds['lon'], bounds['height'])#TODO: Change Back
            self.ellipsoid.set_bounds((-np.pi,np.pi), (-np.pi,np.pi), (0,100))

            reg_resp = self.request_registration(req.mode, self.shaving_side)
            if not reg_resp.success:
                self.publish_feedback(Messages.NO_PARAMS_LOADED)
                return EnableFaceControllerResponse(False)

            reg_pose = PoseConv.to_pose_stamped_msg(reg_resp.e_params.e_frame)
            try:
                now = rospy.Time.now()
                reg_pose.header.stamp = now
                self.tfl.waitForTransform(reg_pose.header.frame_id, '/base_link', 
                                          now, rospy.Duration(8.0))
                ellipse_frame_base = self.tfl.transformPose('/base_link', reg_pose)
            except (LookupException, ConnectivityException, ExtrapolationException, Exception) as e:
                rospy.logwarn("[face_adls_manager] TF Failure converting ellipse to base frame: %s" %e)

            self.ellipsoid.load_ell_params(ellipse_frame_base,
                                           reg_resp.e_params.E,
                                           reg_resp.e_params.is_oblate,
                                           reg_resp.e_params.height)
            global_poses_dir = rospy.get_param("~global_poses_dir", "")
            global_poses_file = params["%s_ell_poses_file" % self.shaving_side]
            global_poses_resolved = roslaunch.substitution_args.resolve_args(
                                            global_poses_dir + "/" + global_poses_file)
            self.global_poses = rosparam.load_file(global_poses_resolved)[0][0]
            self.global_move_poses_pub.publish(sorted(self.global_poses.keys()))

            #Check rotating gripper based on side of body 
            if not self.flip_gripper:
                self.gripper_rot = trans.quaternion_from_euler(np.pi, 0, 0)
                print "Rotating gripper by PI around x-axis"
            else:
                self.gripper_rot = trans.quaternion_from_euler(0, 0, 0)
                print "NOT Rotating gripper around x-axis"

            # check if arm is near head
            cart_pos, cart_quat = self.current_ee_pose(self.ee_frame)
            ell_pos, ell_quat = self.ellipsoid.pose_to_ellipsoidal((cart_pos, cart_quat))
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

#            self.setup_force_monitor()

            success = True
            if not arm_in_bounds:
                self.publish_feedback(Messages.ARM_AWAY_FROM_HEAD)
                success = False

            #Switch back to l_arm_controller if necessary
            if self.controller_switcher.carefree_switch('l', '%s_arm_controller', reset=False):
                print "Controller switch to l_arm_controller succeeded"
                self.publish_feedback(Messages.ENABLE_CONTROLLER)
            else:
                print "Controller switch to l_arm_controller failed"
                success = False
            self.controller_enabled_pub.publish(Bool(success))
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
            curr_pose = self.current_ee_pose(self.ee_frame, '/ellipse_frame')
            ell_pos, _ = self.ellipsoid.pose_to_ellipsoidal(curr_pose)
            if ell_pos[2] < SAFETY_RETREAT_HEIGHT * 0.9:
                self.publish_feedback(Messages.DANGEROUS_FORCE %force)
                self.retreat_move(SAFETY_RETREAT_HEIGHT, 
                                  SAFETY_RETREAT_VELOCITY,
                                  forced=True)
        self.force_monitor.register_dangerous_cb(dangerous_cb)

        def timeout_cb(time):
            start_time = rospy.get_time()
            ell_pos, _ = self.ellipsoid.pose_to_ellipsoidal(self.current_ee_pose(self.ee_frame, '/ellipse_frame'))
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
                #rospy.loginfo("[face_adls_manager] Arm stopped due to making contact.")

        self.force_monitor.register_contact_cb(contact_cb)
        self.force_monitor.start_activity()
        self.force_monitor.update_activity()
        self.is_forced_retreat = False

    def retreat_move(self, height, velocity, forced=False):
        if forced:
            self.is_forced_retreat = True
        cart_pos, cart_quat = self.current_ee_pose(self.ee_frame, '/ellipse_frame')
        ell_pos, ell_quat = self.ellipsoid.pose_to_ellipsoidal((cart_pos, cart_quat))
        #print "Retreat EP:", ell_pos
        latitude = ell_pos[0]
        if self.trim_retreat:
            latitude = min(latitude, TRIM_RETREAT_LATITUDE)
        #rospy.loginfo("[face_adls_manager] Retreating from current location.")

        retreat_pos = [latitude, ell_pos[1], height]
        retreat_quat = [0,0,0,1]
        if forced:
            cart_path = self.ellipsoid.ellipsoidal_to_pose((retreat_pos, retreat_quat))
            cart_msg = [PoseConv.to_pose_msg(cart_path)]
        else:
            ell_path = self.ellipsoid.create_ellipsoidal_path(ell_pos,
                                                              ell_quat,
                                                              retreat_pos,
                                                              retreat_quat,
                                                              velocity=0.01,
                                                              min_jerk=True)
            cart_path = [self.ellipsoid.ellipsoidal_to_pose(ell_pose) for ell_pose in ell_path]
            cart_msg = [PoseConv.to_pose_msg(pose) for pose in cart_path]
            
        head = Header()
        head.frame_id = '/ellipse_frame'
        head.stamp = rospy.Time.now()
        pose_array = PoseArray(head, cart_msg)
        self.pose_traj_pub.publish(pose_array)

        self.is_forced_retreat = False

    def global_input_cb(self, msg):
        #self.force_monitor.update_activity()
        if self.is_forced_retreat:
            return
        rospy.loginfo("[face_adls_manager] Performing global move.")
        if type(msg) == String:
            self.publish_feedback(Messages.GLOBAL_START %msg.data)
            goal_ell_pose = self.global_poses[msg.data]
        elif type(msg) == PoseStamped:
            try:
                self.tfl.waitForTransform(msg.header.frame_id, '/ellipse_frame', msg.header.stamp, rospy.Duration(8.0))
                goal_cart_pose = self.tfl.transformPose('/ellipse_frame', msg)
                goal_cart_pos, goal_cart_quat = PoseConv.to_pos_quat(goal_cart_pose)
                flip_quat = trans.quaternion_from_euler(-np.pi/2, np.pi, 0)
                goal_cart_quat_flipped = trans.quaternion_multiply(goal_cart_quat, flip_quat)
                goal_cart_pose = PoseConv.to_pose_stamped_msg('/ellipse_frame', (goal_cart_pos, goal_cart_quat_flipped))
                self.publish_feedback('Moving around ellipse to clicked position).')
                goal_ell_pose = self.ellipsoid.pose_to_ellipsoidal(goal_cart_pose)
            except (LookupException, ConnectivityException, ExtrapolationException, Exception) as e:
                rospy.logwarn("[face_adls_manager] TF Failure getting clicked position in ellipse_frame:\r\n %s" %e)
                return

        curr_cart_pos, curr_cart_quat = self.current_ee_pose(self.ee_frame, '/ellipse_frame')
        curr_ell_pos, curr_ell_quat = self.ellipsoid.pose_to_ellipsoidal((curr_cart_pos, curr_cart_quat))
        retreat_ell_pos = [curr_ell_pos[0], curr_ell_pos[1], RETREAT_HEIGHT]
        retreat_ell_quat = trans.quaternion_multiply(self.gripper_rot, [1.,0.,0.,0.])
        approach_ell_pos = [goal_ell_pose[0][0], goal_ell_pose[0][1], RETREAT_HEIGHT]
        #approach_ell_quat = trans.quaternion_multiply(self.gripper_rot, goal_ell_pose[1])
        approach_ell_quat = goal_ell_pose[1]
        final_ell_pos = [goal_ell_pose[0][0], goal_ell_pose[0][1], goal_ell_pose[0][2] - HEIGHT_CLOSER_ADJUST]
        #final_ell_quat = trans.quaternion_multiply(self.gripper_rot, goal_ell_pose[1])
        final_ell_quat = goal_ell_pose[1]
        
        cart_ret_pose = self.ellipsoid.ellipsoidal_to_pose((retreat_ell_pos, retreat_ell_quat))
        cart_ret_msg = PoseConv.to_pose_stamped_msg('ellipse_frame',cart_ret_pose)
        self.test_pose.publish(cart_ret_msg)

        cart_app_pose = self.ellipsoid.ellipsoidal_to_pose((approach_ell_pos, approach_ell_quat))
        cart_app_msg = PoseConv.to_pose_stamped_msg('ellipse_frame',cart_app_pose)
        self.test_pose1.publish(cart_app_msg)

        cart_goal_pose = self.ellipsoid.ellipsoidal_to_pose((final_ell_pos, final_ell_quat))
        cart_goal_msg = PoseConv.to_pose_stamped_msg('ellipse_frame',cart_goal_pose)
        self.test_pose2.publish(cart_goal_msg)

        retreat_ell_traj = self.ellipsoid.create_ellipsoidal_path(curr_ell_pos, curr_ell_quat,
                                                                  retreat_ell_pos, retreat_ell_quat,
                                                                  velocity=0.01, min_jerk=True)
        transition_ell_traj = self.ellipsoid.create_ellipsoidal_path(retreat_ell_pos, retreat_ell_quat,
                                                                     approach_ell_pos, approach_ell_quat,
                                                                     velocity=0.01, min_jerk=True)
        approach_ell_traj = self.ellipsoid.create_ellipsoidal_path(approach_ell_pos, approach_ell_quat,
                                                                   final_ell_pos, final_ell_quat,
                                                                   velocity=0.01, min_jerk=True)
        
        full_ell_traj = retreat_ell_traj + transition_ell_traj + approach_ell_traj
        full_cart_traj = [self.ellipsoid.ellipsoidal_to_pose(pose) for pose in full_ell_traj]
        cart_traj_msg = [PoseConv.to_pose_msg(pose) for pose in full_cart_traj]
        head = Header()
        head.frame_id = '/ellipse_frame'
        head.stamp = rospy.Time.now()
        pose_array = PoseArray(head, cart_traj_msg)
        self.pose_traj_pub.publish(pose_array)

        ps = PoseStamped()
        ps.header = head
        ps.pose = cart_traj_msg[0]
        self.test_pose3.publish(ps)
        
#        self.force_monitor.update_activity()
            
    def local_input_cb(self, msg):
        #self.force_monitor.update_activity()
        if self.is_forced_retreat:
            return
        self.stop_moving()
        button_press = msg.data 

        curr_cart_pos, curr_cart_quat = self.current_ee_pose(self.ee_frame, '/ellipse_frame')
        curr_ell_pos, curr_ell_quat = self.ellipsoid.pose_to_ellipsoidal((curr_cart_pos, curr_cart_quat))

        if button_press in ell_trans_params:
            self.publish_feedback(Messages.LOCAL_START % button_names_dict[button_press])
            change_trans_ep = ell_trans_params[button_press]
            goal_ell_pos = [curr_ell_pos[0]+change_trans_ep[0],
                            curr_ell_pos[1]+change_trans_ep[1],
                            curr_ell_pos[2]+change_trans_ep[2]]
            ell_path = self.ellipsoid.create_ellipsoidal_path(curr_ell_pos, curr_ell_quat,
                                                              goal_ell_pos, curr_ell_quat,
                                                              velocity=ELL_LOCAL_VEL, min_jerk=True)
        elif button_press in ell_rot_params:
            self.publish_feedback(Messages.LOCAL_START % button_names_dict[button_press])
            change_rot_ep = ell_rot_params[button_press]
            rot_quat = trans.quaternion_from_euler(*change_rot_ep)
            goal_ell_quat = trans.quaternion_multiply(curr_ell_quat, rot_quat)
            ell_path = self.ellipsoid.create_ellipsoidal_path(curr_ell_pos, curr_ell_quat,
                                                              curr_ell_pos, goal_ell_quat,
                                                              velocity=ELL_ROT_VEL, min_jerk=True)
        elif button_press == "reset_rotation":
            self.publish_feedback(Messages.ROT_RESET_START)
            ell_path = self.ellipsoid.create_ellipsoidal_path(curr_ell_pos, curr_ell_quat,
                                                              curr_ell_pos,(0.,0.,0.,1.),
                                                              velocity=ELL_ROT_VEL, min_jerk=True)
        else:
            rospy.logerr("[face_adls_manager] Unknown local ellipsoidal command")

        cart_traj = [self.ellipsoid.ellipsoidal_to_pose(pose) for pose in ell_path]
        cart_traj_msg = [PoseConv.to_pose_msg(pose) for pose in cart_traj]
        head = Header()
        head.frame_id = '/ellipse_frame'
        head.stamp = rospy.Time.now()
        pose_array = PoseArray(head, cart_traj_msg)
        self.pose_traj_pub.publish(pose_array)
        #self.force_monitor.update_activity()

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

    #def global_input_cb_old(self, msg):
    #    if self.is_forced_retreat:
    #        return
    #    if self.stop_move():
    #        rospy.loginfo("[face_adls_manager] Preempting other movement for global move.")
    #        #self.publish_feedback(Messages.GLOBAL_PREEMPT_OTHER)
    #    self.force_monitor.start_activity()
    #    goal_pose = self.global_poses[msg.data]
    #    goal_pose_name = msg.data
    #    self.publish_feedback(Messages.GLOBAL_START % goal_pose_name)
    #    try:
    #        if not self.ell_ctrl.execute_ell_move(((0, 0, RETREAT_HEIGHT), np.mat(np.eye(3))), 
    #                                              ((0, 0, 1), 1), 
    #                                              self.gripper_rot,
    #                                              APPROACH_VELOCITY,
    #                                              blocking=True):
    #            raise Exception
    #        if not self.ell_ctrl.execute_ell_move(((goal_pose[0][0], goal_pose[0][1], RETREAT_HEIGHT), (0, 0, 0, 1)), 
    #                                              ((1, 1, 1), 0), 
    #                                              self.gripper_rot,
    #                                              GLOBAL_VELOCITY,
    #                                              blocking=True):
    #            raise Exception
    #        final_goal = [goal_pose[0][0], goal_pose[0][1], goal_pose[0][2] - HEIGHT_CLOSER_ADJUST]
    #        if not self.ell_ctrl.execute_ell_move((final_goal, (0, 0, 0, 1)),
    #                                              ((1, 1, 1), 0), 
    #                                              self.gripper_rot,
    #                                              GLOBAL_VELOCITY,
    #                                              blocking=True):
    #            raise Exception
    #    except:
    #        self.publish_feedback(Messages.GLOBAL_PREEMPT % goal_pose_name)
    #        self.force_monitor.stop_activity()
    #        return
    #    self.publish_feedback(Messages.GLOBAL_SUCCESS % goal_pose_name)
    #    self.force_monitor.stop_activity()

#    def local_input_cb(self, msg):
#        if self.is_forced_retreat:
#            return
#        self.stop_moving()
#        self.force_monitor.start_activity()
#        button_press = msg.data 
#
#        curr_cart_pos, curr_cart_quat = self.current_ee_pose(self.ee_frame, '/ellipse_frame')
#        curr_ell_pos, curr_ell_quat = self.ellipsoid.pose_to_ellipsoidal((curr_cart_pos, curr_cart_quat))
#
#        if button_press in ell_trans_params:
#            self.publish_feedback(Messages.LOCAL_START % button_names_dict[button_press])
#            change_trans_ep = ell_trans_params[button_press]
#            goal_ell_pos = [curr_ell_pos[0]+change_trans_ep[0],
#                            curr_ell_pos[1]+change_trans_ep[1],
#                            curr_ell_pos[2]+change_trans_ep[2]]
#            ell_path = self.ellipsoid.create_ellipsoidal_path(curr_ell_pos, curr_ell_quat,
#                                                              goal_ell_pos, curr_ell_quat,
#                                                              velocity=ELL_LOCAL_VEL, min_jerk=True)
##            success = self.ell_ctrl.execute_ell_move((change_trans_ep, (0, 0, 0, 1)),
##                                                     ((0, 0, 0), 0), 
##                                                    self.gripper_rot,
##                                                    ELL_LOCAL_VEL,
##                                                    blocking=True)
#        elif button_press in ell_rot_params:
#            self.publish_feedback(Messages.LOCAL_START % button_names_dict[button_press])
#            change_rot_ep = ell_rot_params[button_press]
#            rot_quat = trans.quaternion_from_euler(*change_rot_ep)
#            goal_ell_quat = trans.quaternion_multiply(curr_ell_quat, rot_quat)
#            ell_path = self.ellipsoid.create_ellipsoidal_path(curr_ell_pos, curr_ell_quat,
#                                                              curr_ell_pos, goal_ell_quat,
#                                                              velocity=ELL_ROT_VEL, min_jerk=True)
#            #success = self.ell_ctrl.execute_ell_move(((0, 0, 0), rot_quat),
#            #                                         ((0, 0, 0), 0), 
#            #                                         self.gripper_rot,
#            #                                         ELL_ROT_VEL,
#            #                                         blocking=True)
#        elif button_press == "reset_rotation":
#            self.publish_feedback(Messages.ROT_RESET_START)
#            ell_path = self.ellipsoid.create_ellipsoidal_path(curr_ell_pos, curr_ell_quat,
#                                                              curr_ell_pos,(0.,0.,0.,1.),
#                                                              velocity=ELL_ROT_VEL, min_jerk=True)
#            #success = self.ell_ctrl.execute_ell_move(((0, 0, 0), np.mat(np.eye(3))),
#            #                                         ((0, 0, 0), 1), 
#            #                                         self.gripper_rot,
#            #                                         ELL_ROT_VEL,
#            #                                         blocking=True)
#        else:
#            rospy.logerr("[face_adls_manager] Unknown local ellipsoidal command")
#
#        cart_traj = [self.ellipsoid.ellipsoidal_to_pose(pose) for pose in ell_path]
#        cart_traj_msg = [PoseConv.to_pose_msg(pose) for pose in cart_traj]
#        head = Header()
#        head.frame_id = '/ellipse_frame'
#        head.stamp = rospy.Time.now()
#        pose_array = PoseArray(head, cart_traj_msg)
#        self.pose_traj_pub.publish(pose_array)
#        #if success:
#        #    self.publish_feedback(Messages.LOCAL_SUCCESS % button_names_dict[button_press])
#        #else:
#        #    self.publish_feedback(Messages.LOCAL_PREEMPT % button_names_dict[button_press])
#        #self.force_monitor.stop_activity()
