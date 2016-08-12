#!/usr/bin/env python

import rospy
import roslib
roslib.load_manifest('hrl_base_selection')
import numpy as np
import os, threading, copy

import hrl_lib.circular_buffer as cb
import tf
from ar_track_alvar_msgs.msg import AlvarMarkers
from helper_functions import createBMatrix, Bmat_to_pos_quat
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, PointStamped, Twist  # , PoseArray
from hrl_msgs.msg import FloatArrayBare
from pr2_controllers_msgs.msg import PointHeadGoal, PointHeadActionGoal


class AR_Tag_Tracking(object):

    def __init__(self, mode):
        print 'Starting detection of', mode, 'AR tag'
        self.frame_lock = threading.RLock()
        self.mode = mode

        self.find_AR = False
        self.currently_acquiring_AR_tag = False
        self.finished_acquiring_AR_tag = False
        self.track_AR = False
        self.servo_base = False
        self.goal_location = None
        self.map_B_ar_pos = None

        self.action_goal = PointHeadActionGoal()
        self.goal = PointHeadGoal()
        self.point = PointStamped()
        self.goal.pointing_frame = 'head_mount_kinect_ir_link'
        self.point.header.frame_id = 'base_link'
        self.goal.min_duration = rospy.Duration(0.25)
        self.currently_tracking_AR = False


        self.listener = tf.TransformListener()
        self.broadcaster = tf.TransformBroadcaster()

        self.out_pos = None
        self.out_quat = None

        self.tag_id = None
        self.tag_side_length = None

        self.autobed_sub = None

        self.hist_size = 10
        self.ar_count = 0
        self.pos_buf = cb.CircularBuffer(self.hist_size, (3,))
        self.quat_buf = cb.CircularBuffer(self.hist_size, (4,))

        # The homogeneous transform from the reference point of interest (e.g. the center of the head, the base of the
        # bed model).
        self.reference_B_ar = np.eye(4)

        # while not self.listener.canTransform('base_link', 'base_link', rospy.Time(0)) and not rospy.is_shutdown():
        #     rospy.sleep(2)
        #     print self.mode, ' AR tag waiting for the map transform.'

        if self.mode == 'autobed':
            self.out_frame = 'autobed/base_link'
            self.bed_state_z = 0.
            self.bed_state_head_theta = 0.
            self.bed_state_leg_theta = 0.
            self.config_autobed_AR_detector()
        else:
            print 'I do not know what AR tag to look for... Abort!'
            return

        self.start_finding_AR_subscriber = rospy.Subscriber('find_AR_now', Bool, self.start_finding_AR_cb)
        # self.start_finding_AR_publisher = rospy.Publisher('find_AR_now', Bool, queue_size=1)

        self.AR_tag_acquired = rospy.Publisher('AR_acquired', Bool, queue_size=1)
        self.AR_tag_tracking = rospy.Publisher('/AR_tracking', Bool, queue_size=1, latch=True)
        self.start_tracking_AR_subscriber = rospy.Subscriber('track_AR_now', Bool, self.start_tracking_AR_cb)
        # self.start_tracking_AR_publisher = rospy.Publisher('track_AR_now', Bool, queue_size=1)
        self.ar_tag_distance_pub = rospy.Publisher('/ar_tag_distance', FloatArrayBare, queue_size = 1)
        self.head_track_AR_pub = rospy.Publisher('/head_traj_controller/point_head_action/goal', PointHeadActionGoal, queue_size=1)

        self.ar_tag_subscriber = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.arTagCallback)

        # self.servo_base_publisher = rospy.Publisher('/base_controller/command', Twist, queue_size=1)

        self.run()

    def start_finding_AR_cb(self, msg):
        with self.frame_lock:
            if msg.data and not self.currently_acquiring_AR_tag:
                self.currently_acquiring_AR_tag = True
                self.finished_acquiring_AR_tag = False
                false_out = Bool()
                false_out.data = False
                self.AR_tag_acquired.publish(false_out)
                self.hist_size = 10
                self.ar_count = 0
                self.pos_buf = cb.CircularBuffer(self.hist_size, (3,))
                self.quat_buf = cb.CircularBuffer(self.hist_size, (4,))
                # while not self.listener.canTransform('base_link', 'base_link', rospy.Time(0)) and not rospy.is_shutdown():
                #     rospy.sleep(2)
                #     print self.mode, ' AR tag waiting for the map transform.'
                #     #now = rospy.Time.now()
                # self.pose_pub = rospy.Publisher(''.join(['ar_tag_tracking/', self.mode, '_pose']), PoseStamped,
                #                                 queue_size=1, latch=True)
                print 'Started acquiring the AR tag location!'

                # rospy.sleep(1)
                # while self.currently_finding_AR and not rospy.is_shutdown():
                #     rospy.sleep(1)
                # self.ar_tag_subscriber.unregister()
            elif not msg.data:
                self.currently_finding_AR = False
                self.finished_acquiring_AR_tag = False
                print 'Stopped finding AR tag'
                false_out = Bool()
                false_out.data = False
                self.AR_tag_acquired.publish(false_out)
                # self.ar_tag_subscriber.unregister()
            else:
                print 'Asked to find AR tag but already in the process of acquiring AR tag!'
                # rospy.sleep(5)

    def start_tracking_AR_cb(self, msg):
        # print msg
        # print self.track_AR
        if msg.data and not self.currently_tracking_AR:
            print 'THINGS AND STUFF Starting to track the AR tag!'
            # self.track_AR = msg.data
            # self.tracking_AR()
        elif not msg.data and self.currently_tracking_AR:
            print 'THINGS AND STUFF Stopping tracking the AR tag!'

        failure = Bool()
        failure.data = False
        self.AR_tag_tracking.publish(failure)
        self.currently_tracking_AR = msg.data

    def tracking_AR(self):
        while self.track_AR and not rospy.is_shutdown() and self.map_B_ar_pos is not None:
            self.action_goal = PointHeadActionGoal()
            self.goal = PointHeadGoal()

            # The point to be looking at is expressed in the 'odom_combined' frame
            self.point = PointStamped()
            point.header.frame_id = 'base_link'
            point.point.x = self.map_B_ar_pos[0]
            point.point.y = self.map_B_ar_pos[1]
            point.point.z = self.map_B_ar_pos[2]
            goal.target = point

            # We want the X axis of the camera frame to be pointing at the target
            goal.pointing_frame = 'head_mount_kinect_ir_link'
            goal.pointing_axis.x = 1
            goal.pointing_axis.y = 0
            goal.pointing_axis.z = 0
            goal.min_duration = rospy.Duration(0.5)

            action_goal.goal = goal
            self.head_track_AR_pub.publish(action_goal)

            rospy.sleep(.5)

    '''
    def servo_base_cb(self, msg):
        self.servo_base = msg.data

    # Takes in a goal in the odom_combined frame
    def servo_base_cb(self, bed_B_goal):
        while self.servo_base and not rospy.is_shutdown():

            now = rospy.Time.now()
            (trans, rot) = self.listener.lookupTransform('/autobed/base_link', '/base_link', now)
            self.bed_B_pr2 = createBMatrix(trans, rot)
            desired_movement = self.bed_B_pr2.I*bed_B_goal
            error_ori = -m.acos(desired_movement[0, 0])
            while (np.linalg.norm(error_ori)>0.1) and not rospy.is_shutdown() and self.servo_base:
                now = rospy.Time.now()
                (trans, rot) = self.listener.lookupTransform('/autobed/base_link', '/base_link', now)
                self.bed_B_pr2 = createBMatrix(trans, rot)
                desired_movement = self.bed_B_pr2.I*bed_B_goal
                error_ori = -m.acos(desired_movement[0, 0])
                error_ori = goal[2] - math.acos(self.Bubase[1,1])
                mat = np.matrix([[math.cos(goal[2]),-math.sin(goal[2]),0],[math.sin(goal[2]),math.cos(goal[2]),0],[0,0,1]])
                Bbasegoal = self.Bubase.I*createGoalBMatrix([0,0,0],tr.matrix_to_quaternion(mat))
                move = -math.acos(Bbasegoal[1,1])
                error_ori = move
                normalized_ori = move / (np.linalg.norm(move)*5)
                print 'Bubase: \n',self.Bubase
                print 'goal: \n', goal
                print 'error_ori: \n', error_ori
                print 'move: \n', move

                    tw = Twist()
                tw.linear.x=0
                tw.linear.y=0
                tw.linear.z=0
                tw.angular.x=0
                tw.angular.y=0
                tw.angular.z=normalized_ori
                pub1.publish(tw)
                rospy.sleep(.1)
            print 'Bubase: \n',self.Bubase
            print 'goal: \n', goal
            error_pos = [goal[0] - self.Bubase[0,3], goal[1] - self.Bubase[1,3] ,0]
            while (np.linalg.norm(error_pos)>0.1) and not rospy.is_shutdown() and self.servo_base:
                error_pos = [goal[0] - self.Bubase[0,3], goal[1] - self.Bubase[1,3] ,0]
                Bbasegoal = self.Bubase.I*createGoalBMatrix([goal[0],goal[1],0],[0,0,0,1])
                move = np.array([Bbasegoal[0,3],Bbasegoal[1,3],Bbasegoal[2,3]])
                normalized_pos = move / (np.linalg.norm(move)*5)
                print 'Bubase: \n',self.Bubase
                print 'goal: \n', goal
                print 'error_pos: \n', error_pos
                print 'move: \n', move

                    tw = Twist()
                tw.linear.x=normalized_pos[0]
                tw.linear.y=normalized_pos[1]
                tw.linear.z=0
                tw.angular.x=0
                tw.angular.y=0
                tw.angular.z=0

            pub1.publish(tw)
            rospy.sleep(.1)
    '''

    def run(self):
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            #print self.out_pos, self.out_quat
            if self.finished_acquiring_AR_tag:
                self.broadcaster.sendTransform(self.out_pos, self.out_quat,
                                               rospy.Time.now(),
                                               self.out_frame,
                                               'base_footprint')
            # if self.currently_tracking_AR:
                #print 'broadcast transform'
            rate.sleep()

    def config_autobed_AR_detector(self):
        self.tag_id = 4  # 9

        # self.autobed_sub = rospy.Subscriber('/abdout0', FloatArrayBare, self.bed_state_cb)
        self.tag_side_length = 0.15  # 0.053  # 0.033

        # This is the translational transform from reference markers to the bed origin.
        # -.445 if right side of body. .445 if left side.
        model_trans_B_ar = np.eye(4)
        # model_trans_B_ar[0:3, 3] = np.array([-0.01, .00, 1.397])
        # Now that I adjust the AR tag pose to be on the ground plane, no Z shift needed.
        model_trans_B_ar[0:3, 3] = np.array([-0.03, 0.00, 0.])
        ar_rotz_B = np.eye(4)
        #ar_rotz_B[0:2, 0:2] = np.array([[-1, 0], [0, -1]])

        ar_roty_B = np.eye(4)
        #ar_roty_B[0:3, 0:3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        # If left side of bed should be np.array([[-1,0],[0,-1]])
        # If right side of bed should be np.array([[1,0],[0,1]])
        ar_rotx_B = np.eye(4)
        # ar_rotx_B[1:3, 1:3] = np.array([[0,1],[-1,0]])
        self.reference_B_ar = np.matrix(model_trans_B_ar)*np.matrix(ar_roty_B)*np.matrix(ar_rotz_B)

    # When we are using the autobed, we probably need to know the state of the autobed. This records the current
    # state of the autobed.
    def bed_state_cb(self, data):
        self.bed_state_z = data.data[1]
        self.bed_state_head_theta = data.data[0]
        self.bed_state_leg_theta = data.data[2]

    def arTagCallback(self, msg):
        with self.frame_lock:
            markers = msg.markers
            for i in xrange(len(markers)):
                if markers[i].id == self.tag_id:
                    cur_p = np.array([markers[i].pose.pose.position.x,
                                      markers[i].pose.pose.position.y,
                                      markers[i].pose.pose.position.z])
                    cur_q = np.array([markers[i].pose.pose.orientation.x,
                                      markers[i].pose.pose.orientation.y,
                                      markers[i].pose.pose.orientation.z,
                                      markers[i].pose.pose.orientation.w])

                    #frame_id = markers[i].pose.header.frame_id
                    #print 'Frame ID is: ', frame_id

                    # if np.linalg.norm(cur_p) > 4.0:
                    #     print "Detected tag is located too far away."
                    #     continue

                    if len(self.quat_buf) < 1:
                        self.pos_buf.append( cur_p )
                        self.quat_buf.append( cur_q )
                    else:
                        first_p = self.pos_buf[0]
                        first_q = self.quat_buf[0]

                        # check close quaternion and inverse
                        if np.dot(cur_q, first_q) < 0.0:
                            cur_q *= -1.0

                        self.pos_buf.append(cur_p)
                        self.quat_buf.append(cur_q)

                    positions = self.pos_buf.get_array()
                    quaternions = self.quat_buf.get_array()

                    pos = None
                    quat = None
                    if False:
                        # Moving average
                        pos = np.sum(positions, axis=0)
                        pos /= float(len(positions))

                        quat = np.sum(quaternions, axis=0)
                        quat /= float(len(quaternions))
                    else:
                        # median
                        positions = np.sort(positions, axis=0)
                        pos_int = positions[len(positions)/2-1:len(positions)/2+1]
                        pos = np.sum(pos_int, axis=0)
                        pos /= float(len(pos_int))

                        quaternions = np.sort(quaternions, axis=0)
                        quat_int = quaternions[len(quaternions)/2-1:len(quaternions)/2+1]
                        quat = np.sum(quat_int, axis=0)
                        quat /= float(len(quat_int))
                    self.map_B_ar_pos = pos
                    if self.currently_acquiring_AR_tag and not self.finished_acquiring_AR_tag and self.ar_count <= self.hist_size:
                        self.ar_count += 1
                    elif self.currently_acquiring_AR_tag and not self.finished_acquiring_AR_tag:
                        success = Bool()
                        success.data = True
                        self.AR_tag_acquired.publish(success)
                        self.finished_acquiring_AR_tag = True
                        self.currently_acquiring_AR_tag = False
                        print 'Finished acquiring AR tag'
                    # else:
                    #     success = Bool()
                    #     success.data = True
                    #     self.AR_tag_acquired.publish(success)
                    #     self.finished_acquiring_AR_tag = True
                    #     self.currently_acquiring_AR_tag = False
                    #     print 'Finished acquiring AR tag'
                    if self.finished_acquiring_AR_tag:
                        map_B_ar = createBMatrix(pos, quat)

                        if self.mode == 'autobed':
                            map_B_ar = self.shift_to_ground(map_B_ar)

                        self.out_pos, self.out_quat = Bmat_to_pos_quat(map_B_ar*self.reference_B_ar.I)
                    #print self.currently_tracking_AR, self.finished_acquiring_AR_tag
                    if self.currently_tracking_AR and self.finished_acquiring_AR_tag:
                        # The point to be looking at is expressed in the 'odom_combined' frame
                        self.point.point.x = self.map_B_ar_pos[0]
                        self.point.point.y = self.map_B_ar_pos[1]
                        self.point.point.z = self.map_B_ar_pos[2]
                        self.goal.target = self.point

                        # We want the X axis of the camera frame to be pointing at the target

                        self.goal.pointing_axis.x = 1
                        self.goal.pointing_axis.y = 0
                        self.goal.pointing_axis.z = 0

                        self.action_goal.goal = self.goal
                        self.ar_tag_distance_pub.publish(self.map_B_ar_pos)
                        self.head_track_AR_pub.publish(self.action_goal)
                        success = Bool()
                        success.data = True
                        self.AR_tag_tracking.publish(success)
                    else:
                        failure= Bool()
                        failure.data = False
                        self.AR_tag_tracking.publish(failure)
                       
                        # ps = PoseStamped()
                        # ps.header.frame_id = 'torso_lift_link'  # markers[i].pose.header.frame_id
                        # ps.header.stamp = rospy.Time.now()  # markers[i].pose.header.stamp
                        # ps.pose.position.x = out_pos[0]
                        # ps.pose.position.y = out_pos[1]
                        # ps.pose.position.z = out_pos[2]
                        #
                        # ps.pose.orientation.x = out_quat[0]
                        # ps.pose.orientation.y = out_quat[1]
                        # ps.pose.orientation.z = out_quat[2]
                        # ps.pose.orientation.w = out_quat[3]

                        # self.pose_pub.publish(ps)

    # I now project the bed pose onto the ground plane to mitigate potential problems with AR tag orientation
    def shift_to_ground(self, this_map_B_ar):
        with self.frame_lock:
            # now = rospy.Time.now()
            # self.listener.waitForTransform('/torso_lift_link', '/base_footprint', now, rospy.Duration(5))
            # (trans, rot) = self.listener.lookupTransform('/torso_lift_link', '/base_footprint', now)

            ar_rotz_B = np.eye(4)
            ar_rotz_B[0:2, 0:2] = np.array([[-1, 0], [0, -1]])

            ar_roty_B = np.eye(4)
            ar_roty_B[0:3, 0:3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            ar_rotx_B = np.eye(4)
            # ar_rotx_B[1:3, 1:3] = np.array([[0,1],[-1,0]])
            orig_B_properly_oriented = np.matrix(ar_roty_B)*np.matrix(ar_rotz_B)

            this_map_B_ar = this_map_B_ar*orig_B_properly_oriented.I

            z_origin = np.array([0, 0, 1])
            x_bed = np.array([this_map_B_ar[0, 0], this_map_B_ar[1, 0], this_map_B_ar[2, 0]])
            y_bed_project = np.cross(z_origin, x_bed)
            y_bed_project = y_bed_project/np.linalg.norm(y_bed_project)
            x_bed_project = np.cross(y_bed_project, z_origin)
            x_bed_project = x_bed_project/np.linalg.norm(x_bed_project)
            map_B_ar_project = np.eye(4)
            for i in xrange(3):
                map_B_ar_project[i, 0] = x_bed_project[i]
                map_B_ar_project[i, 1] = y_bed_project[i]
                map_B_ar_project[i, 3] = this_map_B_ar[i, 3]
            # liftlink_B_footprint = createBMatrix(trans, rot)

            map_B_ar_floor = copy.deepcopy(np.matrix(map_B_ar_project))
            map_B_ar_floor[2, 3] = 0.
            return map_B_ar_floor

if __name__ == '__main__':
    rospy.init_node('ar_tag_servoing')

    import optparse
    p = optparse.OptionParser()
    p.add_option('--mode', action='store', dest='mode', default='autobed', type='string',
                 help='Select what AR tag to look for (e.g. head, autobed)')
    opt, args = p.parse_args()
    atd = AR_Tag_Tracking(opt.mode)
    rospy.spin()
