#!/usr/bin/python

import roslib
import rospy, rospkg, rosparam
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
import numpy as np
import math as m
import os.path
import ghmm
import copy

roslib.load_manifest('hrl_dressing')
import tf, argparse
import threading
import hrl_lib.util as utils

roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle
from hrl_msgs.msg import FloatArray
from pr2_controllers_msgs.msg import SingleJointPositionActionGoal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import tf.transformations as tft

from hrl_base_selection.helper_functions import createBMatrix, Bmat_to_pos_quat
from hrl_geom.pose_converter import rot_mat_to_axis_angle

from geometry_msgs.msg import PoseArray, PointStamped, Pose, PoseStamped, Point, Quaternion, PoseWithCovarianceStamped, Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from ar_track_alvar_msgs.msg import AlvarMarkers
from pr2_controllers_msgs.msg import PointHeadGoal, PointHeadActionGoal
import hrl_lib.circular_buffer as cb
from hrl_msgs.msg import FloatArrayBare


class AR_Tracking_for_Dressing(object):
    def __init__(self, model):
        self.start_tracking_ar_tag = False
        self.tag_id = 4
        self.action_goal = PointHeadActionGoal()
        self.goal = PointHeadGoal()
        self.point = PointStamped()
        self.goal.pointing_frame = 'head_mount_kinect_ir_link'
        self.point.header.frame_id = 'base_link'
        self.goal.min_duration = rospy.Duration(0.25)

        self.hist_size = 17
        self.ar_count = 0
        self.pos_buf = cb.CircularBuffer(self.hist_size, (3,))
        self.quat_buf = cb.CircularBuffer(self.hist_size, (4,))

        self.frame_lock = threading.RLock()

        ang = m.radians(0.)
        if model == 'lab_wheelchair':
            self.wheelchair_B_ar_ground = np.matrix([[m.cos(ang), -m.sin(ang), 0., -0.4],
                                                     [m.sin(ang), m.cos(ang), 0., 0.],
                                                     [0., 0., 1., 0.],
                                                     [0., 0., 0., 1.]])

        self.wheelchair_B_pr2 = np.matrix(np.eye(4))

        self.head_track_AR_pub = rospy.Publisher('/head_traj_controller/point_head_action/goal',
                                                 PointHeadActionGoal, queue_size=1)
        rospy.sleep(0.3)
        self.goal_pose_subscriber = rospy.Subscriber("dressing_pr2_pose", FloatArrayBare, self.goal_pose_cb)
        self.tracking_trigger_subscriber = rospy.Subscriber("dressing_ar_tracking_start", Bool, self.trigger_tracking_ar_tag_cb)
        self.ar_tag_subscriber = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.arTagCallback)
        rospy.sleep(0.1)
        print 'AR Tracking is Ready!'

        inp = ''
        while not rospy.is_shutdown() and not inp.upper() == 'Q':
            inp = raw_input('\nY (y) to start tracking. N (n) to stop tracking. Q (q) quits.\n')
            if len(inp) == 0:
                pass
            elif inp.upper()[0] == 'Y':
                self.start_tracking_ar_tag = True
            elif inp.upper()[0] == 'N':
                self.start_tracking_ar_tag = False
            else:
                pass
            rospy.sleep(0.3)

    def goal_pose_cb(self, msg):
        pr2_params = msg.data
        x = pr2_params[0]
        y = pr2_params[1]
        th = pr2_params[2]

        self.wheelchair_B_pr2 = np.matrix([[m.cos(th), -m.sin(th), 0., x],
                                           [m.sin(th), m.cos(th), 0., y],
                                           [0., 0., 1., 0.],
                                           [0., 0., 0., 1.]])

    def trigger_tracking_ar_tag_cb(self, msg):
        if msg.data:
            self.start_tracking_ar_tag = True
        else:
            self.start_tracking_ar_tag = False

    def arTagCallback(self, msg):
        with self.frame_lock:
            if self.start_tracking_ar_tag:
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

                        if len(self.quat_buf) < 1:
                            self.pos_buf.append(cur_p)
                            self.quat_buf.append(cur_q)
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
                        if True:
                            # median
                            positions = np.sort(positions, axis=0)
                            pos_int = positions[len(positions) / 2 - 1:len(positions) / 2 + 1]
                            pos = np.sum(pos_int, axis=0)
                            pos /= float(len(pos_int))

                            quaternions = np.sort(quaternions, axis=0)
                            quat_int = quaternions[len(quaternions) / 2 - 1:len(quaternions) / 2 + 1]
                            quat = np.sum(quat_int, axis=0)
                            quat /= float(len(quat_int))
                        current_pr2_B_ar = createBMatrix(pos, quat)
                        current_pr2_B_ar_ground = self.shift_to_ground(current_pr2_B_ar)
                        current_pr2_B_wheelchair = current_pr2_B_ar_ground * self.wheelchair_B_ar_ground.I

                        self.point.point.x = pos[0]
                        self.point.point.y = pos[1]
                        self.point.point.z = pos[2]
                        self.goal.target = self.point

                        self.goal.pointing_axis.x = 1
                        self.goal.pointing_axis.y = 0
                        self.goal.pointing_axis.z = 0

                        self.action_goal.goal = self.goal
                        self.head_track_AR_pub.publish(self.action_goal)

                        current_B_target = current_pr2_B_wheelchair*self.wheelchair_B_pr2
                        angle, axis, junk_point = tft.rotation_from_matrix(current_B_target)
                        if axis[np.argmax(np.abs(axis))] < 0.:
                            angle *= -1.

                        print 'Error (x, y, theta(z)):', current_B_target[0, 3], current_B_target[1, 3], m.degrees(angle)

    def shift_to_ground(self, this_map_B_ar):
        with self.frame_lock:
            # now = rospy.Time.now()
            # self.listener.waitForTransform('/torso_lift_link', '/base_footprint', now, rospy.Duration(5))
            # (trans, rot) = self.listener.lookupTransform('/torso_lift_link', '/base_footprint', now)

            z_origin = np.array([0, 0, 1])
            x_bed = np.array([this_map_B_ar[0, 2], this_map_B_ar[1, 2], this_map_B_ar[2, 2]])
            y_bed_project = np.cross(z_origin, x_bed)
            y_bed_project = y_bed_project / np.linalg.norm(y_bed_project)
            x_bed_project = np.cross(y_bed_project, z_origin)
            x_bed_project = x_bed_project / np.linalg.norm(x_bed_project)
            map_B_ar_project = np.eye(4)
            #for i in xrange(3):
            map_B_ar_project[0:3, 0] = x_bed_project
            map_B_ar_project[0:3, 1] = y_bed_project
            map_B_ar_project[0:3, 2] = z_origin
            map_B_ar_project[0, 3] = this_map_B_ar[0, 3]
            map_B_ar_project[1, 3] = this_map_B_ar[1, 3]
            # liftlink_B_footprint = createBMatrix(trans, rot)

            return map_B_ar_project

if __name__ == '__main__':
    rospy.init_node('dressing_ar_tag_tracking')

    import optparse
    p = optparse.OptionParser()
    p.add_option('--model', action='store', dest='model', default='lab_wheelchair', type='string',
                 help='Select what AR tag to look for (e.g. head, autobed)')
    opt, args = p.parse_args()
    tracker = AR_Tracking_for_Dressing(opt.model)
    rospy.spin()

