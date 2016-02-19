#!/usr/bin/env python

import rospy
import roslib
roslib.load_manifest('hrl_base_selection')
import numpy as np
import os, threading, copy

import hrl_lib.circular_buffer as cb

from ar_track_alvar_msgs.msg import AlvarMarkers
from helper_functions import createBMatrix, Bmat_to_pos_quat
from geometry_msgs.msg import PoseStamped  # , PointStamped, PoseArray
from hrl_msgs.msg import FloatArrayBare


class arTagDetector:

    def __init__(self, mode):
        print 'Starting detection of', mode, 'AR tag'
        self.frame_lock = threading.RLock()
        self.mode = mode

        self.tag_id = None
        self.tag_side_length = None

        self.autobed_sub = None

        # The homogeneous transform from the reference point of interest (e.g. the center of the head, the base of the
        # bed model).
        self.reference_B_ar = np.eye(4)

        if self.mode == 'head':
            self.config_head_AR_detector()
        elif self.mode == 'autobed':
            self.bed_state_z = 0.
            self.bed_state_head_theta = 0.
            self.bed_state_leg_theta = 0.
            self.config_autobed_AR_detector()
        else:
            print 'I do not know what AR tag to look for... Abort!'
            return

        self.hist_size = 10
        self.pos_buf  = cb.CircularBuffer(self.hist_size, (3,))
        self.quat_buf = cb.CircularBuffer(self.hist_size, (4,))

        self.pose_pub = rospy.Publisher(''.join(['ar_tag_tracking/',self.mode, '_pose']), PoseStamped, queue_size=1, latch=True)
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.arTagCallback)

    def config_head_AR_detector(self):
        self.tag_id = 10 #9
        self.tag_side_length = 0.053 #0.033

        # This is the translational transform from reference markers to the bed origin.
        # -.445 if right side of body. .445 if left side.
        model_trans_B_ar = np.eye(4)
        model_trans_B_ar[0:3, 3] = np.array([0.1, 0.0, 0.13])
        ar_rotz_B = np.eye(4)
        ar_rotz_B [0:2, 0:2] = np.array([[-1, 0], [0, -1]])
        # If left side of bed should be np.array([[-1,0],[0,-1]])
        # If right side of bed should be np.array([[1,0],[0,1]])
        ar_rotx_B = np.eye(4)
        ar_rotx_B[1:3, 1:3] = np.array([[0, 1], [-1, 0]])
        self.reference_B_ar = np.matrix(model_trans_B_ar)*np.matrix(ar_rotz_B)*np.matrix(ar_rotx_B)

    def config_autobed_AR_detector(self):
        self.tag_id = 4  # 9

        self.autobed_sub = rospy.Subscriber('/abdout0', FloatArrayBare, self.bed_state_cb)
        self.tag_side_length = 0.133  # 0.033

        # This is the translational transform from reference markers to the bed origin.
        # -.445 if right side of body. .445 if left side.
        model_trans_B_ar = np.eye(4)
        model_trans_B_ar[0:3,3] = np.array([0.625, -.445, .275+(self.bed_state_z-9)/100])
        ar_rotz_B = np.eye(4)
        # If left side of bed should be np.array([[-1,0],[0,-1]])
        # If right side of bed should be np.array([[1,0],[0,1]])
        ar_rotx_B = np.eye(4)
        ar_rotx_B[1:3,1:3] = np.array([[0,1],[-1,0]])
        self.reference_B_ar = np.matrix(model_trans_B_ar)*np.matrix(ar_rotz_B)*np.matrix(ar_rotx_B)

    # When we are using the autobed, we probably need to know the state of the autobed. This records the current
    # state of the autobed.
    def bed_state_cb(self, data):
        self.bed_state_z = data.data[1]
        self.bed_state_head_theta = data.data[0]
        self.bed_state_leg_theta = data.data[2]

    def arTagCallback(self, msg):

        markers = msg.markers

        with self.frame_lock:
            for i in xrange(len(markers)):
                if markers[i].id == self.tag_id:
                    cur_p = np.array([markers[i].pose.pose.position.x,
                                      markers[i].pose.pose.position.y,
                                      markers[i].pose.pose.position.z])
                    cur_q = np.array([markers[i].pose.pose.orientation.x,
                                      markers[i].pose.pose.orientation.y,
                                      markers[i].pose.pose.orientation.z,
                                      markers[i].pose.pose.orientation.w])

                    frame_id = markers[i].pose.header.frame_id
                    print frame_id

                    if np.linalg.norm(cur_p) > 4.0:
                        print "Detected tag is located too far away."
                        continue

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
                        pos = positions[len(positions)/2]

                        quaternions = np.sort(quaternions, axis=0)
                        quat = quaternions[len(quaternions)/2]

                    pr2_B_ar = createBMatrix(pos, quat)

                    out_pos, out_quat = Bmat_to_pos_quat(pr2_B_ar*self.reference_B_ar.I)

                    ps = PoseStamped()
                    ps.header.frame_id = markers[i].pose.header.frame_id
                    ps.header.stamp =  markers[i].pose.header.stamp
                    ps.pose.position.x = out_pos[0]
                    ps.pose.position.y = out_pos[1]
                    ps.pose.position.z = out_pos[2]

                    ps.pose.orientation.x = out_quat[0]
                    ps.pose.orientation.y = out_quat[1]
                    ps.pose.orientation.z = out_quat[2]
                    ps.pose.orientation.w = out_quat[3]

                    self.pose_pub.publish(ps)

if __name__ == '__main__':
    rospy.init_node('find_ar_tags')

    import optparse
    p = optparse.OptionParser()
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew frame pickle files.')
    p.add_option('--virtual', '--v', action='store_true', dest='bVirtual',
                 default=False, help='Send a virtual frame.')
    p.add_option('--mode', action='store', dest='mode', default='head', type='string',
                 help='Select what AR tag to look for (e.g. head, autobed)')
    opt, args = p.parse_args()

    atd = arTagDetector(opt.mode)
    rospy.spin()
