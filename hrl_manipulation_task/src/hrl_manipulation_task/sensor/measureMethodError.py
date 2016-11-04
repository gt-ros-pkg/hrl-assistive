#!/usr/bin/env python

import rospy, roslib
import numpy as np
import os, threading, copy

import PyKDL
import time

from tf_conversions import posemath
from hrl_lib import quaternion as qt
import hrl_lib.util as ut
import hrl_lib.circular_buffer as cb
import tf.transformations as tft

from ar_track_alvar_msgs.msg import AlvarMarkers
import geometry_msgs
from geometry_msgs.msg import PoseStamped, PointStamped, PoseArray

QUEUE_SIZE = 10

class arTagDetector:

    def __init__(self, head_tag_id, mouth_tag_id, tag_side_length, pos_thres):

        print "Start arTagMouthConversion"
        self.tag_side_length = tag_side_length

        self.head_tag_id   = head_tag_id
        self.head_calib    = False
        self.head_z_offset = 0 #0.12 #0.13 #12 #0.15

        self.mouth_tag_id   = mouth_tag_id
        self.mouth_calib    = False
        self.mouth_z_offset = 0 #0.12 #0.13 #12 #0.15

        self.mouth_frame_off = None
        self.head_frame   = None
        self.mouth_frame  = None
        self.mouth_offset = None

        self.hist_size = 10
        self.head_pos_buf  = cb.CircularBuffer(self.hist_size, (3,))
        self.head_quat_buf = cb.CircularBuffer(self.hist_size, (4,))               
        self.mouth_pos_buf  = cb.CircularBuffer(self.hist_size, (3,))
        self.mouth_quat_buf = cb.CircularBuffer(self.hist_size, (4,))
        
        self.head_pose_pub = rospy.Publisher("ar_track_alvar/head_pose", PoseStamped, \
                                              queue_size=QUEUE_SIZE, latch=True)
        self.mouth_pose_pub = rospy.Publisher("ar_track_alvar/mouth_pose", PoseStamped, \
                                              queue_size=QUEUE_SIZE, latch=True)
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.arTagCallback)
        rospy.Subscriber("/ar_mouth_pose_marker", AlvarMarkers, self.arTagCallbackMouth)

        self.frame_lock = threading.RLock()                
        self.mouth_frame_lock = threading.RLock()
        
        
    def arTagCallback(self, msg):
        markers = msg.markers
        with self.frame_lock:
            for i in xrange(len(markers)):

                if markers[i].id == self.head_tag_id:
                    head_tag_frame = posemath.fromMsg(markers[i].pose.pose)
                    curr_time = markers[i].header.stamp
                    if head_tag_frame.p.Norm() > 2.0: 
                        print "Detected tag is located at too far location."
                        continue

                    cur_p = np.array([head_tag_frame.p[0], head_tag_frame.p[1], head_tag_frame.p[2]])
                    cur_q = np.array([head_tag_frame.M.GetQuaternion()[0], 
                                      head_tag_frame.M.GetQuaternion()[1], 
                                      head_tag_frame.M.GetQuaternion()[2],
                                      head_tag_frame.M.GetQuaternion()[3]])

                    if len(self.head_quat_buf) < 1:
                        self.head_pos_buf.append( cur_p )
                        self.head_quat_buf.append( cur_q )
                    else:
                        first_p = self.head_pos_buf[0]
                        first_q = self.head_quat_buf[0]

                        # check close quaternion and inverse
                        if np.dot(cur_q, first_q) < 0.0:
                            cur_q *= -1.0
                        head_tag_frame2 = posemath.fromMsg(markers[i].pose.pose)
                        head_tag_frame2.p[0] = cur_p[0]
                        head_tag_frame2.p[1] = cur_p[1]
                        head_tag_frame2.p[2] = cur_p[2]
                        head_tag_frame2.M = PyKDL.Rotation.Quaternion(cur_q[0], cur_q[1], cur_q[2], cur_q[3])
                        head_tag_frame2 = self.adjustFrame(head_tag_frame2)
                        cur_q = np.array(head_tag_frame2.M.GetQuaternion())
                        if np.dot(cur_q, first_q) < 0.0:
                            cur_q *= -1.0

                        self.head_pos_buf.append( cur_p )
                        self.head_quat_buf.append( cur_q )
                            
                        
                    positions  = self.head_pos_buf.get_array()
                    quaternions  = self.head_quat_buf.get_array() 

                    p = None
                    q = None
                    if False:
                        # Moving average
                        p = np.sum(positions, axis=0)                    
                        p /= float(len(positions))
                    
                        #q = np.sum(quaternions, axis=0)
                        #q /= float(len(quaternions))
                        q = qt.quat_avg(np.array(quaternions))
                    else:
                        # median
                        positions = np.sort(positions, axis=0)
                        p = positions[len(positions)/2]

                        quaternions = np.sort(quaternions, axis=0)
                        q = quaternions[len(quaternions)/2]
                        print len(quaternions)/2
                        
                    head_tag_frame.p[0] = p[0]
                    head_tag_frame.p[1] = p[1]
                    head_tag_frame.p[2] = p[2]        
                    print abs(1.0-np.linalg.norm(q))
                    q=qt.quat_normal(q)
                    head_tag_frame.M = PyKDL.Rotation.Quaternion(q[0], q[1], q[2], q[3])

                    self.head_frame = head_tag_frame#self.adjustFrame(head_tag_frame)
                    if self.mouth_offset is None:
                        self.pubPose(self.head_frame, self.head_pose_pub, curr_time=curr_time)
                        if len(self.mouth_pos_buf) == self.hist_size:
                            self.mouth_offset = self.head_frame.Inverse() * self.mouth_frame
                        else:
                            print len(self.mouth_pos_buf)
                    else:
                        self.pubPose(self.head_frame*self.mouth_offset, self.head_pose_pub)

    def arTagCallbackMouth(self, msg):
        markers = msg.markers
        with self.mouth_frame_lock:
            for i in xrange(len(markers)):

                if markers[i].id in self.mouth_tag_id:
                    mouth_tag_frame = posemath.fromMsg(markers[i].pose.pose)

                    if mouth_tag_frame.p.Norm() > 2.0: 
                        print "Detected tag is located at too far location."
                        continue

                    cur_p = np.array([mouth_tag_frame.p[0], mouth_tag_frame.p[1], mouth_tag_frame.p[2]])
                    cur_q = np.array([mouth_tag_frame.M.GetQuaternion()[0], 
                                      mouth_tag_frame.M.GetQuaternion()[1], 
                                      mouth_tag_frame.M.GetQuaternion()[2],
                                      mouth_tag_frame.M.GetQuaternion()[3]])

                    if len(self.mouth_quat_buf) < 1:
                        self.mouth_pos_buf.append( cur_p )
                        self.mouth_quat_buf.append( cur_q )
                    else:
                        first_p = self.mouth_pos_buf[0]
                        first_q = self.mouth_quat_buf[0]

                        # check close quaternion and inverse
                        if np.dot(cur_q, first_q) < 0.0:
                            cur_q *= -1.0
                        mouth_tag_frame2 = posemath.fromMsg(markers[i].pose.pose)
                        mouth_tag_frame2.p[0] = cur_p[0]
                        mouth_tag_frame2.p[1] = cur_p[1]
                        mouth_tag_frame2.p[2] = cur_p[2]
                        mouth_tag_frame2.M = PyKDL.Rotation.Quaternion(cur_q[0], cur_q[1], cur_q[2], cur_q[3])
                        mouth_tag_frame2 = self.adjustFrame(mouth_tag_frame2)
                        cur_q = np.array(mouth_tag_frame2.M.GetQuaternion())
                        if np.dot(cur_q, first_q) < 0.0:
                            cur_q *= -1.0
                        self.mouth_pos_buf.append( cur_p )
                        self.mouth_quat_buf.append( cur_q )
                            
                        
                    positions  = self.mouth_pos_buf.get_array()
                    quaternions  = self.mouth_quat_buf.get_array() 

                    p = None
                    q = None
                    if False:
                        # Moving average
                        p = np.sum(positions, axis=0)                    
                        p /= float(len(positions))
                    
                        #q = np.sum(quaternions, axis=0)
                        #q /= float(len(quaternions))
                        q=qt.quat_avg(np.array(quaternions))
                    else:
                        # median
                        positions = np.sort(positions, axis=0)
                        p = positions[len(positions)/2]

                        quaternions = np.sort(quaternions, axis=0)
                        q = quaternions[len(quaternions)/2]
                        
                    mouth_tag_frame.p[0] = p[0]
                    mouth_tag_frame.p[1] = p[1]
                    mouth_tag_frame.p[2] = p[2]                    
                    q=qt.quat_normal(q)
                    mouth_tag_frame.M = PyKDL.Rotation.Quaternion(q[0], q[1], q[2], q[3])
                    print np.linalg.norm(mouth_tag_frame.M.GetQuaternion())
                    self.mouth_frame = mouth_tag_frame#self.adjustFrame(mouth_tag_frame)
                    self.pubPose(self.mouth_frame, self.mouth_pose_pub)

    def adjustFrame(self, f):
        f.M.DoRotX(np.pi)
        quaternion = f.M.GetQuaternion()
        print np.linalg.norm(quaternion)
        rot_matrix=tft.quaternion_matrix(quaternion)
        temp_rot_matrix=tft.quaternion_matrix(quaternion)
        x_axis = 2#-1
        y_axis = -1
        x_max = -1
        y_max = -1
        #for i in xrange(0, 3):
        #    if rot_matrix[0, i] > x_max:
        #        x_max = rot_matrix[0, i]
        #        x_axis = i
        for i in xrange(0, 3):
            if abs(rot_matrix[1, i]) > y_max and i != x_axis:
                y_max = abs(rot_matrix[1, i])
                y_axis = i
        rot_matrix[:, 2] = temp_rot_matrix[:, x_axis]
        if temp_rot_matrix[1, y_axis] < 0:
            rot_matrix[:, 1] =  -1.0 * temp_rot_matrix[:, y_axis]
        else:
            rot_matrix[:, 1] = temp_rot_matrix[:, y_axis]
        rot_matrix[:-1, 0] = np.cross(rot_matrix[:-1,1],rot_matrix[:-1, 2])
        rot_matrix[:-1, 0] = rot_matrix[:-1, 0] / np.linalg.norm(rot_matrix[:-1, 0]) 
        new_q = tft.quaternion_from_matrix(rot_matrix)
        new_rot = PyKDL.Rotation.Quaternion(new_q[0], new_q[1], new_q[2], new_q[3])
        new_pos = PyKDL.Vector(f.p[0], f.p[1], f.p[2]-0.1)
        new_frame = PyKDL.Frame(new_rot, new_pos)
        return new_frame

                                            
    def getCalibration(self, filename='mouth_frame.pkl'):
        if os.path.isfile(filename) == False: return False

        d = ut.load_pickle(filename)        
        print d.keys()
        self.mouth_frame_off = d['frame']
        
        self.head_calib = True
        print "------------------------------------"
        print "Calibration complete! - mouth offset"
        print "------------------------------------"
        print "P: ", self.mouth_frame_off.p
        print "M: ", self.mouth_frame_off.M
        print "------------------------------------"
        return True

        
    def setCalibration(self, filename='mouth_frame.pkl'):
        self.head_calib = True
        print "------------------------------------"
        print "Calibration complete! - mouth offset"
        print "------------------------------------"
        print "P: ", self.mouth_frame_off.p
        print "M: ", self.mouth_frame_off.M
        print "------------------------------------"

        d = {}
        d['frame'] = self.mouth_frame_off        
        ut.save_pickle(d,filename)        
        

    def pubPose(self, f, publisher, curr_time=None, offset=None, set_angles=None):

        #f = self.head_frame
        
        ps = PoseStamped()
        ps.header.frame_id = 'torso_lift_link'
        if curr_time is None:
            ps.header.stamp = rospy.Time.now()
        else:
            ps.header.stamp = curr_time
        ps.pose.position.x = f.p[0]
        ps.pose.position.y = f.p[1]
        ps.pose.position.z = f.p[2]
        
        ps.pose.orientation.x = f.M.GetQuaternion()[0]
        ps.pose.orientation.y = f.M.GetQuaternion()[1]
        ps.pose.orientation.z = f.M.GetQuaternion()[2]
        ps.pose.orientation.w = f.M.GetQuaternion()[3]
                
        publisher.publish(ps)
            

    def pubVirtualMouthPose(self):

        f = PyKDL.Frame.Identity()
        f.p = PyKDL.Vector(0.85, 0.4, 0.0)
        f.M = PyKDL.Rotation.Quaternion(0,0,0,1)
        f.M.DoRotX(np.pi/2.0)
        f.M.DoRotZ(np.pi/2.0)
        f.M.DoRotX(np.pi)        
        
        # frame pub --------------------------------------
        ps = PoseStamped()
        ps.header.frame_id = 'torso_lift_link'
        ps.header.stamp = rospy.Time.now()
        ps.pose.position.x = f.p[0]
        ps.pose.position.y = f.p[1]
        ps.pose.position.z = f.p[2]
        
        ps.pose.orientation.x = f.M.GetQuaternion()[0]
        ps.pose.orientation.y = f.M.GetQuaternion()[1]
        ps.pose.orientation.z = f.M.GetQuaternion()[2]
        ps.pose.orientation.w = f.M.GetQuaternion()[3]

        self.mouth_pose_pub.publish(ps)


        
if __name__ == '__main__':
    rospy.init_node('ar_tag_mouth_estimation')

    import optparse
    p = optparse.OptionParser()
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew frame pickle files.')
    p.add_option('--virtual', '--v', action='store_true', dest='bVirtual',
                 default=False, help='Send a vitual frame.')
    opt, args = p.parse_args()
    
    total_tags = 1
    ## tag_id = 10 #9
    ## tag_side_length = 0.068 #0.053 #0.033
    tag_id = 26
    mouth_tag_id = [3, 75,71]#14
    tag_side_length = 0.084 #0.053 #0.033
    pos_thres = 0.2
    max_idx   = 18

    # Note need to use relative path
    save_file = os.path.expanduser('~')+'/catkin_ws/src/hrl-assistive/hrl_manipulation_task/params/ar_tag/mouth_offsetframe.pkl' 
    
    atd = arTagDetector(tag_id, mouth_tag_id, tag_side_length, pos_thres)

    if opt.bRenew == False:
        if atd.getCalibration(save_file) == False: opt.bRenew=True
    
    rate = rospy.Rate(10) # 25Hz, nominally.    
    while not rospy.is_shutdown():

        if opt.bVirtual:
            atd.pubVirtualMouthPose()
            continue
        
        ## ret = input("Is head tag fine? ")
        if atd.head_calib == False and opt.bRenew == True:
            ret = ut.get_keystroke('Is head tag fine? (y: yes, n: no)')
            if ret == 'y': atd.setCalibration(save_file)
            
        
        rate.sleep()


        
        
