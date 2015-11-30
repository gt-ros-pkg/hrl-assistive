#!/usr/bin/env python

import rospy, roslib
import numpy as np
import os, threading, copy

import PyKDL

from tf_conversions import posemath
from hrl_lib import quaternion as qt
import hrl_lib.util as ut
import hrl_lib.circular_buffer as cb

from ar_track_alvar.msg import AlvarMarkers
import geometry_msgs
from geometry_msgs.msg import PoseStamped, PointStamped, PoseArray


class arTagDetector:

    def __init__(self, bed_tag_id, tag_side_length, pos_thres):

        print "Start arTagBedConversion"
        self.tag_side_length = tag_side_length

        self.bed_tag_id   = bed_tag_id
        self.bed_calib    = False
        self.bed_z_offset = 0.13 #12 #0.15
        self.bed_frame   = None

        self.hist_size = 10
        self.bed_pos_buf  = cb.CircularBuffer(self.hist_size, (3,))
        self.bed_quat_buf = cb.CircularBuffer(self.hist_size, (4,))               
        
        self.bed_pose_pub = rospy.Publisher("ar_track_alvar/bed_pose", PoseStamped, latch=True)
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.arTagCallback)

        self.frame_lock = threading.RLock()                
        
        
    def arTagCallback(self, msg):

        markers = msg.markers

        with self.frame_lock:
            for i in xrange(len(markers)):

                if markers[i].id == self.bed_tag_id:
                    bed_tag_frame = posemath.fromMsg(markers[i].pose.pose)

                    if bed_tag_frame.p.Norm() > 2.0: 
                        print "Detected tag is located at too far location."
                        continue

                    cur_p = np.array([bed_tag_frame.p[0], bed_tag_frame.p[1], bed_tag_frame.p[2]])
                    cur_q = np.array([bed_tag_frame.M.GetQuaternion()[0], 
                                      bed_tag_frame.M.GetQuaternion()[1], 
                                      bed_tag_frame.M.GetQuaternion()[2],
                                      bed_tag_frame.M.GetQuaternion()[3]])

                    if len(self.bed_quat_buf) < 1:
                        self.bed_pos_buf.append( cur_p )
                        self.bed_quat_buf.append( cur_q )
                    else:
                        first_p = self.bed_pos_buf[0]
                        first_q = self.bed_quat_buf[0]

                        # check close quaternion and inverse
                        if np.dot(cur_q, first_q) < 0.0:
                            cur_q *= -1.0

                        self.bed_pos_buf.append( cur_p )
                        self.bed_quat_buf.append( cur_q )
                            
                        
                    positions  = self.bed_pos_buf.get_array()
                    quaternions  = self.bed_quat_buf.get_array() 

                    p = None
                    q = None
                    if False:
                        # Moving average
                        p = np.sum(positions, axis=0)                    
                        p /= float(len(positions))
                    
                        q = np.sum(quaternions, axis=0)
                        q /= float(len(quaternions))
                    else:
                        # median
                        positions = np.sort(positions, axis=0)
                        p = positions[len(positions)/2]

                        quaternions = np.sort(quaternions, axis=0)
                        q = quaternions[len(quaternions)/2]
                        
                    bed_tag_frame.p[0] = p[0]
                    bed_tag_frame.p[1] = p[1]
                    bed_tag_frame.p[2] = p[2]                    
                    bed_tag_frame.M = PyKDL.Rotation.Quaternion(q[0], q[1], q[2], q[3])
                    
                    self.bed_frame = bed_tag_frame

                    self.pubMouthPose()

                        
        if bed_tag_flag: self.bed_tag_flag = True
                    
                    
   
    def pubBedPose(self):

        f = self.bed_frame 
        f.M.DoRotX(np.pi)        
        
        ps = PoseStamped()
        ps.header.frame_id = 'camera_link'
        ps.header.stamp = rospy.Time.now()
        ps.pose.position.x = f.p[0]
        ps.pose.position.y = f.p[1]
        ps.pose.position.z = f.p[2]
        
        ps.pose.orientation.x = f.M.GetQuaternion()[0]
        ps.pose.orientation.y = f.M.GetQuaternion()[1]
        ps.pose.orientation.z = f.M.GetQuaternion()[2]
        ps.pose.orientation.w = f.M.GetQuaternion()[3]

        self.bed_pose_pub.publish(ps)
            

    def pubVirtualBedPose(self):

        f = PyKDL.Frame.Identity()
        f.p = PyKDL.Vector(0.85, 0.4, 0.0)
        f.M = PyKDL.Rotation.Quaternion(0,0,0,1)
        f.M.DoRotX(np.pi/2.0)
        f.M.DoRotZ(np.pi/2.0)
        f.M.DoRotX(np.pi)        
        
        # frame pub --------------------------------------
        ps = PoseStamped()
        ps.header.frame_id = 'camera_link'
        ps.header.stamp = rospy.Time.now()
        ps.pose.position.x = f.p[0]
        ps.pose.position.y = f.p[1]
        ps.pose.position.z = f.p[2]
        
        ps.pose.orientation.x = f.M.GetQuaternion()[0]
        ps.pose.orientation.y = f.M.GetQuaternion()[1]
        ps.pose.orientation.z = f.M.GetQuaternion()[2]
        ps.pose.orientation.w = f.M.GetQuaternion()[3]

        self.bed_pose_pub.publish(ps)


        
if __name__ == '__main__':
    rospy.init_node('ar_tag_mouth_estimation')

    import optparse
    p = optparse.OptionParser()
    p.add_option('--virtual', '--v', action='store_true', dest='bVirtual',
                 default=False, help='Send a vitual frame.')
    opt, args = p.parse_args()
    
    total_tags = 1
    tag_id = 10 #9
    tag_side_length = 0.053 #0.033
    pos_thres = 0.2
    max_idx   = 18

    atd = arTagDetector(tag_id, tag_side_length, pos_thres)

    rate = rospy.Rate(10) # 25Hz, nominally.    
    while not rospy.is_shutdown():

        if opt.bVirtual:
            atd.pubVirtualMouthPose()
            continue
        
        rate.sleep()


        
        
