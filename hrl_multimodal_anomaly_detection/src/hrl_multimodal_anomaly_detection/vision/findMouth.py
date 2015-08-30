#!/usr/bin/env python

import rospy
import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import numpy as np
import os, threading, copy

import PyKDL

from tf_conversions import posemath
from hrl_lib import quaternion as qt
import hrl_lib.util as ut
import hrl_lib.circular_buffer as cb

import hrl_common_code_darpa_m3.visualization.draw_scene as ds
from ar_track_alvar.msg import AlvarMarkers
import geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, PointStamped, PoseArray


class arTagDetector:

    def __init__(self, head_tag_id, tag_side_length, pos_thres):

        print "Start arTagMouthConversion"
        self.tag_side_length = tag_side_length

        self.head_tag_id   = head_tag_id
        self.head_calib    = False
        self.head_z_offset = 0.12 #12 #0.15
        self.head_tag_flag = False

        self.mouth_frame_off = None
        self.head_frame   = None
        self.mouth_frame  = None

        self.hist_size = 10
        self.head_pos_buf  = cb.CircularBuffer(self.hist_size, (3,))
        self.head_quat_buf = cb.CircularBuffer(self.hist_size, (4,))               
        
        self.draw_mouth_block = ds.SceneDraw("ar_track_alvar/mouth_block", "/torso_lift_link")
        self.draw_mouth_arrow = ds.SceneDraw("ar_track_alvar/mouth_arrow", "/torso_lift_link")
        self.mouth_pose_pub = rospy.Publisher("ar_track_alvar/mouth_pose", PoseStamped, latch=True)
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.arTagCallback)

        self.frame_lock = threading.RLock()                
        
        
    def arTagCallback(self, msg):

        markers = msg.markers

        head_tag_flag  = False

        with self.frame_lock:
            for i in xrange(len(markers)):

                if markers[i].id == self.head_tag_id:
                    head_tag_flag = True
                    head_tag_frame = posemath.fromMsg(markers[i].pose.pose)

                    if head_tag_frame.p.Norm() > 2.0: 
                        print "Detected tag is located at too far location."
                        head_tag_flag = False
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
                    
                        q = np.sum(quaternions, axis=0)
                        q /= float(len(quaternions))
                    else:
                        # median
                        positions = np.sort(positions, axis=0)
                        p = positions[len(positions)/2]

                        quaternions = np.sort(quaternions, axis=0)
                        q = quaternions[len(quaternions)/2]
                        
                    head_tag_frame.p[0] = p[0]
                    head_tag_frame.p[1] = p[1]
                    head_tag_frame.p[2] = p[2]                    
                    head_tag_frame.M = PyKDL.Rotation.Quaternion(q[0], q[1], q[2], q[3])
                    
                    self.head_frame = head_tag_frame

                    if self.head_calib == False:
                        self.updateMouthFrames(head_tag_frame)
                    else:
                        self.pubMouthPose()

                        
        self.draw_mouth(100)
                        
        if head_tag_flag: self.head_tag_flag = True
                    

    def updateMouthFrames(self, head_frame):

        mouth_frame = copy.deepcopy(head_frame)
        
        ## Position
        mouth_frame.p[2] -= self.head_z_offset

        ## Rotation        
        rot = mouth_frame.M

        ## head_z = np.array([rot.UnitX()[0], rot.UnitX()[1], rot.UnitX()[2]])
        tx = PyKDL.Vector(1.0, 0.0, 0.0)
        ty = PyKDL.Vector(0.0, 1.0, 0.0)

        # Projection to xy plane
        px = PyKDL.dot(tx, rot.UnitZ())
        py = PyKDL.dot(ty, rot.UnitZ())

        mouth_y = rot.UnitY()
        mouth_z = PyKDL.Vector(px, py, 0.0)
        mouth_z.Normalize()
        mouth_x = mouth_y * mouth_z
        mouth_y = mouth_z * mouth_x
        
        mouth_rot     = PyKDL.Rotation(mouth_x, mouth_y, mouth_z)
        mouth_frame.M = mouth_rot

        mouth_frame_off = head_frame.Inverse()*mouth_frame
        
        if self.mouth_frame_off == None:            
            self.mouth_frame_off = mouth_frame_off
        else:
            self.mouth_frame_off.p = (self.mouth_frame_off.p + mouth_frame_off.p)/2.0

            pre_quat = geometry_msgs.msg.Quaternion()
            pre_quat.x = self.mouth_frame_off.M.GetQuaternion()[0]
            pre_quat.y = self.mouth_frame_off.M.GetQuaternion()[1]
            pre_quat.z = self.mouth_frame_off.M.GetQuaternion()[2]
            pre_quat.w = self.mouth_frame_off.M.GetQuaternion()[3]
            
            cur_quat = geometry_msgs.msg.Quaternion()
            cur_quat.x = mouth_frame_off.M.GetQuaternion()[0]
            cur_quat.y = mouth_frame_off.M.GetQuaternion()[1]
            cur_quat.z = mouth_frame_off.M.GetQuaternion()[2]
            cur_quat.w = mouth_frame_off.M.GetQuaternion()[3]
            
            quat = qt.slerp(pre_quat, cur_quat, 0.5)
            self.mouth_frame_off.M = PyKDL.Rotation.Quaternion(quat.x, quat.y, quat.z, quat.w)
            
        



    def draw_mouth(self, start_id=0):
        
        ## assert (self.num_blocks == len(self.block_dim)), "num_blocks is different with the number of block_dim."        
        ## self.block_lock.acquire()

        ## if self.head_calib

        with self.frame_lock:

            if self.mouth_frame_off is None or self.head_frame is None: return
            f = self.head_frame * self.mouth_frame_off
            pos_x = f.p[0]
            pos_y = f.p[1]
            pos_z = f.p[2]

            quat_x = f.M.GetQuaternion()[0]
            quat_y = f.M.GetQuaternion()[1]
            quat_z = f.M.GetQuaternion()[2]
            quat_w = f.M.GetQuaternion()[3]

            scale_x = scale_y = self.tag_side_length
            scale_z = 0.005

            self.draw_mouth_block.pub_body([pos_x, pos_y, pos_z],
                                           [quat_x, quat_y, quat_z, quat_w],
                                           [scale_x,scale_y,scale_z], 
                                           [0.0, 1.0, 0.0, 0.7], 
                                           start_id+0, 
                                           self.draw_mouth_block.Marker.CUBE)

 
            ## f = f #* self.y_neg90_frame
            pos1 = np.array([[f.p[0], f.p[1], f.p[2]]]).T
            z_axis = f.M.UnitZ() * 0.1            
            pos2 = np.array([[f.p[0] + z_axis[0], f.p[1] + z_axis[1], f.p[2] + z_axis[2]]]).T
            self.draw_mouth_arrow.pub_arrow(pos1, pos2, [0.0, 1.0, 0.0, 0.7], str(0.0))


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
        

    def pubMouthPose(self):

        f = self.head_frame * self.mouth_frame_off
        
        ps = PoseStamped()
        ps.header.frame_id = 'ar_mouth'
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
    opt, args = p.parse_args()
    
    total_tags = 1
    tag_id = 9
    tag_side_length = 0.053 #0.033
    pos_thres = 0.2
    max_idx   = 18

    save_file = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/params/ar_tag/mouth_offsetframe.pkl' 
    
    atd = arTagDetector(tag_id, tag_side_length, pos_thres)

    if opt.bRenew == False:
        if atd.getCalibration(save_file) == False: opt.bRenew=True
    
    rate = rospy.Rate(10) # 25Hz, nominally.    
    while not rospy.is_shutdown():

        ## ret = input("Is head tag fine? ")
        if atd.head_calib == False and opt.bRenew == True:
            ret = ut.get_keystroke('Is head tag fine? (y: yes, n: no)')
            if ret == 'y': atd.setCalibration(save_file)
            
        
        rate.sleep()


        
        
