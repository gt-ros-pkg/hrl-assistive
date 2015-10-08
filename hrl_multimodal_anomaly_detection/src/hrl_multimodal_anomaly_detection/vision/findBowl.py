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

    def __init__(self, bowl_tag_id, tag_side_length, pos_thres):

        print "Start arTagBowl_cenConversion"
        self.tag_side_length = tag_side_length

        self.bowl_tag_id   = bowl_tag_id
        self.bowl_calib    = False
        self.bowl_z_offset = 0.05
        self.bowl_forward_offset = 0.075
        self.bowl_tag_flag = False

        self.bowl_cen_frame_off = None
        self.bowl_frame   = None
        self.bowl_cen_frame  = None
        self.hist_size = 10
        self.bowl_pos_buf  = cb.CircularBuffer(self.hist_size, (3,))
        self.bowl_quat_buf = cb.CircularBuffer(self.hist_size, (4,))               

        self.x_neg90_frame = PyKDL.Frame.Identity()
        self.x_neg90_frame.M = PyKDL.Rotation.Quaternion(-np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))
        self.y_90_frame = PyKDL.Frame.Identity()
        self.y_90_frame.M = PyKDL.Rotation.Quaternion(0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5))

        
        self.draw_bowl_cen_block = ds.SceneDraw("ar_track_alvar/bowl_cen_block", "/torso_lift_link")
        self.draw_bowl_cen_arrow = ds.SceneDraw("ar_track_alvar/bowl_cen_arrow", "/torso_lift_link")
        self.bowl_cen_pose_pub = rospy.Publisher("ar_track_alvar/bowl_cen_pose", PoseStamped, latch=True)
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.arTagCallback)

        self.frame_lock = threading.RLock()                
        
        
    def arTagCallback(self, msg):

        markers = msg.markers

        bowl_tag_flag  = False

        with self.frame_lock:
            for i in xrange(len(markers)):

                if markers[i].id == self.bowl_tag_id:
                    bowl_tag_flag = True
                    bowl_tag_frame = posemath.fromMsg(markers[i].pose.pose)

                    if bowl_tag_frame.p.Norm() > 2.0: 
                        print "Detected tag is located at too far location."
                        bowl_tag_flag = False
                        continue

                    cur_p = np.array([bowl_tag_frame.p[0], bowl_tag_frame.p[1], bowl_tag_frame.p[2]])
                    cur_q = np.array([bowl_tag_frame.M.GetQuaternion()[0], 
                                      bowl_tag_frame.M.GetQuaternion()[1], 
                                      bowl_tag_frame.M.GetQuaternion()[2],
                                      bowl_tag_frame.M.GetQuaternion()[3]])

                    if len(self.bowl_quat_buf) < 1:
                        self.bowl_pos_buf.append( cur_p )
                        self.bowl_quat_buf.append( cur_q )
                    else:
                        first_p = self.bowl_pos_buf[0]
                        first_q = self.bowl_quat_buf[0]

                        # check close quaternion and inverse
                        if np.dot(cur_q, first_q) < 0.0:
                            cur_q *= -1.0

                        self.bowl_pos_buf.append( cur_p )
                        self.bowl_quat_buf.append( cur_q )
                            
                        
                    positions  = self.bowl_pos_buf.get_array()
                    quaternions  = self.bowl_quat_buf.get_array() 

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
                        
                    bowl_tag_frame.p[0] = p[0]
                    bowl_tag_frame.p[1] = p[1]
                    bowl_tag_frame.p[2] = p[2]                    
                    bowl_tag_frame.M = PyKDL.Rotation.Quaternion(q[0], q[1], q[2], q[3])
                    
                    self.bowl_frame = bowl_tag_frame

                    if self.bowl_calib == False:
                        self.updateBowlcenFrames(bowl_tag_frame)
                    else:
                        self.pubBowlcenPose()

                        
        self.draw_bowl_cen(200)
                        
        if bowl_tag_flag: self.bowl_tag_flag = True
                    

    def updateBowlcenFrames(self, bowl_frame):

        bowl_cen_frame = copy.deepcopy(bowl_frame)
        
        ## Rotation        
        rot = bowl_cen_frame.M

        ## bowl_z = np.array([rot.UnitX()[0], rot.UnitX()[1], rot.UnitX()[2]])
        tx = PyKDL.Vector(1.0, 0.0, 0.0)
        ty = PyKDL.Vector(0.0, 1.0, 0.0)

        # Projection to xy plane
        px = PyKDL.dot(tx, rot.UnitZ())
        py = PyKDL.dot(ty, rot.UnitZ())

        bowl_cen_y = rot.UnitY()
        bowl_cen_z = PyKDL.Vector(px, py, 0.0)
        bowl_cen_z.Normalize()
        bowl_cen_x = bowl_cen_y * bowl_cen_z 
        bowl_cen_y = bowl_cen_z * bowl_cen_x
        
        bowl_cen_rot     = PyKDL.Rotation(bowl_cen_x, bowl_cen_y, bowl_cen_z)
        bowl_cen_frame.M = bowl_cen_rot
        
        ## Position
        bowl_cen_frame.p[2] -= self.bowl_z_offset
        bowl_cen_frame.p += bowl_cen_z * self.bowl_forward_offset

        if bowl_cen_y[2] > bowl_cen_x[2]:
            # -90 x axis rotation
            bowl_cen_frame = bowl_cen_frame * self.x_neg90_frame 
        else:
            # 90 y axis rotation
            bowl_cen_frame = bowl_cen_frame * self.y_90_frame         

        bowl_cen_frame_off = bowl_frame.Inverse()*bowl_cen_frame
        
        if self.bowl_cen_frame_off == None:            
            self.bowl_cen_frame_off = bowl_cen_frame_off
        else:
            self.bowl_cen_frame_off.p = (self.bowl_cen_frame_off.p + bowl_cen_frame_off.p)/2.0

            pre_quat = geometry_msgs.msg.Quaternion()
            pre_quat.x = self.bowl_cen_frame_off.M.GetQuaternion()[0]
            pre_quat.y = self.bowl_cen_frame_off.M.GetQuaternion()[1]
            pre_quat.z = self.bowl_cen_frame_off.M.GetQuaternion()[2]
            pre_quat.w = self.bowl_cen_frame_off.M.GetQuaternion()[3]
            
            cur_quat = geometry_msgs.msg.Quaternion()
            cur_quat.x = bowl_cen_frame_off.M.GetQuaternion()[0]
            cur_quat.y = bowl_cen_frame_off.M.GetQuaternion()[1]
            cur_quat.z = bowl_cen_frame_off.M.GetQuaternion()[2]
            cur_quat.w = bowl_cen_frame_off.M.GetQuaternion()[3]
            
            quat = qt.slerp(pre_quat, cur_quat, 0.5)
            self.bowl_cen_frame_off.M = PyKDL.Rotation.Quaternion(quat.x, quat.y, quat.z, quat.w)
            
        



    def draw_bowl_cen(self, start_id=0):
        
        ## assert (self.num_blocks == len(self.block_dim)), "num_blocks is different with the number of block_dim."        
        ## self.block_lock.acquire()

        ## if self.bowl_calib

        with self.frame_lock:

            if self.bowl_cen_frame_off is None or self.bowl_frame is None: return
            f = self.bowl_frame * self.bowl_cen_frame_off
            pos_x = f.p[0]
            pos_y = f.p[1]
            pos_z = f.p[2]

            quat_x = f.M.GetQuaternion()[0]
            quat_y = f.M.GetQuaternion()[1]
            quat_z = f.M.GetQuaternion()[2]
            quat_w = f.M.GetQuaternion()[3]

            scale_x = scale_y = self.tag_side_length
            scale_z = 0.005

            self.draw_bowl_cen_block.pub_body([pos_x, pos_y, pos_z],
                                           [quat_x, quat_y, quat_z, quat_w],
                                           [scale_x,scale_y,scale_z], 
                                           [0.0, 1.0, 0.0, 0.7], 
                                           start_id+0, 
                                           self.draw_bowl_cen_block.Marker.CUBE)

 
            ## f = f #* self.y_neg90_frame
            pos1 = np.array([[f.p[0], f.p[1], f.p[2]]]).T
            z_axis = f.M.UnitZ() * 0.1            
            pos2 = np.array([[f.p[0] + z_axis[0], f.p[1] + z_axis[1], f.p[2] + z_axis[2]]]).T
            self.draw_bowl_cen_arrow.pub_arrow(pos1, pos2, [0.0, 1.0, 0.0, 0.7], str(0.0))

    def getCalibration(self, filename='bowl_frame.pkl'):
        if os.path.isfile(filename) == False: return False
        
        d = ut.load_pickle(filename)        
        self.bowl_cen_frame_off = d['frame']
        
        self.bowl_calib = True
        print "------------------------------------"
        print "Calibration complete! - bowl_cen offset"
        print "------------------------------------"
        print "P: ", self.bowl_cen_frame_off.p
        print "M: ", self.bowl_cen_frame_off.M
        print "------------------------------------"
        return True
            
    def setCalibration(self, filename='bowl_frame.pkl'):
        self.bowl_calib = True
        print "------------------------------------"
        print "Calibration complete! - bowl_cen offset"
        print "------------------------------------"
        print "P: ", self.bowl_cen_frame_off.p
        print "M: ", self.bowl_cen_frame_off.M
        print "------------------------------------"
        d = {}
        d['frame'] = self.bowl_cen_frame_off        
        ut.save_pickle(d,filename)        
       
    def pubBowlcenPose(self):

        f = self.bowl_frame * self.bowl_cen_frame_off
        
        ps = PoseStamped()
        ps.header.frame_id = 'ar_bowl_cen'
        ps.header.stamp = rospy.Time.now()
        ps.pose.position.x = f.p[0]
        ps.pose.position.y = f.p[1]
        ps.pose.position.z = f.p[2]
        
        ps.pose.orientation.x = f.M.GetQuaternion()[0]
        ps.pose.orientation.y = f.M.GetQuaternion()[1]
        ps.pose.orientation.z = f.M.GetQuaternion()[2]
        ps.pose.orientation.w = f.M.GetQuaternion()[3]

        self.bowl_cen_pose_pub.publish(ps)
            

if __name__ == '__main__':
    rospy.init_node('ar_tag_bowl_cen_estimation')

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

    save_file = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/params/ar_tag/bowl_offsetframe.pkl' 
        
    atd = arTagDetector(tag_id, tag_side_length, pos_thres)

    if opt.bRenew == False:
        if atd.getCalibration(save_file) == False: opt.bRenew=True
            
    rate = rospy.Rate(10) # 25Hz, nominally.    
    while not rospy.is_shutdown():

        ## ret = input("Is bowl tag fine? ")
        if atd.bowl_calib == False and opt.bRenew == True:
            ret = ut.get_keystroke('Is bowl tag fine? (y: yes, n: no)')
            if ret == 'y': atd.setCalibration(save_file)
            
        
        rate.sleep()


        
        
