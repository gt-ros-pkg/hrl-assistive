#!/usr/bin/env python

import rospy
import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import numpy as np
import threading

from ar_track_alvar.msg import AlvarMarkers
import geometry_msgs
from tf_conversions import posemath
import PyKDL
from hrl_lib import quaternion as qt

import hrl_common_code_darpa_m3.visualization.draw_scene as ds
from visualization_msgs.msg import Marker, MarkerArray

class arTagBundle:

    def __init__(self, n_tags, main_tag_id, tag_side_length, pos_thres, max_idx):

        
        self.n_tags = n_tags
        self.main_tag_id = main_tag_id
        self.tag_side_length = tag_side_length
        self.main_tag_frame = None
        self.max_idx = max_idx

        self.corner1 = PyKDL.Vector(-tag_side_length/2.0, -tag_side_length/2.0, 0.0)
        self.corner2 = PyKDL.Vector( tag_side_length/2.0, -tag_side_length/2.0, 0.0)
        self.corner3 = PyKDL.Vector( tag_side_length/2.0,  tag_side_length/2.0, 0.0)
        self.corner4 = PyKDL.Vector(-tag_side_length/2.0,  tag_side_length/2.0, 0.0)
        self.pos_thres = pos_thres

        self.bundle_list = [self.main_tag_id]
        self.bundle_dict = {}
        self.bundle_dict[str(self.main_tag_id)] = PyKDL.Frame.Identity()
        ## self.bundle_dict[str(self.main_tag_id)] = [self.corner1, self.corner2, self.corner3, self.corner4]

        self.z_neg90_frame = PyKDL.Frame.Identity()
        self.z_neg90_frame.M = PyKDL.Rotation.Quaternion(0.0, 0.0, -np.sqrt(0.5), np.sqrt(0.5))

        self.bundle_result = {}

        self.draw_blocks    = ds.SceneDraw("ar_track_alvar/bundle_viz", "/torso_lift_link")
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.arTagCallback)

        self.bundle_dict_lock = threading.RLock()                
        
    def draw_bundle(self, start_id=0):
        
        ## assert (self.num_blocks == len(self.block_dim)), "num_blocks is different with the number of block_dim."        
        ## self.block_lock.acquire()

        if self.main_tag_frame is None: return
        
        with self.bundle_dict_lock:                
            self.getBundleResult()

            for i, tag_id in enumerate(self.bundle_list):

                if tag_id > self.max_idx: continue

                f = self.main_tag_frame * self.bundle_dict[str(tag_id)]
                pos_x = f.p[0]
                pos_y = f.p[1]
                pos_z = f.p[2]

                quat_x = f.M.GetQuaternion()[0]
                quat_y = f.M.GetQuaternion()[1]
                quat_z = f.M.GetQuaternion()[2]
                quat_w = f.M.GetQuaternion()[3]

                scale_x = scale_y = self.tag_side_length
                scale_z = 0.005

                corner1 = f * self.corner1
                corner2 = f * self.corner2
                corner3 = f * self.corner3
                corner4 = f * self.corner4
                ## print "-----------------------"
                ## print tag_id, " : "
                ## ## print self.main_tag_frame
                ## print corner1
                ## print corner2
                ## print corner3
                ## print corner4
                ## print "-----------------------"
                
                self.draw_blocks.pub_body([pos_x, pos_y, pos_z],
                                            [quat_x, quat_y, quat_z, quat_w],
                                            [scale_x,scale_y,scale_z], 
                                            [0.0, 1.0, 0.0, 0.7], 
                                            start_id+i, 
                                            self.draw_blocks.Marker.CUBE)
                
    def arTagCallback(self, msg):

        markers = msg.markers

        main_tag_flag  = False
        
        for i in xrange(len(markers)):
            if markers[i].id > self.max_idx: continue
        
            if markers[i].id == self.main_tag_id:
                main_tag_flag = True
                self.main_tag_frame = posemath.fromMsg(markers[i].pose.pose)*self.z_neg90_frame 
                
                if self.main_tag_frame.p.Norm() > 2.0: return
                
        if main_tag_flag == False: return

        for i in xrange(len(markers)):

            if markers[i].id > self.max_idx: continue
        
            if markers[i].id != self.main_tag_id:                
                tag_id    = markers[i].id
                tag_frame = posemath.fromMsg(markers[i].pose.pose)*self.z_neg90_frame

                # position filtering
                if (self.main_tag_frame.p - tag_frame.p).Norm() > self.pos_thres : return
                
                frame_diff = self.main_tag_frame.Inverse()*tag_frame
                self.updateFrames(tag_id, frame_diff)

                
    def updateFrames(self, tag_id, frame):


        with self.bundle_dict_lock:                

            tag_flag = False
            for key in self.bundle_dict.keys():
                if key == str(tag_id):
                    tag_flag = True

                    self.bundle_dict[key].p = (self.bundle_dict[key].p + frame.p)/2.0
                    pre_quat = geometry_msgs.msg.Quaternion()
                    pre_quat.x = self.bundle_dict[key].M.GetQuaternion()[0]
                    pre_quat.y = self.bundle_dict[key].M.GetQuaternion()[1]
                    pre_quat.z = self.bundle_dict[key].M.GetQuaternion()[2]
                    pre_quat.w = self.bundle_dict[key].M.GetQuaternion()[3]

                    cur_quat = geometry_msgs.msg.Quaternion()
                    cur_quat.x = frame.M.GetQuaternion()[0]
                    cur_quat.y = frame.M.GetQuaternion()[1]
                    cur_quat.z = frame.M.GetQuaternion()[2]
                    cur_quat.w = frame.M.GetQuaternion()[3]

                    quat = qt.slerp(pre_quat, cur_quat, 0.5)
                    self.bundle_dict[key].M = PyKDL.Rotation.Quaternion(quat.x, quat.y, quat.z, quat.w)

            # New tag
            if tag_flag == False:
                self.bundle_list.append(tag_id)            
                self.bundle_dict[str(tag_id)] = frame

                print "Detected tags: ", self.bundle_list


    def getBundleResult(self):

        for i in self.bundle_list:
        
            if i > self.max_idx: continue

            frame = self.bundle_dict[str(i)]

            corner1 = frame * self.corner1
            corner2 = frame * self.corner2
            corner3 = frame * self.corner3
            corner4 = frame * self.corner4
        
            self.bundle_result[str(i)] = [corner1, corner2, corner3, corner4]

    def print_xml(self):

        print "-----------------------------------------------------------"
        print '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>'
        print '<multimarker markers="'+str(self.n_tags)+'">'

        for i in self.bundle_list:
            if i > self.max_idx: continue

            d = self.bundle_result[str(i)]

            print '    <marker index="'+str(i)+'" status="1">'
            print '        <corner x="'+str(d[0][0])+'" y="'+str(d[0][1])+'" z="'+str(d[0][2])+'" />'
            print '        <corner x="'+str(d[1][0])+'" y="'+str(d[1][1])+'" z="'+str(d[1][2])+'" />'
            print '        <corner x="'+str(d[2][0])+'" y="'+str(d[2][1])+'" z="'+str(d[2][2])+'" />'
            print '        <corner x="'+str(d[3][0])+'" y="'+str(d[3][1])+'" z="'+str(d[3][2])+'" />'
            print '    </marker>'

        print '</multimarker>'

    def save_xml(self):

        f=open('./test.xml', 'w+')

        f.write('<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n')
        f.write('<multimarker markers="'+str(self.n_tags)+'">\n')

        for i in self.bundle_list:
            if i > self.max_idx: continue

            d = self.bundle_result[str(i)]

            f.write('    <marker index="'+str(i)+'" status="1">\n')
            f.write('        <corner x="'+str(d[0][0])+'" y="'+str(d[0][1])+'" z="'+str(d[0][2])+'" />\n')
            f.write('        <corner x="'+str(d[1][0])+'" y="'+str(d[1][1])+'" z="'+str(d[1][2])+'" />\n')
            f.write('        <corner x="'+str(d[2][0])+'" y="'+str(d[2][1])+'" z="'+str(d[2][2])+'" />\n')
            f.write('        <corner x="'+str(d[3][0])+'" y="'+str(d[3][1])+'" z="'+str(d[3][2])+'" />\n')
            f.write('    </marker>\n')

        f.write('</multimarker>\n')        

        f.close()

        

if __name__ == '__main__':
    rospy.init_node('ar_tag_bundle_estimation')

    total_tags = 3
    main_tag_id = 9
    tag_side_length = 0.053 #0.033
    pos_thres = 0.2
    max_idx   = 18
    
    atb = arTagBundle(total_tags, main_tag_id, tag_side_length, pos_thres, max_idx)
    
    rate = rospy.Rate(10) # 25Hz, nominally.    
    while not rospy.is_shutdown():
        atb.draw_bundle(100)
        ## log.log_state()
        rate.sleep()

    atb.getBundleResult()        
    atb.print_xml()
    atb.save_xml()

        
