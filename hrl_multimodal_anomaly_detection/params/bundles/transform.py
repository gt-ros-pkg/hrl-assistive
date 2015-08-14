#!/usr/bin/env python

import rospy
import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import numpy as np

from ar_track_alvar.msg import AlvarMarkers
import geometry_msgs
from tf_conversions import posemath

class arTagBundle:

    def __init__(self, n_tags, main_tag_id, corner1, corner2, corner3, corner4, pos_thres):

        self.n_tags = n_tags
        self.main_tag_id = main_tag_id
        self.corner1 = corner1
        self.corner2 = corner2
        self.corner3 = corner3
        self.corner4 = corner4
        self.pos_thres = pos_thres

        self.bundle_list = [self.main_tag_id]
        self.bundle_dict = {}
        self.bundle_dict[str(self.main_tag_id)] = [self.corner1, self.corner2, self.corner3, self.corner4]
        
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.arTagCallback)


    def arTagCallback(self, msg):

        markers = msg.markers

        main_tag_flag  = False
        main_tag_frame = None
        
        for i in xrange(len(markers)):
            if markers[i].id == self.main_id:
                main_tag_flag = True
                main_tag_frame = posemath.fromMsg(markers[i].pose.pose)
                
        if main_tag_flag == False: return
                    
        for i in xrange(len(markers)):

            if markers[i].id != self.main_tag_id:                
                tag_id    = markers[i].id
                tag_frame = posemath.fromMsg(markers[i].pose.pose)
                
                frame_diff = main_tag_frame.Inverse()*tag_frame
                self.updateDictFromFrame(tag_id, frame_diff)

                
    def updateDictFromFrame(self, tag_id, frame):

        # position filtering
        if (main_tag_frame.p - frame.p).Norm() > self.pos_thres : return
        
        corner1 = frame * self.corner1
        corner2 = frame * self.corner2
        corner3 = frame * self.corner3
        corner4 = frame * self.corner4
        
        tag_flag = False
        for key in self.bundle_dict.keys():
            if key == str(tag_id):
                tag_flag = True

                self.bundle_dict[key][0] = (self.bundle_dict[key][0] + corner1) / 2.0
                self.bundle_dict[key][1] = (self.bundle_dict[key][1] + corner2) / 2.0
                self.bundle_dict[key][2] = (self.bundle_dict[key][2] + corner3) / 2.0
                self.bundle_dict[key][3] = (self.bundle_dict[key][3] + corner4) / 2.0
                
                
        # New tag
        if tag_flag == False:
            self.bundle_list.append(tag_id)            
            self.bundle_dict[str(tag_id)] = [corner1, corner2, corner3, corner4]

        print "Detected tags: ", self.bundle_list


    def print_xml(self)

        print '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>'
        print '<multimarker markers="'+str(self.n_tags)+'">'

        for i in self.bundle_list:

            d = self.bundle_dict[str(i)]

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

            d = self.bundle_dict[str(i)]

            f.write('    <marker index="'+str(i)+'" status="1">\n')
            f.write('        <corner x="'+str(d[0][0])+'" y="'+str(d[0][1])+'" z="'+str(d[0][2])+'" />\n')
            f.write('        <corner x="'+str(d[1][0])+'" y="'+str(d[1][1])+'" z="'+str(d[1][2])+'" />\n')
            f.write('        <corner x="'+str(d[2][0])+'" y="'+str(d[2][1])+'" z="'+str(d[2][2])+'" />\n')
            f.write('        <corner x="'+str(d[3][0])+'" y="'+str(d[3][1])+'" z="'+str(d[3][2])+'" />\n')
            f.write('    </marker>\n')

        print '</multimarker>\n'        

        f.close()

        

if __name__ == '__main__':
    rospy.init_node('ar_tag_bundle_estimation')

    total_tags = 3
    main_tag_id = 9
    corner1 = PyKDL.Vector(-1.65, -1.65, 0.0)
    corner2 = PyKDL.Vector( 1.65, -1.65, 0.0)
    corner3 = PyKDL.Vector( 1.65,  1.65, 0.0)
    corner4 = PyKDL.Vector(-1.65,  1.65, 0.0)
    pos_thres = 0.2
    
    atb = arTagBundle(total_tags, main_tag_id, corner1, corner2, corner3, corner4, pos_thres)
    
    rate = rospy.Rate(10) # 25Hz, nominally.    
    while not rospy.is_shutdown():
        ## log.log_state()
        rate.sleep()
    
    atb.print_xml()
    atb.save_xml()

        
