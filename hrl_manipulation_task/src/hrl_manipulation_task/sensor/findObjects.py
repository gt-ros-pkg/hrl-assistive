#!/usr/bin/env python

import rospy, roslib
import numpy as np
import os, threading, copy

import PyKDL

from tf_conversions import posemath
from hrl_lib import quaternion as qt
import hrl_lib.util as ut
import hrl_lib.circular_buffer as cb

from ar_track_alvar_msgs.msg import AlvarMarkers
import geometry_msgs
from geometry_msgs.msg import PoseStamped, PointStamped, PoseArray

QUEUE_SIZE = 10

class arTagDetector:

    def __init__(self, task_name):

        print "Start object observation"
        self.task = task_name
        
        self.initParams()
        self.initComms()
       
        self.frame_lock = threading.RLock()                

    def initParams(self):

        self.nTags      = rospy.get_param('hrl_manipulation_task/'+self.task+'/artag_total_tags')
        self.tag_id     = rospy.get_param('hrl_manipulation_task/'+self.task+'/artag_id')
        ## self.tag_length = rospy.get_param('hrl_manipulation_task/artag_length')
        ## self.tag_max_id = rospy.get_param('hrl_manipulation_task/artag_max_id')
        ## self.tag_buf_size = rospy.get_param('hrl_manipulation_task/artag_buf_size')
        ## self.pos_thres  = rospy.get_param('hrl_manipulation_task/artag_pos_thres')

        self.hist_size = 10
        self.pos_buf  = []
        self.quat_buf = []
        for i in xrange(self.nTags):
            self.pos_buf.append( cb.CircularBuffer(self.hist_size, (3,)) )
            self.quat_buf.append( cb.CircularBuffer(self.hist_size, (4,)) )
        
        
    def initComms(self):
        self.pose_pub = []
        for i in xrange(self.nTags):
            self.pose_pub.append( rospy.Publisher("ar_track_alvar/pose_"+str(i), PoseStamped, \
                                                  queue_size=QUEUE_SIZE, latch=True) )

        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.arTagCallback)

                
    def arTagCallback(self, msg):

        markers = msg.markers

        with self.frame_lock:
            for i in xrange(len(markers)):
                for j in xrange(len(self.tag_id)):

                    if markers[i].id == self.tag_id[j]:
                        tag_frame = posemath.fromMsg(markers[i].pose.pose)

                        if tag_frame.p.Norm() > 2.0: 
                            print "Detected tag is located at too far location."
                            continue

                        cur_p = np.array([tag_frame.p[0], tag_frame.p[1], tag_frame.p[2]])
                        cur_q = np.array([tag_frame.M.GetQuaternion()[0], 
                                          tag_frame.M.GetQuaternion()[1], 
                                          tag_frame.M.GetQuaternion()[2],
                                          tag_frame.M.GetQuaternion()[3]])

                        if len(self.quat_buf[j]) < 1:
                            self.pos_buf[j].append( cur_p )
                            self.quat_buf[j].append( cur_q )
                        else:
                            first_p = self.pos_buf[j][0]
                            first_q = self.quat_buf[j][0]

                            # check close quaternion and inverse
                            if np.dot(cur_q, first_q) < 0.0:
                                cur_q *= -1.0

                            self.pos_buf[j].append( cur_p )
                            self.quat_buf[j].append( cur_q )

                        positions  = self.pos_buf[j].get_array()
                        quaternions  = self.quat_buf[j].get_array() 

                        # median
                        positions = np.sort(positions, axis=0)
                        p = positions[len(positions)/2]

                        quaternions = np.sort(quaternions, axis=0)
                        q = quaternions[len(quaternions)/2]

                        tag_frame.p[0] = p[0]
                        tag_frame.p[1] = p[1]
                        tag_frame.p[2] = p[2]                    
                        tag_frame.M = PyKDL.Rotation.Quaternion(q[0], q[1], q[2], q[3])
                        self.pubTagPose(j, tag_frame)


    def pubTagPose(self, idx, tag_frame):
        
        ps = PoseStamped()
        ps.header.frame_id = 'torso_lift_link'
        ps.header.stamp = rospy.Time.now()
        ps.pose.position.x = tag_frame.p[0]
        ps.pose.position.y = tag_frame.p[1]
        ps.pose.position.z = tag_frame.p[2]
        
        ps.pose.orientation.x = tag_frame.M.GetQuaternion()[0]
        ps.pose.orientation.y = tag_frame.M.GetQuaternion()[1]
        ps.pose.orientation.z = tag_frame.M.GetQuaternion()[2]
        ps.pose.orientation.w = tag_frame.M.GetQuaternion()[3]

        ## print idx, self.tag_id[idx]
        self.pose_pub[idx].publish(ps)


    def pubVirtualTagPose(self):

        for i in xrange(self.nTags):
        
            f = PyKDL.Frame.Identity()
            f.p = PyKDL.Vector(0.5, 0.2, -0.2)
            f.M = PyKDL.Rotation.Quaternion(0,0,0,1)

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

            self.pose_pub[i].publish(ps)
        

if __name__ == '__main__':
    rospy.init_node('ar_tag_estimation')

    import optparse
    p = optparse.OptionParser()
    ## ## p.add_option('--renew', action='store_true', dest='bRenew',
    ## ##              default=False, help='Renew frame pickle files.')
    p.add_option('--task', '--t', action='store', type='string', dest='task_name',
                 default='pushing', help='Set the name of current task.')
    p.add_option('--virtual', '--v', action='store_true', dest='bVirtual',
                 default=False, help='Send a vitual frame.')
    opt, args = p.parse_args()


    atd = arTagDetector(opt.task_name)

    rate = rospy.Rate(10) # 25Hz, nominally.    
    while not rospy.is_shutdown():

        ## if opt.bVirtual:
        ##     atd.pubVirtualTagPose()
        ##     continue        
        rate.sleep()


        
        
