#!/usr/bin/env python
#
# Copyright (c) 2014, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

# system
import rospy, roslib
import os, threading, copy

# util
import numpy as np
import hrl_lib.circular_buffer as cb
import hrl_lib.quaternion as qt
from tf_conversions import posemath

# msg
from ar_track_alvar.msg import AlvarMarkers
## from ar_track_alvar_msgs.msg import AlvarMarkers
from geometry_msgs.msg import PoseStamped, PointStamped, PoseArray

class artag_vision(threading.Thread):

    def __init__(self, task='feeding', verbose=False, viz=False):
        super(artag_vision, self).__init__()        
        self.daemon = True
        self.cancelled = False
        self.isReset = False
        self.verbose = verbose
        self.viz     = viz
        self.task    = task
       
        # instant data
        self.time       = None
        
        # Declare containers        
        self.vision_tag_pos  = None
        self.vision_tag_quat = None

        self.lock = threading.Lock()        

        self.initParams()
        self.initComms()
        if self.verbose: print "artag_vision>> initialization complete"
        
    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''
        if self.verbose: print "artag_vision>> Initialize pusblishers and subscribers"
        for i in xrange(self.nTags):
            self.artag_pub = rospy.Publisher("ar_track_alvar/artag_vision_pose_"+str(i), PoseStamped, latch=True)
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.arTagCallback)

    def initParams(self):
        '''
        Get parameters
        '''
        self.nTags          = rospy.get_param('hrl_manipulation_task/'+self.task+'/artag_total_tags')        
        self.tag_id         = rospy.get_param('hrl_manipulation_task/'+self.task+'/artag_id')
        ## self.tag_length     = rospy.get_param('hrl_manipulation_task/'+self.task+'/artag_length')
        ## self.tag_max_id     = rospy.get_param('hrl_manipulation_task/'+self.task+'/artag_max_id')
        self.tag_buf_size   = rospy.get_param('hrl_manipulation_task/'+self.task+'/artag_buf_size')

        self.artag_pos  = np.zeros((3*self.nTags,1))
        self.artag_quat = np.zeros((4*self.nTags,1))
        self.pos_buf    = []
        self.quat_buf   = []
        for i in xrange(self.nTags):
            self.pos_buf.append( cb.CircularBuffer(self.tag_buf_size, (3,)) )
            self.quat_buf.append( cb.CircularBuffer(self.tag_buf_size, (4,)) )
        
    def pubARtag(self, idx, p, q):

        ps = PoseStamped()
        ps.header.frame_id = 'torso_lift_link'
        ps.header.stamp = rospy.Time.now()
        ps.pose.position.x = p[0]
        ps.pose.position.y = p[1]
        ps.pose.position.z = p[2]
        
        ps.pose.orientation.x = q[0]
        ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]
        ps.pose.orientation.w = q[3]

        self.artag_pub[idx].publish(ps)
        
    def arTagCallback(self, msg):

        time_stamp = msg.header.stamp
        markers    = msg.markers

        with self.lock:
            for i in xrange(len(markers)):
                for j in xrange(len(self.tag_id)):

                    if markers[i].id == self.tag_id[j]:

                        cur_p = np.array([markers[i].pose.pose.position.x, 
                                          markers[i].pose.pose.position.y, 
                                          markers[i].pose.pose.position.z])
                        cur_q = np.array([markers[i].pose.pose.orientation.x, 
                                          markers[i].pose.pose.orientation.y, 
                                          markers[i].pose.pose.orientation.z,
                                          markers[i].pose.pose.orientation.w])

                        if np.linalg.norm(cur_p) > 2.0: 
                            if self.verbose: print "Detected tag is located at too far location."
                            continue

                        if len(self.quat_buf) < 1:
                            self.pos_buf.append( cur_p )
                            self.quat_buf.append( cur_q )
                        else:
                            first_p = self.pos_buf[-1]
                            first_q = self.quat_buf[-1]

                            # check close quaternion and inverse
                            if np.dot(cur_q, first_q) < 0.0:
                                cur_q *= -1.0

                            self.pos_buf.append( cur_p )
                            self.quat_buf.append( cur_q )

                        positions  = self.pos_buf[j].get_array()
                        quaternions  = self.quat_buf[j].get_array() 

                        p = None
                        q = None
                        if False:
                            # Moving average
                            p = np.mean(positions, axis=0)                                        
                            q = qt.quat_avg(quaternions)
                        else:
                            # median
                            p = np.median(positions, axis=0)
                            q = np.median(quaternions, axis=0)
                            q = qt.quat_normal(q)

                        self.time       = time_stamp.to_sec() #- self.init_time
                        self.artag_pos[3*j:3*j+3]    = p.reshape(3,1)
                        self.artag_quat[4*j+0:4*j+4] = q.reshape(4,1)

                        if self.viz: self.pubARtag(j, p,q)
                        
        
    ## def run(self):
    ##     """Overloaded Thread.run, runs the update
    ##     method once per every xx milliseconds."""
    ##     rate = rospy.Rate(20)
    ##     while not self.cancelled and not rospy.is_shutdown():
    ##         if self.isReset:

    ##             if self.counter > self.counter_prev:
    ##                 self.counter_prev = self.counter

    ##                 self.lock.acquire()                            
                    
    ##                 self.time_data.append(rospy.get_time() - self.init_time)
    ##                 if self.vision_tag_pos is None:
    ##                     self.vision_tag_pos = self.artag_pos
    ##                     self.vision_tag_quat = self.artag_quat
    ##                 else:
    ##                     self.vision_tag_pos = np.hstack([self.vision_tag_pos, self.artag_pos])
    ##                     self.vision_tag_quat = np.hstack([self.vision_tag_quat, self.artag_quat])

    ##                 self.lock.release()
    ##         rate.sleep()

    ## def cancel(self):
    ##     """End this timer thread"""
    ##     self.cancelled = True
    ##     self.isReset = False

    ## def reset(self):
    ##     self.isReset = True

    ##     # Reset containers
    ##     self.vision_tag_pos  = None
    ##     self.vision_tag_quat = None

       
    def isReady(self):
        flag = True

        for i in xrange(self.nTags):            
            if self.artag_pos[i] is None or self.artag_quat[i] is None:
                flag = False

        return flag
        
        


                    
