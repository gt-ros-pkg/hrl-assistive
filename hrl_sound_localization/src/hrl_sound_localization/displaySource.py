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

import numpy as np, math
import roslib; roslib.load_manifest('hrl_sound_localization')
import rospy
import threading

# ROS message
## import tf
from std_msgs.msg import Bool, Empty, Int32, Int64, Float32, Float64
from hark_msgs.msg import HarkSource

# HRL library
from hrl_common_code_darpa_m3.visualization import draw_scene as ds


class displaySource():
    def __init__(self):
        rospy.init_node('display_sources')

        self.count = 0
        self.exist_src_num = 0
        self.src   = 0

        self.source_color  = [0.7,0,0,0.7]
        self.text_color = [0.,0.,0.,0.7]        

        self.rviz_obs_num   = 0
        self.rviz_obs_names = [[] for i in range(100)]
        self.rviz_obs_pos   = [[[] for j in range(3)] for i in range(100)]
        self.rviz_obs_dim   = [[[] for j in range(3)] for i in range(100)]

        self.src_lock = threading.RLock()

        self.initComms()

    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''
        print "Initialize pusblishers and subscribers"
        rospy.Subscriber('HarkSource', HarkSource, self.harkSourceCallback)

        # drawing
        self.source_viz = ds.SceneDraw("hark/source_viz", "/head_mount_kinect_depth_link")


    def harkSourceCallback(self, msg):
        '''
        Get all the source locations from hark. 
        '''
        with self.src_lock:        
            self.count = msg.count
            self.exist_src_num = msg.exist_src_num
            self.src   = msg.src
        
    def draw_sources(self, start_id=0):
        '''
        Draw id and location of sounds
        '''
        self.src_lock.acquire()
        for i in xrange(self.exist_src_num):
            src_id    = self.src[i].id
            power     = self.src[i].power 
            pos_x     = self.src[i].x
            pos_y     = self.src[i].y
            pos_z     = self.src[i].z
            azimuth   = self.src[i].azimuth
            elevation = self.src[i].elevation

            if power < 25.5: continue
            self.source_viz.pub_body([pos_x, pos_y, pos_z],
                                     [0,0,0,1],
                                     [0.1, 0.1, 0.1],
                                     self.source_color, 
                                     start_id+i,
                                     self.source_viz.Marker.SPHERE)
            print src_id, power
        self.src_lock.release()


    ## def tfBroadcaster(self):
    ##     br = tf.TransformBroadcaster()
    ##     br.sendTransform((0, 0, 0),
    ##                      tf.transformations.quaternion_from_euler(0, 0, 0),
    ##                      rospy.Time.now(),
    ##                      "torso_lift_link",
    ##                      "world")
        

    def run(self):
        '''
        Update the visualization data and publish it.
        '''
        rt = rospy.Rate(20)
        while not rospy.is_shutdown():
            ## self.tfBroadcaster()
            self.draw_sources(10)
            rt.sleep()
            

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    opt, args = p.parse_args()

    ds = displaySource()
    ds.run()
    

