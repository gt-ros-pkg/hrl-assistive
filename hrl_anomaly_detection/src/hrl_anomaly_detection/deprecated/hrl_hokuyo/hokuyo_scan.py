#
# Copyright (c) 2009, Georgia Tech Research Corporation
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

#  \author Advait Jain (Healthcare Robotics Lab, Georgia Tech.)

import math, numpy as np
import sys, time

NAME = 'utm_python_listener'
## import roslib; roslib.load_manifest('hrl_hokuyo')
import rospy
from sensor_msgs.msg import LaserScan
from threading import RLock

## import hokuyo_processing as hp
import copy

import numpy as np

class HokuyoScan():
    ''' This class has the data of a laser scan.
    '''
    def __init__(self,hokuyo_type,angular_res,max_range,min_range,start_angle,end_angle):
        ''' hokuyo_type - 'urg', 'utm'
            angular_res - angle (radians) between consecutive points in the scan.
            min_range, max_range -  in meters
            ranges,angles,intensities - 1xn_points numpy matrix.
        '''
        self.hokuyo_type = hokuyo_type
        self.angular_res = angular_res
        self.max_range   = max_range
        self.min_range   = min_range
        self.start_angle = start_angle
        self.end_angle   = end_angle
        self.n_points    = int((end_angle-start_angle)/angular_res)+1

        self.ranges      = None
        self.intensities = None
        self.angles = []

        for i in xrange(self.n_points):
            self.angles.append(hp.index_to_angle(self,i))
        self.angles = np.matrix(self.angles)


class Urg():
    ''' get a scan from urg using player.
    '''
    def __init__(self, urg_number,start_angle=None,end_angle=None, host='localhost', port=6665):
        ''' host, port - hostname and port of the player server
        '''
        import player_functions as pf
        import playerc as pl

        self.player_client = pf.player_connect(host,port)
        self.laser = pl.playerc_laser(self.player_client, urg_number)
        if self.laser.subscribe(pl.PLAYERC_OPEN_MODE) != 0:
            raise RuntimeError("hokuyo_scan.Urg.__init__: Unable to connect to urg!\n")

        for i in range(10):
            self.player_client.read()

        start_angle_fullscan = -math.radians(654./2*0.36)
        end_angle_fullscan = math.radians(654./2*0.36)
        if start_angle == None or start_angle<start_angle_fullscan:
            start_angle_subscan = start_angle_fullscan
        else:
            start_angle_subscan = start_angle

        if end_angle == None or end_angle>end_angle_fullscan:
            end_angle_subscan = end_angle_fullscan
        else:
            end_angle_subscan = end_angle

        ang_res = math.radians(0.36)
        self.start_index_fullscan=(start_angle_subscan-start_angle_fullscan)/ang_res
        self.end_index_fullscan=(end_angle_subscan-start_angle_fullscan)/ang_res

        self.hokuyo_scan = HokuyoScan(hokuyo_type = 'urg', angular_res = math.radians(0.36),
                                      max_range = 4.0, min_range = 0.2,
                                      start_angle=start_angle_subscan,
                                      end_angle=end_angle_subscan)



class Utm():
    ''' get a scan from a UTM using ROS
    '''
    def __init__(self, utm_number,start_angle=None,end_angle=None,ros_init_node=True):

        hokuyo_node_name = '/utm%d'%utm_number
#        max_ang = rospy.get_param(hokuyo_node_name+'/max_ang')
#        min_ang = rospy.get_param(hokuyo_node_name+'/min_ang')
#        start_angle_fullscan = min_ang
#        end_angle_fullscan = max_ang
#        print 'max_angle:', math.degrees(max_ang)
#        print 'min_angle:', math.degrees(min_ang)

        ## max_ang_degrees = rospy.get_param(hokuyo_node_name+'/max_ang_degrees')
        ## min_ang_degrees = rospy.get_param(hokuyo_node_name+'/min_ang_degrees')
        start_angle_fullscan = math.radians(min_ang_degrees)
        end_angle_fullscan = math.radians(max_ang_degrees)

        # This is actually determined by the ROS node params and not the UTM.
#        start_angle_fullscan = -math.radians(1080./2*0.25) #270deg
#        end_angle_fullscan = math.radians(1080./2*0.25)
#        start_angle_fullscan = -math.radians(720./2*0.25)  #180deg
#        end_angle_fullscan = math.radians(720./2*0.25)
#        start_angle_fullscan = -math.radians(559./2*0.25)   #140deg
#        end_angle_fullscan = math.radians(559./2*0.25)

        if start_angle == None or start_angle<start_angle_fullscan:
            start_angle_subscan = start_angle_fullscan
        else:
            start_angle_subscan = start_angle

        if end_angle == None or end_angle>end_angle_fullscan:
            end_angle_subscan = end_angle_fullscan
        else:
            end_angle_subscan = end_angle

        ang_res = math.radians(0.25)
        self.start_index_fullscan=int(round((start_angle_subscan-start_angle_fullscan)/ang_res))
        self.end_index_fullscan=int(round((end_angle_subscan-start_angle_fullscan)/ang_res))

        self.hokuyo_scan = HokuyoScan(hokuyo_type = 'utm', angular_res = math.radians(0.25),
                                      max_range = 10.0, min_range = 0.1,
                                      start_angle=start_angle_subscan,
                                      end_angle=end_angle_subscan)

        self.lock = RLock()
        self.lock_init = RLock()
        self.connected_to_ros = False

        self.ranges,self.intensities = None,None

        ## if ros_init_node:
        ##     try:
        ##         print 'Utm: init ros node.'
        ##         rospy.init_node(NAME, anonymous=True)
        ##     except rospy.ROSException, e:
        ##         print 'Utm: rospy already initialized. Got message', e
        ##         pass

        ## rospy.Subscriber("utm%d_scan"%(utm_number), LaserScan,
        ##                  self.callback, queue_size = 1)


class Hokuyo():
    ''' common class for getting scans from both urg and utm.
    '''

    def __init__(self, hokuyo_type, hokuyo_number, start_angle=None,
                 end_angle=None, flip=False, ros_init_node=True):
        ''' hokuyo_type - 'urg', 'utm'
                        - 'utm' requires ros revision 2536
                                         ros-kg revision 5612
            hokuyo_number - 0,1,2 ...
            start_angle, end_angle - unoccluded part of the scan. (radians)
            flip - whether to flip the scans (hokuyo upside-down)
        '''
        self.hokuyo_type = hokuyo_type
        self.flip = flip

        if hokuyo_type == 'urg':
            self.hokuyo = Urg(hokuyo_number,start_angle,end_angle)
        elif hokuyo_type == 'utm':
            self.hokuyo = Utm(hokuyo_number,start_angle,end_angle,ros_init_node=ros_init_node)
        else:
            raise RuntimeError('hokuyo_scan.Hokuyo.__init__: Unknown hokuyo type: %s\n'%(hokuyo_type))

