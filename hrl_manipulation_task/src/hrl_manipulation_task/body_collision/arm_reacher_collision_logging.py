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

# system library
import time
import datetime

# ROS library
import rospy, roslib

# HRL library
from hrl_srvs.srv import String_String
import hrl_lib.util as ut

from hrl_manipulation_task.record_data import logger


def swing(armReachActionLeft, armReachActionRight, log, detection_flag):

    log.task = 'touching'
    log.initParams()
    
    ## Scooping -----------------------------------    
    print "Initializing left arm for scooping"
    print armReachActionLeft("initSwing")
    ut.get_keystroke('Hit a key to proceed next')
        
    print "Start to log!"    
    log.log_start()
    if detection_flag: log.enableDetector(True)
    
    print "Running scooping!"
    print armReachActionLeft("runSwing")

    if detection_flag: log.enableDetector(False)
    print "Finish to log!"    
    log.close_log_file()

        
    
if __name__ == '__main__':
    
    import optparse
    p = optparse.OptionParser()
    p.add_option('--data_pub', '--dp', action='store_true', dest='bDataPub',
                 default=False, help='Continuously publish data.')
    opt, args = p.parse_args()

    rospy.init_node('arm_reach_client')

    rospy.wait_for_service("/arm_reach_enable")
    armReachActionLeft  = rospy.ServiceProxy("/arm_reach_enable", String_String)
    armReachActionRight = rospy.ServiceProxy("/right/arm_reach_enable", String_String)

    log = logger(ft=False, audio=True, kinematics=True, vision_artag=False, vision_change=True, pps=False, 
                 skin=True, subject="gatsbii", task='collision', data_pub=opt.bDataPub, verbose=False)

    last_trial  = '4'
    last_detect = '2'
                 
    while not rospy.is_shutdown():

        detection_flag = False
        
        trial  = raw_input('Enter trial\'s status (e.g. 1:swing, else: exit): ')
        if trial=='': trial=last_trial
            
        if trial is '1' or trial is '2':
            detect = raw_input('Enable anomaly detection? (e.g. 1:enable else: disable): ')
            if detect == '': detect=last_detect
            if detect == '1': detection_flag = True
            
            if trial == '1':
                swing(armReachActionLeft, armReachActionRight, log, detection_flag)
            else:
                swing(armReachActionLeft, armReachActionRight, log, detection_flag)
        else:
            break

        last_trial  = trial
        last_detect = detect
    
    ## t1 = datetime.datetime.now()
    ## t2 = datetime.datetime.now()
    ## t  = t2-t1
    ## print "time delay: ", t.seconds
    
