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


def pushing_microwave_white(armReachActionLeft, armReachActionRight, log, detection_flag, \
                            train=False, abnormal=False, bCont=False, status='skip'):

    log.task = 'pushing_microwhite'
    log.initParams()
    
    ## Scooping -----------------------------------    
    print "Initializing left arm for scooping"
    print armReachActionLeft("getMainTagPos")
    print armReachActionLeft("initMicroWhite")
    
    print "Start to log!"    
    log.log_start()
    if detection_flag: log.enableDetector(True)
    
    print "Running scooping!"
    print armReachActionLeft("runMicroWhite")

    if detection_flag: log.enableDetector(False)
    print "Finish to log!"    
    log.close_log_file(bCont, status)

def pushing_microwave_black(armReachActionLeft, armReachActionRight, log, detection_flag, \
                            train=False, abnormal=False, bCont=False, status='skip'):

    log.task = 'pushing_microblack'
    log.initParams()
    
    ## Scooping -----------------------------------    
    print "Initializing left arm for scooping"
    print armReachActionLeft("getMainTagPos")
    print armReachActionLeft("initMicroBlack")
    
    print "Start to log!"    
    log.log_start()
    if detection_flag: log.enableDetector(True)
    
    print "Running pushing!"
    print armReachActionLeft("runMicroBlack")

    if detection_flag: log.enableDetector(False)
    print "Finish to log!"    
    log.close_log_file(bCont, status)

    
def pushing_toolcase(armReachActionLeft, armReachActionRight, log, detection_flag, \
                     train=False, abnormal=False, bCont=False, status='skip'):

    log.task = 'pushing_toolcase'
    log.initParams()
    
    ## Scooping -----------------------------------    
    print "Initializing left arm for scooping"
    print armReachActionLeft("getMainTagPos")
    print armReachActionLeft("initToolCase")
    print armReachActionLeft("runToolCase1")
    
    print "Start to log!"    
    log.log_start()
    if detection_flag: log.enableDetector(True)
    
    print "Running pushing!"
    print armReachActionLeft("runToolCase2")

    if detection_flag: log.enableDetector(False)
    print "Finish to log!"    
    log.close_log_file(bCont, status)

    print armReachActionLeft("initToolCase")
    
    
if __name__ == '__main__':
    
    import optparse
    p = optparse.OptionParser()
    p.add_option('--data_pub', '--dp', action='store_true', dest='bDataPub',
                 default=False, help='Continuously publish data.')
    p.add_option('--continue', '--c', action='store_true', dest='bCont',
                 default=False, help='Continuously run program.')
    p.add_option('--status', '--s', action='store', dest='bStatus',
                 default='success', help='continous data collection status [sucesss, failure, skip(default)]')
    opt, args = p.parse_args()

    rospy.init_node('arm_reach_client')

    rospy.wait_for_service("/arm_reach_enable")
    armReachActionLeft  = rospy.ServiceProxy("/arm_reach_enable", String_String)
    armReachActionRight = rospy.ServiceProxy("/right/arm_reach_enable", String_String)


    ## task_name = 'pushing_microwhite'
    task_name = 'pushing_microblack'
    ## task_name = 'pushing_toolcase'
    
    log = logger(ft=True, audio=False, audio_wrist=True, kinematics=True, vision_artag=True, \
                 vision_change=False, \
                 pps=False, skin=False, \
                 subject="gatsbii", task=task_name, data_pub=opt.bDataPub, verbose=False)

    # need to be removed somehow
    last_trial  = '2'
    last_detect = '2'
    
    while not rospy.is_shutdown():

        if opt.bCont: trial = last_trial
        else:
            trial  = raw_input('Enter trial\'s status (e.g. 1:MW, 2: MB, 3: TC else: exit): ')
            if trial=='': trial=last_trial
        
        if trial is '1' or trial is '2' or trial is '3' or trial is '4' or trial is '5':

            if opt.bCont: detect = last_detect
            else: detect = raw_input('Enable anomaly detection? (e.g. 1:enable else: disable): ')
            if detect == '': detect=last_detect
            if detect == '1': detection_flag = True
            else: detection_flag = False
            
            if trial == '1':
                if task_name == 'pushing_microwhite':
                    pushing_microwave_white(armReachActionLeft, armReachActionRight, log, detection_flag, \
                                            bCont=opt.bCont, status=opt.bStatus)
            elif trial == '2':
                if task_name == 'pushing_microblack':
                    pushing_microwave_black(armReachActionLeft, armReachActionRight, log, detection_flag, \
                                            bCont=opt.bCont, status=opt.bStatus)
            elif trial == '3':
                if task_name == 'pushing_toolcase':
                    pushing_toolcase(armReachActionLeft, armReachActionRight, log, detection_flag,\
                                            bCont=opt.bCont, status=opt.bStatus)
        else:
            break

        last_trial  = trial
        last_detect = detect
    
    ## t1 = datetime.datetime.now()
    ## t2 = datetime.datetime.now()
    ## t  = t2-t1
    ## print "time delay: ", t.seconds
    
