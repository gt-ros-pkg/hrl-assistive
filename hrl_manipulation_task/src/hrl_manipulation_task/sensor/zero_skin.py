#!/usr/bin/env python

#   Copyright 2013 Georgia Tech Research Corporation
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#  http://healthcare-robotics.com/

# @package hrl_manipulation_task
#
# @author Daehyung Park
# @version 0.1
# @copyright Apache 2.0

import sys
import rospy

from std_msgs.msg import String, Empty

QUEUE_SIZE = 10



if __name__ == '__main__':
    rospy.init_node('zero_skin')

    import optparse
    p = optparse.OptionParser()
    p.add_option('--arm', '-a', action='store', dest='arm', type='string',
                 help='which arm to use (l, r)', default=None)
    opt, args = p.parse_args()
    
    zero_forearm_pub = rospy.Publisher(
        '/pr2_fabric_'+opt.arm+'_forearm_sensor/zero_sensor', Empty, queue_size=QUEUE_SIZE)
    zero_upperarm_pub = rospy.Publisher(
        '/pr2_fabric_'+opt.arm+'_upperarm_sensor/zero_sensor', Empty, queue_size=QUEUE_SIZE)
    

    rospy.sleep(1.0)
    zero_forearm_pub.publish(Empty())
    zero_upperarm_pub.publish(Empty())
    rospy.sleep(1.0)


