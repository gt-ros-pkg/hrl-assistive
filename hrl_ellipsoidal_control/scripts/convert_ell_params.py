#!/usr/bin/python

import sys

import roslib
roslib.load_manifest("hrl_ellipsoidal_control")
import rosbag

from hrl_ellipsoidal_control.msg import EllipsoidParams

def main():
    old_bag = rosbag.Bag(sys.argv[1], 'r')
    for topic, msg, ts in old_bag.read_messages():
        old_topic = topic
        old_ep = msg
    old_bag.close()
    new_ep = EllipsoidParams()
    new_ep.e_frame = old_ep.e_frame
    new_ep.height = old_ep.height
    new_ep.E = old_ep.E
    new_ep.is_oblate = False
    new_bag = rosbag.Bag(sys.argv[1], 'w')
    new_bag.write(old_topic, new_ep)
    new_bag.close()

if __name__ == "__main__":
    main()
