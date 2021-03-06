#!/usr/bin/env python

import sys
import argparse
from threading import Lock

import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers

from hrl_task_planning.msg import PDDLState


class ARTag(object):
    def __init__(self, tag_id, name, last_seen=None, timeout=7):
        self.lock = Lock()
        self.tag_id = tag_id
        self.name = name.upper()
        self.last_seen = rospy.Time(0) if last_seen is None else last_seen
        self.timeout = rospy.Duration(timeout)
        self.pose = None

    def is_fresh(self):
        return self.last_seen + self.timeout > rospy.Time.now()

    def is_stale(self):
        return not self.is_fresh()


class ARTagMonitor(object):
    def __init__(self, tags_list):
        self.tags = {}
        for tag_id, name in tags_list:
            tag_id = int(tag_id)
            self.tags[tag_id] = ARTag(tag_id, name)
        self.found_tags = []
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)
        rospy.Subscriber('/ar_pose_marker', AlvarMarkers, self.marker_cb)

    def marker_cb(self, msg):
        for marker in msg.markers:
            # Ignore tags we don't care about
            if marker.id not in self.tags:
                continue
            # update last seen time
            with self.tags[marker.id].lock:
                self.tags[marker.id].last_seen = marker.header.stamp
                self.tags[marker.id].pose = marker.pose

                if marker.id not in self.found_tags:
                    self.found_tags.append(marker.id)
                    state_msg = PDDLState()
                    state_msg.predicates = ['(FOUND-TOOL %s)' % self.tags[marker.id].name]
                    self.state_pub.publish(state_msg)

    def refresh_tags(self):
        for tag_id in self.found_tags:
            with self.tags[tag_id].lock:
                if self.tags[tag_id].is_stale():
                    self.found_tags.remove(tag_id)
                    state_msg = PDDLState()
                    state_msg.predicates = ['(NOT (FOUND-TOOL %s))' % self.tags[tag_id].name]
                    self.state_pub.publish(state_msg)


def main():
    parser = argparse.ArgumentParser(description="Report when an known AR Tag is found.")
    parser.add_argument('--tag', '-t', action='append', nargs=2, help="Pair of tag numbers and names. i.e. 0 scratcher")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    print "Args: ", args

    rospy.init_node('ar_tag_monitor')
    monitor = ARTagMonitor(args.tag)
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        monitor.refresh_tags()
        rate.sleep()
