#!/usr/bin/env python

import sys
import argparse
import numpy as np

import rospy
from tf import TransformListener, ConnectivityException, LookupException, ExtrapolationException
from geometry_msgs.msg import PoseStamped

from hrl_task_planning import pddl_utils as pddl
from hrl_task_planning.msg import PDDLState


def _dict_to_pose_stamped(ps_dict):
    ps = PoseStamped()
    ps.header.seq = ps_dict['header']['seq']
    ps.header.stamp.secs = ps_dict['header']['stamp']['secs']
    ps.header.stamp.nsecs = ps_dict['header']['stamp']['nsecs']
    ps.header.frame_id = ps_dict['header']['frame_id']
    ps.pose.position.x = ps_dict['pose']['position']['x']
    ps.pose.position.y = ps_dict['pose']['position']['y']
    ps.pose.position.z = ps_dict['pose']['position']['z']
    ps.pose.orientation.x = ps_dict['pose']['orientation']['x']
    ps.pose.orientation.y = ps_dict['pose']['orientation']['y']
    ps.pose.orientation.z = ps_dict['pose']['orientation']['z']
    ps.pose.orientation.w = ps_dict['pose']['orientation']['w']
    return ps


class ProximityMonitor(object):
    def __init__(self, domain, object_name, frame, distance_threshold=0.1):
        self.domain = domain
        self.object_name = object_name
        self.frame = frame
        self.dist_thresh = distance_threshold
        self.grasping = []
        self.near_locations = []
        self.tfl = TransformListener()
        self.state = []
        self.known_locations = {}
        self.state_update_pub = rospy.Publisher("/pddl_tasks/%s/state_updates" % self.domain, PDDLState)
        self.domain_state_sub = rospy.Subscriber("/pddl_tasks/%s/state" % self.domain, PDDLState, self.domain_state_cb)

    def domain_state_cb(self, state_msg):
        # When state is updated, find all known locations in updated state
        pub = False
        predicates = map(pddl.Predicate.from_string, state_msg.predicates)
        current_known_locations = []
        current_grasped_items = []
        for pred in predicates:
            if pred.name == 'KNOWN':
                loc = pred.args[0]
                current_known_locations.append(loc)
                if loc not in self.known_locations:
                    try:
                        pose_dict = rospy.get_param("/pddl_tasks/%s/KNOWN/%s" % (self.domain, loc))
                        self.known_locations[loc] = _dict_to_pose_stamped(pose_dict)
                    except KeyError:
                        rospy.logwarn("[%s] Expected location of %s on parameter server, but not found!", rospy.get_name(), pred.args[0])
                        pass
            if pred.name == 'GRASPING' and pred.args[0] == self.object_name:
                    current_grasped_items.append(pred.args[1])
                    if pred.args[1] not in self.grasping:
                        self.grasping.append(pred.args[1])

        # Identify forgotten locations, remove from internal state list, and replace with negation
        update_preds = []
        for loc in self.known_locations:
            if loc not in current_known_locations:
                self.known_locations.pop(loc)
                if loc in self.near_locations:
                    update_preds.append(pddl.Predicate('AT', [self.object_name, loc], neg=True))
                    self.near_locations.remove(loc)

        for item in self.grasping:
            if item not in current_grasped_items:
                self.grasping.remove(item)
                for loc in self.near_locations:
                    update_preds.append(pddl.Predicate('AT', [item, loc]))  # the item was grasped, now isn't (dropped), so must be at the location

        for pred in update_preds:
            pos_pred = pddl.Predicate(pred.name, pred.args)
            if pred.neg and pos_pred in self.state:
                self.state.remove(pos_pred)
                pub = True
            if not pred.neg and pred not in self.state:
                self.state.add(pred)
                pub = True

        if pub:
            update_msg = PDDLState()
            update_msg.domain = self.domain
            update_msg.predicates = map(str, self.state)
            self.state_update_pub.publish(update_msg)

    def check_state(self):
        pub = False
        now_near = []

        for loc_name, loc_pose in self.known_locations.iteritems():
            try:
                (trans, _) = self.tfl.lookupTransform(loc_pose.header.frame_id, self.frame, rospy.Time(0))
                loc = np.array([loc_pose.pose.position.x, loc_pose.pose.position.y, loc_pose.pose.position.z])
                if np.linalg.norm(loc-trans) < self.dist_thresh:
                    now_near.append(loc_name)
            except (LookupException, ConnectivityException, ExtrapolationException):
                pass

        update_preds = []
        # Add newly near locations
        for loc in now_near:
            if loc not in self.near_locations:
                self.near_locations.append(loc)
                update_preds.append(pddl.Predicate('AT', [self.object_name, loc]))
        # remove no-longer-near locations
        for loc in self.near_locations:
            if loc not in now_near:
                self.near_locations.remove(loc)
                update_preds.append(pddl.Predicate('AT', [self.object_name, loc], neg=True))

        for pred in update_preds:
            pos_pred = pddl.Predicate(pred.name, pred.args)
            if pred.neg and pos_pred in self.state:
                self.state.remove(pos_pred)
                pub = True
            if not pred.neg and pred not in self.state:
                self.state.append(pred)
                pub = True

        if pub:
            update_msg = PDDLState()
            update_msg.domain = self.domain
            update_msg.predicates = map(str, self.state)
            self.state_update_pub.publish(update_msg)


def main():
    parser = argparse.ArgumentParser(description="Monitor location of frames, and update domain state when close to known locations.")
    parser.add_argument('domain', help="The domain for which the parameter will be monitored.")
    parser.add_argument('object', help="The pddl name of the object to monitor for nearness to locations.")
    parser.add_argument('frame', help="The TF frame corresponding to the pddl object.")
    parser.add_argument('--distance', '-d', default=0.1, help="The threshold distance to declare 'near.'")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node('%s_proximity_monitor' % args.domain)
    monitor = ProximityMonitor(args.domain, args.object, args.frame, args.distance)
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        monitor.check_state()
        rate.sleep()
