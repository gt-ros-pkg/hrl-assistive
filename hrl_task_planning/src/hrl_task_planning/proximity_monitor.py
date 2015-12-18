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
    def __init__(self, domain, object_frames, distance_threshold=0.1):
        self.domain = domain
        self.object_frames = object_frames
        self.dist_thresh = distance_threshold
        self.tfl = TransformListener()
        self.state = []
        self.known_locations = set()
        self.location_poses = {}
        self.state_update_pub = rospy.Publisher("/pddl_tasks/%s/state_updates" % self.domain, PDDLState)
        self.domain_state_sub = rospy.Subscriber("/pddl_tasks/%s/state" % self.domain, PDDLState, self.domain_state_cb)

    def domain_state_cb(self, state_msg):
        # When state is updated, find all known locations in updated state
        known_set_update = set()
        for pred_str in state_msg.predicates:
            pred = pddl.Predicate.from_string(pred_str)
            if pred.name == 'KNOWN':
                known_set_update.add(pred.args[0])

        # Identify forgotten locations, remove from internal state list, and replace with negation
        forgotten_locations = self.known_locations.difference(known_set_update)
        removals = []
        for loc in forgotten_locations:
            for pred in self.state:
                if loc in pred.args:
                    removals.append(pred)
        for remove_pred in removals:
            self.state.remove(remove_pred)
            remove_pred.negate()
            self.state.append(remove_pred)

        # Get the location poses for newly known states
        new_locations = known_set_update.difference(self.known_locations)
        for loc in new_locations:
            self.location_poses[loc] = _dict_to_pose_stamped(rospy.get_param('/pddl_tasks/%s/KNOWN/%s' % (self.domain, loc)))

        update_msg = PDDLState()
        update_msg.domain = self.domain
        update_msg.predicates = map(str, self.state)
        self.state_update_pub.publish(update_msg)

    def check_state(self):
        pub = False
        now_near = []
        for gripper, frame in self.object_frames.iteritems():
            for loc, loc_pose in self.location_poses.iteritems():
                try:
                    (trans, _) = self.tfl.lookupTransform(loc_pose.header.frame_id, frame, rospy.Time(0))
                    loc = np.array([loc_pose.pose.position.x, loc_pose.pose.position.y, loc_pose.pose.position.z])
                    if np.linalg.norm(loc-trans) < self.dist_thresh:
                        now_near.append(pddl.Predicate('NEAR', [gripper, loc]))
                except (LookupException, ConnectivityException, ExtrapolationException):
                    pass
        for pred in self.state:
            pos_pred = pddl.Predicate(pred.name, pred.args)
            if pred.neg and pos_pred in now_near:
                pred.negate()  # Was NOT near, now is
                pub = True
            if not pred.neg and pred not in now_near:
                self.state.remove(pred)  # Was near, now isn't
                pub = True
        newly_near = [pred for pred in now_near if pred not in self.state]  # add newly near states
        if newly_near:
            self.state.extend(newly_near)
            pub = True

        if pub:
            update_msg = PDDLState()
            update_msg.domain = self.domain
            update_msg.predicates = map(str, self.state)
            self.state_update_pub.publish(update_msg)


def main():
    parser = argparse.ArgumentParser(description="Monitor location of frames, and update domain state when close to known locations.")
    parser.add_argument('domain', help="The domain for which the parameter will be monitored.")
    parser.add_argument('--objects', '-o', nargs="*", default=[], help="A list of pddl object names to monitor for nearness to locations.")
    parser.add_argument('--frames', '-f', nargs="*", default=[], help="A list of frames corresponding to the list of pddl object names.")
    parser.add_argument('--distance', '-d', default=0.1, help="The threshold distance to declare 'near.'")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    assert len(args.objects) == len(args.frames), "A TF frame name must be given for each listed object."
    if not args.objects:
        object_frames = {"RIGHT_GRIPPER": "r_gripper_tool_frame",
                         "LEFT_GRIPPER": "l_gripper_tool_frame"}
    else:
        object_frames = {}
        for obj, frame in zip(args.objects, args.frames):
            object_frames[obj] = frame

    rospy.init_node('%s_proximity_monitor' % args.domain)
    monitor = ProximityMonitor(args.domain, object_frames, args.distance)
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        monitor.check_state()
        rate.sleep()
