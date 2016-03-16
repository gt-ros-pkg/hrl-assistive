#!/usr/bin/env python

import sys
import argparse

import rospy
from hrl_task_planning import pddl_utils as pddl
from hrl_task_planning.msg import PDDLState


class DomainStateAggregator(object):
    def __init__(self, domain):
        """ Subscribes to state updates from detectors and publishes a latched complete domain state, with constant predicates."""
        self.domain = domain
        while not rospy.has_param("/pddl_tasks/%s/constant_predicates" % self.domain):
            rospy.loginfo("[%s] Waiting for constant predicates of %s domain.", rospy.get_name(), self.domain)
            rospy.sleep(1)
        self.constant_predicates = rospy.get_param("/pddl_tasks/%s/constant_predicates" % self.domain)
        if self.constant_predicates:
            self.state = pddl.State(map(pddl.Predicate.from_string, self.constant_predicates))
        else:
            self.state = pddl.State()
        self.state_pub = rospy.Publisher("/pddl_tasks/%s/state" % self.domain, PDDLState, queue_size=10, latch=True)
        self.update_sub = rospy.Subscriber("/pddl_tasks/%s/state_updates" % self.domain, PDDLState, self.update_cb)
        rospy.loginfo("[%s] Aggregator for %s domain is ready.", rospy.get_name(), self.domain)
        rospy.Timer(rospy.Duration(1.5), self.publish_state, oneshot=True)  # Send initial state to latched topic

    def update_cb(self, state_update_msg):
        """ Receive updates, verify that they are for this domain, update the full state, and publish."""
#        if not state_update_msg.domain == self.domain:
#            rospy.logwarn("[%s] State Aggregator for %s domain received state update for %s domain",
#                          rospy.get_name(), self.domain, state_update_msg.domain)
#            return
        if not state_update_msg.predicates:
            return  # Empty list, no updates (shouldn't really receive these...)
        # Apply updates
        for pred_str in state_update_msg.predicates:
            self.state.add(pddl.Predicate.from_string(pred_str))
        # Publish updates full state
        self.publish_state()

    def publish_state(self, event=None):
        state_msg = PDDLState()
        state_msg.domain = self.domain
        state_msg.predicates = self.state.string_list()
        self.state_pub.publish(state_msg)


def main():
    rospy.init_node('domain_state_aggregators')
    parser = argparse.ArgumentParser(description="Aggregate and publish full domain state information.")
    parser.add_argument('--domains', '-d', nargs='+', help="The domain(s) to aggregate data and publish state.")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    aggregators = {}
    for dom in args.domains:
        aggregators[dom] = DomainStateAggregator(dom)
    rospy.spin()
