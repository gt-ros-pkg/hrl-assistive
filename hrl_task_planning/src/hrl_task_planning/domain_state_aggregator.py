#!/usr/bin/env python

import sys
import argparse

import rospy
from hrl_task_planning import pddl_utils as pddl
from hrl_task_planning.msg import PDDLState


class DomainStateAggregator(object):
    def __init__(self, domain, predicates):
        """ Subscribes to state updates from detectors and publishes a latched complete domain state, with constant predicates."""
        self.domain = domain
        self.predicates = predicates
        # Load axioms (constantly true predicates) from yaml/param server
        while not rospy.has_param("/pddl_tasks/%s/constant_predicates" % self.domain):
            rospy.loginfo("[%s] Waiting for constant predicates of %s domain.", rospy.get_name(), self.domain)
            rospy.sleep(1)
        self.constant_predicates = rospy.get_param("/pddl_tasks/%s/constant_predicates" % self.domain)
        if self.constant_predicates:
            self.state = pddl.State(map(pddl.Predicate.from_string, self.constant_predicates))
        else:
            self.state = pddl.State()
        # Setup subscriber for state_updates
        self.state_pub = rospy.Publisher("/pddl_tasks/%s/state" % self.domain, PDDLState, queue_size=10, latch=True)
        self.update_sub = rospy.Subscriber("/pddl_tasks/state_updates", PDDLState, self.update_cb)
        rospy.loginfo("[%s] Domain State Aggregator Ready: %s - %s", rospy.get_name(), self.domain, ', '.join(self.predicates))
        rospy.Timer(rospy.Duration(1.5), self.publish_state, oneshot=True)  # Send initial state to latched topic

    def update_cb(self, state_update_msg):
        """ Receive updates, verify that they are for this domain, update the full state, and publish."""
        if not state_update_msg.predicates:
            return  # Empty list, no updates (shouldn't really receive these...)
        # Apply updates
        new_preds = [pddl.Predicate.from_string(pred) for pred in state_update_msg.predicates]
        publish = False
        for pred in new_preds:
            if pred.name in self.predicates:
                if not pred.neg and pred not in self.state:
                    self.state.add(pred)
                    publish = True
                if pred.neg:
                    test = pddl.Predicate(pred.name, pred.args)
                    if test in self.state:
                        self.state.add(pred)
                        publish = True
        if publish:
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
    for domain in args.domains:
        try:
            domain_str = rospy.get_param('/pddl_tasks/%s/domain' % domain)
            domain_obj = pddl.Domain.from_string(domain_str)
            domain_predicates = domain_obj.predicates.keys()
        except KeyError:
            rospy.logerr("[%s] Cannot load domain file for domain: %s", rospy.get_name(), domain)
        aggregators[domain] = DomainStateAggregator(domain, domain_predicates)
    rospy.spin()
