#!/usr/bin/env python

import sys
import argparse

import rospy
from hrl_task_planning import pddl_utils as pddl
from hrl_task_planning.msg import PDDLState


class ParameterMonitor(object):
    def __init__(self, domain, predicate, args):
        self.domain = domain
        self.predicate = predicate
        self.args = args
        self.params = ["/pddl_tasks/%s/%s/%s" % (self.domain, self.predicate, arg) for arg in self.args]
        self.state = []
        self.state_update_pub = rospy.Publisher('/pddl_tasks/%s/state_update' % self.domain, PDDLState)

    def run(self):
        pub = False
        for arg in self.args:
            pred = pddl.Predicate(self.predicate, arg)
            if rospy.has_param("/pddl_tasks/%s/%s/%s" % (self.domain, self.predicate, arg)):
                if pred not in self.state:
                    self.state.append(pred)
                    pub = True
            else:
                try:
                    self.state.remove(pred)
                    pub = True
                except ValueError:
                    pass
        if pub:
            msg = PDDLState()
            msg.domain = self.domain
            msg.predicates = self.state
            self.state_update_pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(description="Monitor parameters representing changes in a pddl domain state.")
    parser.add_argument('domain', help="The domain for which the parameter will be monitored.")
    parser.add_argument('predicate', help="The name of the predicate being monitored.")
    parser.add_argument('--args', '-a', nargs="*", default=[], help="The possible arguments of the predicate to be monitored.")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node('%s_%s_param_monitor' % (args.domain, args.predicate))
    monitor = ParameterMonitor(args.domain, args.predicate, args.args)
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        monitor.run()
        rate.sleep()
