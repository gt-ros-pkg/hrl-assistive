#!/usr/bin/env python

import itertools as it
from hrl_task_planning import pddl_utils as pddl

if __name__ == '__main__':
    domain_file = "../../params/pick_and_place.domain"
    problem_file = "../../params/pick_and_place.problem"
    initial_states = [['(AT BLOC PICK_LOC)', '(AT TARGET PLACE_LOC)']]
    initial_states = [pddl.State([pddl.Predicate.from_string(pred) for pred in state]) for state in initial_states]
    problem = pddl.Problem.from_file(problem_file)
    domain = pddl.Domain.from_file(domain_file)
    situ = pddl.Situation(domain, problem)
    all_states = situ.test_domain(initial_states)

    planner = pddl.FF("../ff")
    N = len(all_states)**2
    count = 0
    for (init, goal) in it.product(all_states, repeat=2):
        if count % 100 == 0:
            print "%f%%" % (float(count)/N*100)
        if not init.predicates:
            continue
        problem.init = init.predicates
        problem.goal = goal.predicates
        try:
            planner.solve(domain, problem)
        except pddl.PlanningException:
            print "Failed to plan:"
            print "From:", map(str, problem.init)
            print "To:", map(str, problem.goal)
            raw_input("Check")
        count += 1
