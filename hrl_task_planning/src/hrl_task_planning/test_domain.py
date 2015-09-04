#!/usr/bin/env python

from pddl_utils import PDDLObject, PDDLPredicate, PDDLProblem
from PyFF import FF


def objects_to_pddl(object_list):
    pddl_objects = []
    for obj in object_list:
        pddl_objects.append(PDDLObject(obj[0], obj[1]))
    return pddl_objects


def expand_states(state_lists):
    from functools import reduce  # for py3 compatability
    import itertools as it
    return list(reduce(it.product, state_lists))


def states_to_pddl(state_list):
    pddl_state_list = []
    for state in state_list:
        pddl_state_list.append(predicates_to_pddl(state))
    return pddl_state_list


def predicates_to_pddl(predicate_list):
    """ Takes a list of predicate specificates, returns a list of PDDLPredicate Objects. """
    # print "Predicate list:\n", predicate_list
    pddl_predicate_list = []
    for predicate in predicate_list:
        if not predicate:  # empty list -> empty
            pddl_predicate_list.append([])
        else:
            try:
                pddl_predicate_list.append(PDDLPredicate(predicate[0], predicate[1]))
            except Exception as e:
                print predicate
                raise e
    # print "Converted: ", pddl_predicate_list
    return pddl_predicate_list


def flatten(state):
    l = []
    while isinstance(state[0], tuple):
        l.extend(state[1])
        state = state[0]
    l.extend(state[0])
    return l


def test(init_states, goal_states, objects=[], constants=[]):
    """ Test complete state space for problem domain to check solutions."""
    import sys
    ntests = len(init_states)*len(goal_states)
    print "Total tests: %d" % ntests

    count = 0
    pct = -1
    for i, init in enumerate(init_states):
        init = flatten(init)
        init.extend(constants)
        for g, goal in enumerate(goal_states):
            count += 100
            if count/ntests > pct:
                pct = count/ntests
                sys.stdout.write("\r%d%%" % pct)
                sys.stdout.flush()
            goal = flatten(goal)
            prob = PDDLProblem('shaving', 'SHAVING-DOMAIN',
                               objects=objects,
                               init=init,
                               goal=goal)
            solver = FF(prob, 'shaving.domain', ff_executable="./FF-v2.3/ff")
            solver.solve()
            if isinstance(solver.solution, list):
                continue
            if not solver.solution:
                solver.print_solution()
                print "INIT:", init
                for pred in init:
                    print pred
                print "Goal:", goal
                for pred in goal:
                    print pred
                sys.exit(1)

def complete_test(domain_file):



def main():
    from shaving_states import objects, init_state_lists, goal_state_lists, constants  # modify source to test different state spaces

    objects = objects_to_pddl(objects)

    init_state_lists = [states_to_pddl(states) for states in init_state_lists]
    init_states = expand_states(init_state_lists)

    goal_state_lists = [states_to_pddl(states) for states in goal_state_lists]
    goal_states = expand_states(goal_state_lists)

    constants = predicates_to_pddl(constants)
    test(init_states, goal_states, objects, constants)
