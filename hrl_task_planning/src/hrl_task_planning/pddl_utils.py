#!/usr/bin/env python

import copy


def _separate_string(string):
    string = string.replace(')', ' ) ')
    string = string.replace('(', ' ( ')
    return string.split()


def _to_lists(items):
    list_ = []
    while items:
        item = items.pop(0)
        if item == '(':
            list_.append(_to_lists(items))
        elif item == ')':
            return list_
        else:
            list_.append(item)
    return list_[0]  # return final list at end (if we run out of items), remove extra enclosure


def lisp_to_list(string):
    return _to_lists(_separate_string(string))


def get_sublist(lists, tag):
    for list_ in lists:
        if tag == list_[0]:
            return list_


class PDDLObject(object):
    """ A class describing an Object in PDDL. """
    @classmethod
    def init_from_string(cls, string):
        """ Create a PDDLObject instance from a formatted string."""
        string = string.strip('( )')
        name, type_ = string.split('-')
        return cls(name.strip(), type_.strip())

    def __init__(self, name, type=None):
        self.name = name
        self.type = type

    def __str__(self):
        return "%s - %s" % (self.name.upper(), self.type.upper())

    def __repr__(self):
        return self.__str__()


class PDDLPredicate(object):
    """ A class describing a predicate in PDDL. """
    @classmethod
    def init_from_string(cls, string):
        """ Create a PDDLPredicate instance from a formatted string."""
        string = string.strip('( )')
        assert string.count('(') <= 1, "Badly formed predicate string.  Too many opening parentheses"
        string = string.upper()
        parts = string.split('(')
        neg = False
        if len(parts) > 1:
            neg = parts[0]
            parts = parts[1:]
            neg = True if 'NOT' == neg.strip() else False
        name_args = parts[0].split()
        name, args = name_args[0], name_args[1:]
        return cls(name, args, neg)

    def __init__(self, name=None, args=None, neg=False):
        self.name = name
        self.args = args
        self.neg = neg

    def __str__(self):
        msg = "(%s %s)" % (self.name, ' '.join(self.args))
        if self.neg:
            msg = ''.join(["( NOT ", msg, ")"])
        return msg.upper()

    def __repr__(self):
        return self.__str__()


class PDDLAction(object):
    """ A class describing an action in PDDL. """
    def __init__(self, name, parameters, preconditions, effects):
        self.name = name
        self.parameters = parameters
        self.preconditions = preconditions
        self.effects = effects


class PDDLDomain(object):
    """ A class describing a domain instance in PDDL."""
    def __init__(self, name):
        self.name = name
        self.requirements = []
        self.types = {}
        self.predicates = {}
        self.actions = {}

    def _parse_types(self, typeslist):
        types = {}
        while len(typeslist) > 0:
            try:
                idx = typeslist.index('-')
            except ValueError:
                return
            types[copy.deepcopy(typeslist[idx + 1])] = copy.deepcopy(typeslist[0:idx])  # Items before hyphen are of type listed after hyphen
            typeslist = typeslist[idx + 2:]
        return types

    def _parse_constants(self, const_list):
        assert(len(const_list) % 3 == 0), "Error parsing constants: should be (object, [hyphen], type) triples"
        objects = []
        n_consts = len(const_list) / 3
        for n in range(n_consts):
            objects.append(PDDLObject(const_list[n*3], const_list[n*3+2]))
        return objects

    def _parse_predicates(self, pred_list):
        predicates = {}
        for pred in pred_list:
            args = []
            for i in range(pred.count('-')):
                idx = pred.index('-')
                args.append(pred[idx+1])
                pred.pop(idx)
            predicates[pred[0]] = PDDLPredicate(pred[0], args)
        return predicates

    def _parse_action(self, act):
        # NOT IMPLEMENTED: Not required for producing combinations of possilbe states for testing cases.
        return act[0], None

    def from_file(self, domain_file):
        raise NotImplementedError
        # TODO: Tested using content from Peter Norvig's lis.py, should find/make own replacement

    def _get_constants_by_type(self):
        type_dict = {}
        type_set = set(self.types.iterkeys())
        for subtypes in self.types.itervalues():
            [type_set.add(subtype) for subtype in subtypes]
        for type_, objs in self.types.iteritems():
            type_dict[type_] = objs
            for obj in objs:
                if obj in self.types:
                    type_dict[type_].extend(self.types[obj])  # Add items of sub-class
                    type_dict[type_].pop(type_dict[type_].index(obj))  # Remove sub-class itself
        objs_dict = {}
        for type_ in type_set:
            objs_dict[type_] = []
            for obj in self.objects:
                if obj.type == type_:
                    objs_dict[type_].append(obj)
                if (type_ in type_dict and obj.type in type_dict[type_]):
                    objs_dict[type_].append(obj)
        return objs_dict


class PDDLProblem(object):
    """ A class describing a problem instance in PDDL. """
    def __init__(self, name, domain, objects=[], init=[], goal=[]):
        self.name = name
        self.domain = domain
        self.objects = [i for i in objects if isinstance(i, PDDLObject)]  # Make sure type is correct
        self.init = [i for i in init if isinstance(i, PDDLPredicate)]
        self.goal = [i for i in goal if isinstance(i, PDDLPredicate)]

    def __str__(self):
        title = "PDDL Problem: %s" % self.name
        domain = "PDDL Domain: %s" % self.domain
        objects_title = "Objects:"
        objects = "\t"+"\n\t".join(map(str, self.objects))
        predicates_title = "Initial Predicates:"
        predicates = "\t"+"\n\t".join(map(str, self.init))
        goal_title = "Goal Predicates:"
        goal = "\t"+"\n\t".join(map(str, self.goal))
        return '\n'.join([title, domain, objects_title, objects, predicates_title, predicates, goal_title, goal])

    def __repr__(self):
        return self.__str__()

    @classmethod
    def init_from_file(cls, filename):
        """ Load a PDDL Problem from a PDDL problem file. """
        with open(filename, 'r') as pfile:
            string = ''.join(pfile.readlines())
        return cls.init_from_string(string)

    @classmethod
    def init_from_string(cls, string):
        data = lisp_to_list(string.upper())
        problem_name = get_sublist(data, 'PROBLEM')[1]
        domain_name = get_sublist(data, ':DOMAIN')[1]
        objects = cls._parse_objects(get_sublist(data, ":OBJECTS")[1:])  # Remove first entry, which is the tag just found
        init = cls._parse_init(get_sublist(data, ":INIT")[1:])
        goal = cls._parse_goal(get_sublist(data, ":GOAL")[1:][0])  # also remove spare wrapping list...
        return cls(problem_name, domain_name, objects, init, goal)

    @classmethod
    def _parse_objects(cls, item_list):
        """ Extract the objects defined for the problem."""
        assert(len(item_list) % 3 == 0), "Error parsing constants: should be (object, [hyphen], type) triples"
        objs = []
        for i in range(len(item_list)/3):
            objs.append(PDDLObject(item_list[3*i], item_list[3*i+2]))
        return objs

    @classmethod
    def _parse_init(cls, items):
        """ Extract predicates from a defined list. """
        return [PDDLPredicate(item[0], item[1:]) for item in items]

    @classmethod
    def _parse_goal(cls, items):
        preds = []
        for item in items:
            if item == 'AND':
                continue
            elif item[0] == 'NOT':
                preds.append(PDDLPredicate(item[1][0], item[1][1:], True))
            else:
                preds.append(PDDLPredicate(item[0], item[1:]))
        return preds

    def to_file(self, filename=None):
        """ Write a PDDL Problem file based on a PDDLProblem instance. """
        filename = '.'.join([self.name, 'problem']) if filename is None else filename
        s = self.to_string()
        with open(filename, 'w') as prob_file:
            prob_file.write(s)

    def to_string(self):
        """ Write a PDDL Problem as a string in PDDL File style """
        s = "(DEFINE\n(PROBLEM " + self.name + ")\n"
        s += "(:DOMAIN " + self.domain + ")\n"
        # Define objects
        s += "(:OBJECTS "
        s += '\n\t'.join([str(obj) for obj in self.objects])
        s += ")\n"
        # Define initial conditions
        s += "(:INIT "
        s += '\n\t'.join(map(str, self.init))
        s += ')\n'
        # Defind goal conditions
        s += "(:GOAL\n\t(AND \n"
        s += '\n\t'.join(map(str, self.goal))
        s += ')))\n'
        return s


class Planner(object):
    """ Base class for planners to solve PDDL problems. """
    def __init__(self, problem, domain_file):
        self.domain_file = domain_file
        self.problem = problem
        self.solution = None

    def solve(self):
        raise NotImplementedError()

    def print_solution(self):
        """ Print solution steps. """
        if self.solution is None:
            print "This problem has not been solved yet."
        elif self.solution == []:
            print "Result: Initial State satisfies the Goal State"
        elif not self.solution:
            print "Result: FF failed to find a solution"
        else:
            print "Result:\n\tPlan:"
            for step in self.solution:
                args = ', '.join(step['args'])
                print ''.join(["\t", step['act'], "(", args, ")"])
