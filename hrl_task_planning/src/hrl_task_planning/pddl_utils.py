#!/usr/bin/env python

# from lis import parse
import copy
import re
import sys


def parse_lisp(string):
    parts = re.split('\(|\)|\n| ', string.upper())
    return [part.strip() for part in parts if part.strip()]  # Only keep each part if it isn't an empy string


def separate_list(string):
    string = string.replace(')', ' ) ')
    string = string.replace('(', ' ( ')
    return string.split()

def to_lists(items):
    item = items.pop(0)
    list_ = []
    if item == '(':
        print "open brace"
        sys.stdout.flush()
        list_.append(to_lists(items))
    elif item == ')':
        print "close brace"
        sys.stdout.flush()
        return list_
    else:
        print "append %s" % item
        sys.stdout.flush()
        list_.append(item)


def parse_problem(string):
    sep_list = separate_list(string)
    objects = []
    init = []
    goal = []
    mode = None
    level = 0
    mode_level = 0
    for i, item in enumerate(sep_list):
        if item == '(':
            level += 1
        elif item == ')':
            level -= 1
            if level < mode_level:
                mode = None
        elif item == 'DEFINE':  # ignore
          continue
        elif item == "PROBLEM":
            problem_name = sep_list[i+1]
        elif item == ":DOMAIN":
            domain_name = sep_list[i+1]
        elif item == ":OBJECTS":
            mode = ":OBJECTS"
            mode_level = level
        elif item == ":INIT":
            mode = ":INIT"
            mode_level = level
        elif item == ":GOAL":
            mode = ":GOAL"
            mode_level = level
        else:
            if mode is None:
                continue
            elif mode == ":OBJECTS":
                if sep_list[i+1] == '-':
                    objects.append(PDDLObject(item, sep_list[i+2]))
                continue
            elif mode == ":Init":
                pass








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
#        with open(domain_file, 'r') as f:
#            domain = f.read()
#        parsed_dom = parse(domain)
#        for statement in parsed_dom:
#            if statement == 'define':  # Ignore opening def statement
#                continue
#            elif statement[0] == 'domain':  # Get name of domain
#                self.name = statement[1]
#            elif statement[0] == ':requirements':
#                self.requirements.extend(statement[1:])  # Get requirements
#            elif statement[0] == ':types':
#                self.types = self._parse_types(statement[1:])
#            elif statement[0] == ':constants':
#                self.objects = self._parse_constants(statement[1:])
#            elif statement[0] == ':predicates':
#                self.predicates = self._parse_predicates(statement[1:])
#            elif statement[0] == ':action':
#                name, action = self._parse_action(statement[1:])
#                self.actions[name] = action

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
    def __init__(self, name, domain,
                 domain_file=None,
                 problem_file=None,
                 objects=[],
                 init=[],
                 goal=[]):
        self.name = name
        self.domain = domain
        self.domain_file = domain_file
        self.problem_file = '.'.join([self.name, 'problem']) if problem_file is None else problem_file
        self.objects = [i for i in objects if isinstance(i, PDDLObject)]
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
    def _grab_inits(cls, string):
        """ Extract initial predicates from a string"""
        # TODO: Doesn't catch negative initial cases
        op = 0
        for i, c in enumerate(string):
            op += c == '('
            op -= c == ')'
            if op < 0:
                init_str = string[:i]
                break
        init_str = init_str.split('(')[1:]
        preds = []
        for pred in init_str:
            p = pred.split(')')[0].split()
            preds.append(PDDLPredicate(p[1], p[1:]))
        return preds

    @classmethod
    def init_from_file(cls, filename):
        """ Load a PDDL Problem from a PDDL problem file. """
        with open(filename, 'r') as pfile:
            string = ''.join(pfile.readlines())
        return cls.init_from_string(string)

    @classmethod
    def parse_objects(cls, stringlist):
        """ Extract the objects defined for the problem."""
        assert(len(stringlist) % 3 == 0), "Error parsing constants: should be (object, [hyphen], type) triples"
        obj_list = []
        for i in range(len(stringlist)/3):
            obj_list.append(PDDLObject(stringlist[3*i], stringlist[3*i+2]))
        return obj_list

    @classmethod
    def extract_init_text(cls, string):
        """ Extract predicates from a defined list. """
        predicate_strings = []
        s = string[string.find(":INIT"):]
        open_idx = s.find("(")
        close_idx = s.find("(")
        if open_idx < close_idx:
            pass




        i = 0
        open_level = 1
        while open_level > 0:
            blockstart = None
            print i, s[i]
            if s[i] == "(":
                if blockstart is None:
                    blockstart = i
                else:
                    open_level += i
            elif s[i] == ")":
                predicate_strings.append(string[blockstart:i])
                open_level -= 1
                blockstart = None
            i += 1
        return predicate_strings



    @classmethod
    def init_from_string(cls, string):
        string = string.upper()
        parts = parse_lisp(string)
        parts.remove("DEFINE")  # Get rid of the def statement
        # Extract problem name
        problem_idx = parts.index("PROBLEM")
        problem_name = parts[problem_idx+1]
        parts.pop(problem_idx)  # Removes problem statement
        parts.pop(problem_idx)  # Removes problem name argument (falls back to vacated index after first pop)
        # Extract domain name
        dom_idx = parts.index(":DOMAIN")
        problem_domain = parts[dom_idx + 1]
        parts.pop(dom_idx)  # Removes domain statement
        parts.pop(dom_idx)  # Removes domain name argument (falls back to vacated index after first pop)
        # Extract components
        obj_idx = string.find(":OBJECTS")
        init_idx = string.find(":INIT")
        goal_idx = string.find(":GOAL")
        blocks = {obj_idx: ":OBJECTS",
                  init_idx: ":INIT",
                  goal_idx: ":GOAL"}
        keys = blocks.keys()
        keys.sort()
        strings = {}
        strings[blocks[keys[0]]] = parts[keys[0]+1:keys[1]]
        strings[blocks[keys[1]]] = parts[keys[1]+1:keys[2]]
        strings[blocks[keys[2]]] = parts[keys[2]+1:]
        objects = cls.parse_objects(groups[":objects"])
        init = cls.parse_predicates(groups[":init"])
        goal = cls.parse_predicates(groups[":goal"])


        init_start = string.find(':INIT') + 5
        init_preds = cls._grab_inits(string[init_start:])
        # Extract the goal states
        goal_start = string.find(':GOAL') + 5
        goal_string = string[string.find('AND', goal_start) + 3:].strip()
        goal_items = [l.strip().split(')')[0] for l in goal_string.split('(')[1:]]
        neg = False
        goal_preds = []
        for entry in goal_items:
            if entry == 'NOT':
                neg = True
                continue
            p = entry.split()
            goal_preds.append(PDDLPredicate(p[0], p[1:], neg=neg))
            neg = False
        return cls(problem_name, problem_domain, objects=pyobjs, init=init_preds, goal=goal_preds)

    def to_file(self, filename=None):
        """ Write a PDDL Problem file based on a PDDLProblem instance. """
        s = "(define (problem " + self.name + ")\n"
        s += "\t(:domain " + self.domain + ")\n"
        # Define objects
        s += "\t(:objects "
        for obj in self.objects:
            s += ''.join(['\t', obj.name])
            if obj.type is not None:
                s += " - " + obj.type
            s += "\n"
        s += ")\n"
        # Define initial conditions
        s += "(:init "
        for pred in self.init:
            parts = ['\t(not ', '', ')\n'] if pred.neg else ['\t', '', '\n']
            parts[1] = ' '.join(['(', pred.name, ' '.join(pred.args), ')'])
            s += ''.join(parts)
        s += ')\n'
        # Defind goal conditions
        s += "(:goal\n\t(and \n"
        for pred in self.goal:
            parts = ['\t(not ', '', ')\n'] if pred.neg else ['\t', '', '\n']
            parts[1] = ' '.join(['(', pred.name, ' '.join(pred.args), ')'])
            s += ''.join(parts)
        s += ')))\n'
        # Write file to disk
        filename = '.'.join([self.name, 'problem']) if filename is None else filename
        with open(filename, 'w') as prob_file:
            prob_file.write(s)


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
