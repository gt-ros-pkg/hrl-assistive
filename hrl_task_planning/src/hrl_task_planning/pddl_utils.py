#!/usr/bin/env python


from lis import parse
import copy


class PDDLObject(object):
    """ A class describing an Object in PDDL. """
    def __init__(self, name, type=None):
        self.name = name
        self.type = type

    def __str__(self):
        return "%s( %s )" % (self.name.upper(), self.type.upper())

    def __repr__(self):
        return self.__str__()


class PDDLPredicate(object):
    """ A class describing a predicate in PDDL. """
    def __init__(self, name, args=None, neg=False):
        self.name = name
        self.args = args
        self.neg = neg

    def __str__(self):
        msg = "NOT" if self.neg else ""
        return ' '.join([msg, "%s( %s )" % (self.name.upper(), ', '.join(self.args))])

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
        with open(domain_file, 'r') as f:
            domain = f.read()
        parsed_dom = parse(domain)
        for statement in parsed_dom:
            if statement == 'define':  # Ignore opening def statement
                continue
            elif statement[0] == 'domain':  # Get name of domain
                self.name = statement[1]
            elif statement[0] == ':requirements':
                self.requirements.extend(statement[1:])  # Get requirements
            elif statement[0] == ':types':
                self.types = self._parse_types(statement[1:])
            elif statement[0] == ':constants':
                self.objects = self._parse_constants(statement[1:])
            elif statement[0] == ':predicates':
                self.predicates = self._parse_predicates(statement[1:])
            elif statement[0] == ':action':
                name, action = self._parse_action(statement[1:])
                self.actions[name] = action

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
        s = "+"*20, " Problem ", "+"*20
        s += "\nObjects:"
        for obj in self.objects:
            s += str(obj)
        s += "\nInitial Predicates:"
        for pred in self.init:
            s += str(pred)
        s += "\nGoal Predicates:"
        for pred in self.goal:
            s += str(pred)
        return s

    def __repr__(self):
        return self.__str__()

    def _grab_inits(self, string):
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
            preds.append(PDDLPredicate(p[1        if problem_file is None:

        else:
            self.problem_file = problem_file
], p[1:]))
        return preds

    def from_file(self, filename):
        """ Load a PDDL Problem from a PDDL problem file. """
        with open(filename, 'r') as pfile:
            string = ''.join(pfile.readlines())
        string = string.upper().replace('\n', ' ')
        # Extract the problem name
        ps = string.find('PROBLEM') + 7  # after def problem
        pe = string.find(')', ps)
        problem_name = string[ps:pe].strip()
        # Extract the domain name
        ds = string.find(':DOMAIN') + 7
        de = string.find(')', ds)
        problem_domain = string[ds:de].strip()
        # Extract the objects defined for the problem
        obj_start = string.find(':OBJECTS') + 8
        oe = string.find(')', obj_start)
        objs = string[obj_start:oe].strip().split()
        pyobjs = []
        exp = 'name'
        for obj in objs:
            if obj == '-':
                exp = 'type'
                continue
            if exp == 'name':
                pyobjs.append(PDDLObject(obj))
            else:
                pyobjs[-1].type = obj
                exp = 'name'

        init_start = string.find(':INIT') + 5
        init_preds = self._grab_inits(string[init_start:])
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
        self.name = problem_name
        self.problem_file = filename
        self.problem_domain = problem_domain
        self.objects = pyobjs
        self.init = init_preds
        self.goal = goal_preds

        self.__init__(problem_name,
                      problem_domain,
                      objects=pyobjs,
                      init=init_preds,
                      goal=goal_preds)

    def to_file(self, problem, filename=None):
        """ Write a PDDL Problem file based on a PDDLProblem instance. """
        s = "(define (problem " + problem.name + ")\n"
        s += "\t(:domain " + problem.domain + ")\n"
        # Define objects
        s += "\t(:objects "
        for obj in problem.objects:
            s += ''.join(['\t', obj.name])
            if obj.type is not None:
                s += " - " + obj.type
            s += "\n"
        s += ")\n"
        # Define initial conditions
        s += "(:init "
        for pred in problem.init:
            parts = ['\t(not ', '', ')\n'] if pred.neg else ['\t', '', '\n']
            parts[1] = ' '.join(['(', pred.name, ' '.join(pred.args), ')'])
            s += ''.join(parts)
        s += ')\n'
        # Defind goal conditions
        s += "(:goal\n\t(and \n"
        for pred in problem.goal:
            parts = ['\t(not ', '', ')\n'] if pred.neg else ['\t', '', '\n']
            parts[1] = ' '.join(['(', pred.name, ' '.join(pred.args), ')'])
            s += ''.join(parts)
        s += ')))\n'
        # Write file to disk
        filename = '.'.join([problem.name, 'problem']) if filename is None else filename
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
