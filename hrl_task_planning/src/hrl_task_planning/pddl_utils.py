#!/usr/bin/env python


def _separate_string(string):
    """ Space out parentheses and split to separate items in a lisp string."""
    string = string.replace(')', ' ) ')
    string = string.replace('(', ' ( ')
    return string.split()


def _to_lists(items):
    """ Create list structure mimicing the structure of a lisp string already separated by parentheses."""
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
    """ Create list structure mimicing the structure of a lisp string """
    return _to_lists(_separate_string(string))


def get_sublist(lists, tag):
    """ Get a sub-list from a list of lists based on the string in the first entry of the list. """
    for list_ in lists:
        if tag == list_[0]:
            return list_


class PDDLType(object):
    """ A class describing a type in PDDL."""
    def __init__(self, name, supertype=None):
        self.name = name
        self.supertype = supertype

    def __str__(self):
        if self.is_subtype():
            return " - ".join([self.name, self.supertype.name])
        else:
            return self.name

    def is_subtype(self):
        """ Check if this type has supertype."""
        return bool(self.supertype)

    def is_type(self, check_type):
        """ Check if this type is a type of subtype of the given type."""
        if self.name == check_type:
            return True
        elif self.is_subtype():
            return self.supertype.is_type(check_type)
        else:
            return False


class PDDLObject(object):
    """ A class describing an Object in PDDL. """
    def __init__(self, name, type=None):
        self.name = name
        self.type = type

    def __str__(self):
        if self.type is None:
            return self.name.upper()
        else:
            return "%s - %s" % (self.name.upper(), self.type.upper())

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and (self.__dict__ == other.__dict__))

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_variable(self):
        """ Check if this object defines a variable (True), or a specific object instance (False)."""
        return bool(self.name[0] == '?')

    @classmethod
    def from_string(cls, string):
        """ Create a PDDLObject instance from a formatted string."""
        string = string.strip('( )')
        name, type_ = string.split(' - ')
        return cls(name.strip(), type_.strip())


class PDDLPredicate(object):
    """ A class describing a predicate in PDDL. """
    def __init__(self, name, args=[], neg=False):
        self.name = name
        self.args = []
        self.neg = neg
        for arg in args:
            if isinstance(arg, PDDLObject):
                self.args.append(arg)
            elif isinstance(arg, str):
                self.args.append(PDDLObject.from_string(arg))

    def __str__(self):
        msg = "(%s %s)" % (self.name, ' '.join(map(str, self.args)))
        if self.neg:
            msg = ''.join(["( NOT ", msg, ")"])
        return msg.upper()

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and (self.__dict__ == other.__dict__))

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_abstract(self):
        """ Check if this is an abstract predicate definition (True), or a specific Predicate statement (False)."""
        return any([bool(arg.name[0] == '?') for arg in self.args])

    @classmethod
    def from_string(cls, string):
        """ Create a PDDLPredicate instance from a formatted string."""
        return cls.from_list(lisp_to_list(string))

    @classmethod
    def from_list(cls, pred_list):
        """ Create a PDDLPredicate instance from a separated list of items."""
        neg = False
        if pred_list[0] == 'NOT':
            neg = True
            pred_list = pred_list[1]
        name = pred_list[0]
        if '-' in pred_list:
            name_type_pairs = pred_list[1:].count('-')
            args = []
            for i in range(name_type_pairs):
                args.append(PDDLObject(pred_list[1:][3*i], pred_list[1:][3*i+2]))
        else:
            args = [PDDLObject(arg) for arg in pred_list[1:]]
        res = cls(name, args, neg)
        return res


class PDDLPlanStep(object):
    """ A class specifying a PDDL action and the parameters with which to call apply it. """
    def __init__(self, name, args=[]):
        self.name = name
        self.args = args

    @classmethod
    def from_string(cls, string):
        """ Create a PDDLPlanStep from a formatted string."""
        name, args = lisp_to_list(string)
        return cls(name, args)

    def __str__(self):
        return ''.join(["(", self.name, "(", ' '.join(self.args), "))"]).upper()


class ActionException(Exception):
    """ Exception raised when action cannot be executed from a given state."""
    pass


class PDDLAction(object):
    """ A class describing an action in PDDL. """
    def __init__(self, name, parameters=[], preconditions=[], effects=[]):
        self.name = name
        self.parameters = parameters
        self.preconditions = preconditions
        self.effects = effects

    def __str__(self):
        string = ''.join(["(:ACTION ", self.name, '\n'])
        string += ":PARAMETERS ("
        string += ' '.join(map(str, self.parameters))
        string += ')\n'
        if self.preconditions:
            string += ":PRECONDITION "
            if len(self.preconditions) > 1:
                string += '(AND '
            for precond in self.preconditions:
                string += self._precondition_str(precond)
                string += '\n'
            if len(self.preconditions) > 1:
                string += ')'
            string += '\n'
        string += ":EFFECT  "
        string += self._effects_str(self.effects)
        string += ")\n"
        return string

    @classmethod
    def from_string(cls, string):
        """ Create a PDDLAction from a formatted string."""
        act = lisp_to_list(string)
        act = act[1:] if act[0] == ":ACTION" else act
        return cls.from_list(act)

    @classmethod
    def from_list(cls, act):
        """ Create a PDDLAction from a formatted list of items."""
        name = act[0]
        preconditions = []
        params = []
        effects = []
        try:  # Evaluate parameters passed to action
            param_list = act[act.index(':PARAMETERS') + 1]
            for i in range(len(param_list)/3):
                params.append(PDDLObject(param_list[3*i], param_list[3*i+2]))
        except ValueError:
            pass
        try:  # Evaluate Preconditions
            precond_list = act[act.index(":PRECONDITION") + 1]
            precond_list = precond_list[1:] if precond_list[0] == 'AND' else precond_list  # Ignore initial AND
            for cond in precond_list:
                if cond[0] == 'FORALL':
                    param = PDDLObject(cond[1][0], cond[1][2])
                    pred = PDDLPredicate.from_list(cond[2])
                    preconditions.append(['FORALL', param, pred])
                else:
                    preconditions.append(PDDLPredicate.from_list(cond))
        except ValueError:
            pass
        try:
            effect_list = act[act.index(":EFFECT") + 1]
            effects = cls._parse_effect(effect_list)
        except ValueError, e:
            raise e
        return cls(name, params, preconditions, effects)

    @classmethod
    def _parse_effect(cls, effect):
        """ Parse the effects of an action from a formatted list."""
        if effect[0] == 'AND':
            return [cls._parse_effect(eff) for eff in effect[1:]]
        elif effect[0] == 'FORALL':
            param = PDDLObject(effect[1][0], effect[1][2])
            pred = cls._parse_effect(effect[2])
            return ['FORALL', param, pred]
        elif effect[0] == 'WHEN':
            param = PDDLPredicate.from_list(effect[1])
            pred = cls._parse_effect(effect[2])
            return ['WHEN', param, pred]
        else:
            return ["PREDICATE", PDDLPredicate.from_list(effect)]

    def _effects_str(self, effect):
        """ Produce a properly formatted string of the effects of an action instance."""
        if effect[0] == 'FORALL':
            return ''.join(["(FORALL (", str(effect[1]), ") ", self._effects_str(effect[2]), ")"])
        elif effect[0] == 'WHEN':
            return ''.join(["(WHEN ", str(effect[1]), self._effects_str(effect[2]), ")"])
        elif effect[0] == 'PREDICATE':
            return str(effect[1])
        elif len(effect) == 1:
            return self._effects_str(effect[0])
        else:
            return ''.join(["(AND ", '\n'.join([self._effects_str(eff) for eff in effect]), ")"])

    def _precondition_str(self, precond):
        """ Produce a properly formatted string for a precondition of an action instance."""
        if isinstance(precond, PDDLPredicate):
            return str(precond)
        elif precond[0] == 'FORALL':
            return ''.join(['(FORALL (', str(precond[1]), ') ', str(precond[2]), ")"])


class PDDLDomain(object):
    """ A class describing a domain instance in PDDL."""
    def __init__(self, name, requirements=[], types={}, constants=[], predicates={}, actions={}):
        self.name = name
        self.requirements = requirements
        self.types = types
        self.constants = constants
        self.predicates = predicates
        self.actions = actions

    def __str__(self):
        string = "(DEFINE (DOMAIN %s)\n\n" % self.name
        string += "(:REQUIREMENTS %s)\n\n" % ' '.join(self.requirements)
        types = [t for t in self.types.itervalues() if t.is_subtype()]  # Put all sub-types up front...
        for t in self.types.itervalues():  # ...and all supertypes at the end of the list
            if t not in types:
                types.append(t)
        string += "(:TYPES\n%s)\n\n" % '\n'.join(map(str, types))
        string += "(:CONSTANTS\n%s)\n\n" % '\n'.join(map(str, self.constants))
        string += "(:PREDICATES\n%s)\n\n" % '\n'.join(map(str, self.predicates.itervalues()))
        string += '\n\n'.join(map(str, self.actions.itervalues()))
        string += ")"
        return string

    def check_problem(self, problem):
        """ Verify that the problem can be applied to this domain."""
        if problem.domain != self.name:
            print "Problem-specified domain (%s) does not match this domain (%s)" % (problem.domain, self.name)
            return False
        for obj in problem.objects:
            if obj.type not in self.types:
                print "Problem contains item (%s) of unknown type (%s). Domain types are: (%s)" % (obj, obj.type, self.types.keys())
                return False
        for pred in problem.init:
            if pred.name not in self.predicates:
                print "Problem INIT specifies unknown predicate (%s). Domain predicates are: (%s) " % (pred, self.predicates.keys())
                return False
        for pred in problem.goal:
            if pred.name not in self.predicates:
                print "Problem GOAL specifies unknown predicate (%s). Domain predicates are: (%s) " % (pred, self.predicates.keys())
                return False
        return True  # Everything looks good. We can try to solve this problem in this domain

    @classmethod
    def _parse_objects(cls, item_list):
        """ Extract the objects defined for the problem."""
        assert(len(item_list) % 3 == 0), "Error parsing constants: should be (object, [hyphen], type) triples"
        objs = []
        for i in range(len(item_list)/3):
            objs.append(PDDLObject(item_list[3*i], item_list[3*i+2]))
        return objs

    @classmethod
    def _parse_types(cls, types_list):
        """ Extract heirarchical PDDLTypes from a formatted list. """
        types = {}
        # Catch simple case of no sub-types
        if '-' not in types_list:
            for t in types_list:
                types[t] = PDDLType(t)
            return types
        # Split super-types and sub-types
        type_set = set(types_list)
        type_set.discard('-')  # Get rid of hyphen if present
        chunks = ' '.join(types_list)
        chunks = chunks.split(' - ')
        subtypes = [chunks.pop(0).split()]
        supertypes = []
        added = []
        for group in chunks:
            sp = group.split()
            supertypes.append(sp[0])  # The first item after the hypen is the supertype of the preceeding types
            subtypes.append(sp[1:])  # The remaining objects are their own types, or subtypes of the next super...
        # If more sub-type groups than supertypes, add extra sub-types now
        if len(subtypes) > len(supertypes):
            spare_types = subtypes.pop()  # can only be one extra set of un-classed types, remove them
            for t in spare_types:
                types[t] = PDDLType(t)
                added.append(t)
        # Deal with the rest of the mess
        subtype_list = [typ for group in subtypes for typ in group]
        for t in type_set:
            if (t in supertypes) and (t not in subtype_list):  # Start with top-level types
                types[t] = PDDLType(t)
                added.append(t)
        while not set(added) == type_set:
            for supertype in added:
                try:
                    ind = supertypes.index(supertype)
                except ValueError:
                    continue
                subs = subtypes[ind]
                for sub in subs:
                    types[sub] = PDDLType(sub, types[supertype])
                    added.append(sub)
        return types

    @classmethod
    def _parse_predicates(cls, pred_list):
        """ Produce a dict of PDDLPreciates from a list of predicate definition lists."""
        preds = {}
        for pred in pred_list:
            preds[pred[0]] = PDDLPredicate.from_list(pred)
        return preds

    @classmethod
    def from_file(cls, domain_file):
        """ Produce a PDDLDomain object from the specified PDDL domain file."""
        with open(domain_file, 'r') as f:
            string = f.read()
        return cls.from_string(string)

    @classmethod
    def from_string(cls, string):
        """ Produce a PDDLDomain object from a PDDL domain file string."""
        items = lisp_to_list(string.upper())
        ind = items.index('DEFINE')
        items.pop(ind)
        domain_name = get_sublist(items, "DOMAIN")[1]
        domain_requirements = get_sublist(items, ":REQUIREMENTS")[1:]
        domain_types = cls._parse_types(get_sublist(items, ":TYPES")[1:])
        constants = cls._parse_objects(get_sublist(items, ":CONSTANTS")[1:])
        predicates = cls._parse_predicates(get_sublist(items, ":PREDICATES")[1:])
        actions_list = [item[1:] for item in items if item[0] == ':ACTION']
        actions = {}
        for action in actions_list:
            actions[action[0]] = PDDLAction.from_list(action)  # create dict of actions by name
        return cls(domain_name, domain_requirements, domain_types, constants, predicates, actions)

    def to_file(self, filename):
        """ Write this PDDLDomain as a properly-formatted domain file."""
        with open(filename, 'w') as f:
            f.write(str(self))

#    def _get_constants_by_type(self):
#        # TODO Check to make sure this still works!?
#        type_dict = {}
#        type_set = set(self.types.iterkeys())
#        for subtypes in self.types.itervalues():
#            [type_set.add(subtype) for subtype in subtypes]
#        for type_, objs in self.types.iteritems():
#            type_dict[type_] = objs
#            for obj in objs:
#                if obj in self.types:
#                    type_dict[type_].extend(self.types[obj])  # Add items of sub-class
#                    type_dict[type_].pop(type_dict[type_].index(obj))  # Remove sub-class itself
#        objs_dict = {}
#        for type_ in type_set:
#            objs_dict[type_] = []
#            for obj in self.objects:
#                if obj.type == type_:
#                    objs_dict[type_].append(obj)
#                if (type_ in type_dict and obj.type in type_dict[type_]):
#                    objs_dict[type_].append(obj)
#        return objs_dict


class PDDLProblem(object):
    """ A class describing a problem instance in PDDL. """
    def __init__(self, name, domain, objects=[], init=[], goal=[]):
        self.name = name
        self.domain = domain
        self.objects = objects
        self.init = init
        self.goal = goal

    def __str__(self):
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

    @classmethod
    def from_msg(cls, msg):
        objects = [PDDLObject.from_string(obj_str) for obj_str in msg.objects]
        init = [PDDLObject.from_string(pred) for pred in msg.init]
        goal = [PDDLPredicate.from_string(pred) for pred in msg.goal]
        return cls(msg.name, msg.domain, objects, init, goal)

    @classmethod
    def from_file(cls, filename):
        """ Load a PDDL Problem from a PDDL problem file. """
        with open(filename, 'r') as f:
            string = f.read()
        return cls.from_string(string.upper())

    @classmethod
    def from_string(cls, string):
        data = lisp_to_list(string)
        problem_name = get_sublist(data, 'PROBLEM')[1]
        domain_name = get_sublist(data, ':DOMAIN')[1]
        objects = cls._parse_objects(get_sublist(data, ":OBJECTS")[1:])  # Remove first entry, which is the tag just found
        init = cls._parse_init(get_sublist(data, ":INIT")[1:])
        goal = cls._parse_goal(get_sublist(data, ":GOAL")[1:][0])  # also remove spare wrapping list...
        return cls(problem_name, domain_name, objects, init, goal)

    @classmethod
    def _parse_objects(cls, item_list):
        """ Extract the objects defined for the problem."""
        objs = []
        while item_list:
            try:
                hyphen_ind = item_list.index('-')
                supertype = item_list[hyphen_ind + 1]
            except ValueError:
                supertype = None
            obj = item_list.pop(0)
            if obj == '-':
                item_list.pop(0)  # discard supertype name which comes after hyphen
                continue
            objs.append(PDDLObject(obj, supertype))
        return objs

    @classmethod
    def _parse_init(cls, items):
        """ Extract predicates from a defined list. """
        return [PDDLPredicate(item[0], [PDDLObject(name) for name in item[1:]]) for item in items]

    @classmethod
    def _parse_goal(cls, items):
        preds = []
        for item in items:
            if item == 'AND':
                continue
            elif item[0] == 'NOT':
                preds.append(PDDLPredicate(item[1][0], [PDDLObject(name) for name in item[1][1:]], True))
            else:
                preds.append(PDDLPredicate(item[0], [PDDLObject(name) for name in item[1:]]))
        return preds

    def to_file(self, filename=None):
        """ Write a PDDL Problem file based on a PDDLProblem instance. """
        filename = '.'.join([self.name, 'problem']) if filename is None else filename
        with open(filename, 'w') as prob_file:
            string = str(self)
            prob_file.write(string)


class PDDLSituation(object):
    def __init__(self, domain, problem):
        if not domain.check_problem(problem):
            raise RuntimeError("Problem cannot be applied to this domain.")
        self.domain = domain
        self.problem = problem
        self.solution = self.solve()
        self.objects = self._merge_objects(domain, problem)

    def _get_object_type(self, obj):
        for known_object in self.objects:
            if obj == known_object.name:
                return known_object.type
        return None

    def _get_objects_of_type(self, type_):
        objs = []
        for obj in self.objects:
            if self.domain.types[obj.type].is_type(type_):
                objs.append(obj)
        return objs

    def _merge_objects(self, domain, problem):
        objs = domain.constants
        for obj in problem.objects:
            if obj not in objs:
                objs.append(obj)
        return objs

    def _apply_effect(self, effect, state):
        pass

    def _expand_conditions(self, action, args):
        """ Create specific predicates for all preconditions of an action."""
        arg_map = self._resolve_args(action, args)
        condition_predicates = []
        for cond in action.preconditions:
            if isinstance(cond, PDDLPredicate):
                condition_predicates.append(PDDLPredicate(cond.name, [arg_map[arg] for arg in cond.args], cond.neg))
            else:
                for obj in self._get_objects_of_type(cond[1].type):
                    condition_predicates.append(PDDLPredicate(cond[3].name, [

    def _check_precondition(self, condition_preds,  state_preds):
        """ Make sure that the initial state to which the action is being applied meets the required preconditions."""
        print condition
        if isinstance(condition, PDDLPredicate):
            print "Checking Predicate"
            if condition not in state:
                return False
        elif condition[0] == 'FORALL':
            print "Recursing over forall"
            args = self._get_objects_type(condition[1].type)
            return self._check_preconditions(self, condition[2])

    def _resolve_args(self, action, args):
        param_arg_map = {}
        for arg, param in zip(args, action.parameters):
            arg_type = self._get_object_type(arg)
            if not self.domain.types[arg_type].is_type(param.type):
                raise ActionException("Planed action arguments do not match action parameter types")
            param_arg_map[param] = PDDLObject(arg, arg_type)
        return param_arg_map

    def apply_action(self, action, args, state):
        """ Apply an action to the given state. Returns (success, resulting_state)."""
        if not self._check_preconditions(action, state):
            raise ActionException("Cannot perform %s in current state." % action.name)
        action.args = self._resolve_args(action, args)
        for effect in action.effects:
            self._apply_effect(effect, state)

    def get_plan_intermediary_states(self, problem, plan):
        states = [problem.init]
        for step in plan:
            states.append(self.apply_action(step.name, step.args, states[-1]))
        return states

    def solve(self):
        """ Solve the given problem in this domain. """
        solver = FF(self.domain, self.problem, ff_executable="../ff")
        solution = solver.solve()
        return solution


class Planner(object):
    """ Base class for planners to solve PDDL problems. """
    def __init__(self, domain, problem):
        self.domain = domain
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


from tempfile import NamedTemporaryFile
from subprocess import check_output, CalledProcessError
from os import remove


class FF(Planner):
    """ A solver instance based on an FF executable. """
    def __init__(self, domain, problem, ff_executable='./ff'):
        super(FF, self).__init__(domain, problem)
        self.ff_executable = ff_executable

    def _parse_solution(self, soln_txt):
        """ Extract list of solution steps from FF output. """
        sol = []
        soln_txt = soln_txt.split('step')[1].strip()
        soln_txt = soln_txt.split('time spent')[0].strip()
        steps = [step.strip() for step in soln_txt.splitlines()]
        for step in steps:
            args = step.split(':')[1].lstrip().split()
            act = args.pop(0)  # Remove action, leave all args
            sol.append(PDDLPlanStep(act, args))
        return sol

    def solve(self):
        """ Create a temporary problem file and call FF to solve. """
        with NamedTemporaryFile() as problem_file:
            self.problem.to_file(problem_file.name)
            with NamedTemporaryFile() as domain_file:
                self.domain.to_file(domain_file.name)
                try:
                    soln_txt = check_output([self.ff_executable, '-o', domain_file.name, '-f', problem_file.name])
                except CalledProcessError as cpe:
                    if "goal can be simplified to TRUE." in cpe.output:
                        return True
                    else:
                        print "FF Could not find a solution to problem: %s" % self.problem.domain
                        return []
                finally:
                    # clean up the soln file produced by ff (avoids large dumps of files in /tmp)
                    try:
                        remove('.'.join([problem_file.name, 'soln']))
                    except OSError as ose:
                        if ose.errno != 2:
                            raise ose
        return self._parse_solution(soln_txt)
