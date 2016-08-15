#!/usr/bin/env python

import copy
import itertools as it


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


class Type(object):
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


class Object(object):
    """ A class describing an Object in PDDL. """
    def __init__(self, name, type_=None):
        assert isinstance(name, str), "Object name must be a string."
        assert isinstance(type_, str) or type_ is None, "Object name must be a string."
        self.name = name
        self.type = type_

    def __str__(self):
        if self.type is None:
            return self.name.upper()
        else:
            return "%s - %s" % (self.name.upper(), self.type.upper())

    def __eq__(self, other):
        return isinstance(other, self.__class__) and (self.__dict__ == other.__dict__)

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


class Predicate(object):
    """ A class describing a predicate in PDDL. """
    def __init__(self, name, args=None, neg=False):
        self.name = name
        self.args = [] if args is None else args
        self.neg = neg

    def __str__(self):
        msg = "(%s %s)" % (self.name, ' '.join(map(str, self.args)))
        if self.neg:
            msg = ''.join(["( NOT ", msg, ")"])
        return msg.upper()

    def __eq__(self, other):
        return isinstance(other, self.__class__) and (self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def negate(self):
        self.neg = not self.neg

    def is_abstract(self):
        """ Check if this is an abstract predicate definition (True), or a specific Predicate statement (False)."""
        return any([bool(arg.name[0] == '?') for arg in self.args])

    @classmethod
    def from_string(cls, string):
        """ Create a Predicate instance from a formatted string."""
        return cls.from_list(lisp_to_list(string))

    @classmethod
    def from_list(cls, pred_list):
        """ Create a Predicate instance from a separated list of items."""
        neg = False
        if pred_list[0] == 'NOT':
            neg = True
            pred_list = pred_list[1]
        name = pred_list[0]
        if '-' in pred_list:
            name_type_pairs = pred_list[1:].count('-')
            args = []
            for i in range(name_type_pairs):
                args.append(Object(pred_list[1:][3*i], pred_list[1:][3*i+2]))
        else:
            args = [Object(arg) for arg in pred_list[1:]]
        res = cls(name, args, neg)
        return res


class State(object):
    def __init__(self, iterable=None):
        self.predicates = []
        iterable = [] if iterable is None else iterable
        for pred in iterable:
            assert isinstance(pred, Predicate), "Argument to pddl.State must be a list of pddl.Predicate objects"
            self.add(pred)

    def __str__(self):
        return "[" + ', '.join(map(str, self.predicates)) + "]"

    def __len__(self):
        return len(self.predicates)

    def __contains__(self, pred):
        return pred in self.predicates

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.satisfies_predicates(other.predicates) and
                other.satisfies_predicates(self.predicates))

    def __ne__(self, other):
        return not self.__eq__(other)

    def string_list(self):
        return map(str, self.predicates)

    @staticmethod
    def _remove_duplicates(pred_list):
        items = []
        for pred in pred_list:
            if pred not in items:
                items.append(pred)
        return items

    def satisfies_predicates(self, preds):
        """ Determine if the given state satisfies all of the listed predicates """
        for pred in preds:
            pos_pred = Predicate(pred.name, pred.args)  # Use equivalent non-negated (avoids switching negation flag on predicate itself)
            if (pred.neg and pos_pred in self.predicates) or (not pred.neg and pos_pred not in self.predicates):
                return False
        return True

    def add(self, new_pred):
        """ Add a predicate to the state, or remove a predicate if adding a negative."""
        if new_pred.neg:
            try:
                self.predicates.remove(Predicate(new_pred.name, new_pred.args))  # Use equivalent non-negated (avoids switching negation flag on predicate itself)
            except ValueError:
                pass  # Positive predicate not in list, so don't need to remove
        else:
            if new_pred not in self.predicates:
                self.predicates.append(new_pred)

    def difference(self, other):
        """ Returns the set of predicates which would bring this state to match the argument state."""
        diff_list = []
        for pred in self.predicates:
            if pred not in other:
                diff_list.append(Predicate(pred.name, pred.args, neg=True))
        for pred in other.predicates:
            if pred not in self.predicates:
                diff_list.append(Predicate(pred.name, pred.args))
        return diff_list

    def apply_update(self, update):
        assert isinstance(update, StateUpdate), "apply_update only accepts StateUpdate obects."
        for predicate in update.predicates:
            if predicate.neg:
                try:
                    self.predicates.remove(predicate)
                except ValueError:
                    pass
            else:
                self.prediates.add(predicate)

    @classmethod
    def from_msg(cls, msg):
        return cls(msg.predicates)


class StateUpdate(State):
    """ A class representing a change in PDDL State."""
    def add(self, new_pred):
        if new_pred.neg:
            try:
                self.predicates.remove(Predicate(new_pred.name, new_pred.args))  # Use equivalent non-negated (avoids switching negation flag on predicate itself)
            except ValueError:
                pass  # Positive predicate not in list, so don't need to remove
        if new_pred not in self.predicates:
            self.predicates.append(new_pred)


class GoalState(State):
    """ A Goal PDDL State (can contain negative predicates)"""
    def add(self, new_pred):
        """ Add a predicate to the state, or remove a predicate if adding a negative."""
        if new_pred.neg:
            try:
                self.predicates.remove(Predicate(new_pred.name, new_pred.args))  # Use equivalent non-negated (avoids switching negation flag on predicate itself)
            except ValueError:
                pass  # Positive predicate not in list, so don't need to remove
            self.predicates.append(new_pred)
        if new_pred not in self.predicates:
            self.predicates.append(new_pred)

    def is_satisfied(self, state):
        for pred in self.predicates:
            if not pred.neg:
                if pred not in state:
                    return False
            else:
                if Predicate(pred.name, pred.args) in state:
                    return False
        return True


class PlanStep(object):
    """ A class specifying a PDDL action and the parameters with which to call apply it. """
    def __init__(self, name, args=None):
        self.name = name
        self.args = [] if args is None else args

    @classmethod
    def from_string(cls, string):
        """ Create a PlanStep from a formatted string."""
        name, args = lisp_to_list(string)
        return cls(name, args)

    def __str__(self):
        return ''.join(["(", self.name, "(", ' '.join(self.args), "))"]).upper()


class ActionException(Exception):
    """ Exception raised when action cannot be executed from a given state."""
    pass


class PlanningException(Exception):
    """ Exception raised by a planner when no solution is available."""
    pass


class Action(object):
    """ A class describing an action in PDDL. """
    def __init__(self, name, parameters=None, preconditions=None, effects=None):
        self.name = name
        self.parameters = [] if parameters is None else parameters
        self.preconditions = [] if preconditions is None else preconditions
        self.effects = [] if effects is None else effects

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
        """ Create a Action from a formatted string."""
        act = lisp_to_list(string)
        act = act[1:] if act[0] == ":ACTION" else act
        return cls.from_list(act)

    @classmethod
    def from_list(cls, act):
        """ Create a Action from a formatted list of items."""
        name = act[0]
        preconditions = []
        params = []
        effects = []
        try:  # Evaluate parameters passed to action
            param_list = act[act.index(':PARAMETERS') + 1]
            for i in range(len(param_list)/3):
                params.append(Object(param_list[3*i], param_list[3*i+2]))
        except ValueError:
            pass
        try:  # Evaluate Preconditions
            precond_list = act[act.index(":PRECONDITION") + 1]
            precond_list = precond_list[1:] if precond_list[0] == 'AND' else precond_list  # Ignore initial AND
            for cond in precond_list:
                if cond[0] == 'FORALL':
                    param = Object(cond[1][0], cond[1][2])
                    pred = Predicate.from_list(cond[2])
                    preconditions.append(['FORALL', param, pred])
                else:
                    preconditions.append(Predicate.from_list(cond))
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
            param = Object(effect[1][0], effect[1][2])
            pred = cls._parse_effect(effect[2])
            return ['FORALL', param, pred]
        elif effect[0] == 'WHEN':
            param = Predicate.from_list(effect[1])
            pred = cls._parse_effect(effect[2])
            return ['WHEN', param, pred]
        else:
            return ["PREDICATE", Predicate.from_list(effect)]

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

    @staticmethod
    def _precondition_str(precond):
        """ Produce a properly formatted string for a precondition of an action instance."""
        if isinstance(precond, Predicate):
            return str(precond)
        elif precond[0] == 'FORALL':
            return ''.join(['(FORALL (', str(precond[1]), ') ', str(precond[2]), ")"])

    def get_parameter_types(self):
        return [param.type for param in self.parameters]


class Domain(object):
    """ A class describing a domain instance in PDDL."""
    def __init__(self, name, requirements=None, types=None, constants=None, predicates=None, actions=None):
        self.name = name
        self.requirements = [] if requirements is None else requirements
        self.types = {} if types is None else types
        self.constants = [] if constants is None else constants
        self.predicates = {} if predicates is None else predicates
        self.actions = {} if actions is None else actions

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
        if problem.domain_name.upper() != self.name.upper():
            print "Problem-specified domain (%s) does not match this domain (%s)" % (problem.domain_name, self.name)
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
            objs.append(Object(item_list[3*i], item_list[3*i+2]))
        return objs

    @classmethod
    def _parse_types(cls, types_list):
        """ Extract heirarchical Types from a formatted list. """
        types = {}
        # Catch simple case of no sub-types
        if '-' not in types_list:
            for t in types_list:
                types[t] = Type(t)
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
                types[t] = Type(t)
                added.append(t)
        # Deal with the rest of the mess
        subtype_list = [typ for group in subtypes for typ in group]
        for t in type_set:
            if (t in supertypes) and (t not in subtype_list):  # Start with top-level types
                types[t] = Type(t)
                added.append(t)
        while not set(added) == type_set:
            for supertype in added:
                try:
                    ind = supertypes.index(supertype)
                except ValueError:
                    continue
                subs = subtypes[ind]
                for sub in subs:
                    types[sub] = Type(sub, types[supertype])
                    added.append(sub)
        return types

    @classmethod
    def _parse_predicates(cls, pred_list):
        """ Produce a dict of Preciates from a list of predicate definition lists."""
        preds = {}
        for pred in pred_list:
            preds[pred[0]] = Predicate.from_list(pred)
        return preds

    @classmethod
    def from_file(cls, domain_file):
        """ Produce a Domain object from the specified PDDL domain file."""
        with open(domain_file, 'r') as f:
            string = f.read()
        return cls.from_string(string)

    @classmethod
    def from_string(cls, string):
        """ Produce a Domain object from a PDDL domain file string."""
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
            actions[action[0]] = Action.from_list(action)  # create dict of actions by name
        return cls(domain_name, domain_requirements, domain_types, constants, predicates, actions)

    def to_file(self, filename):
        """ Write this Domain as a properly-formatted domain file."""
        with open(filename, 'w') as f:
            f.write(str(self))


class Problem(object):
    """ A class describing a problem instance in PDDL. """
    def __init__(self, name, domain_name, objects=None, init=None, goal=None):
        self.name = name
        self.domain_name = domain_name
        self.objects = [] if objects is None else objects
        self.init = [] if init is None else init
        self.goal = [] if goal is None else goal

    def __str__(self):
        """ Write a PDDL Problem as a string in PDDL File style """
        s = "(DEFINE\n(PROBLEM " + self.name + ")\n"
        s += "(:DOMAIN " + self.domain_name + ")\n"
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
        objects = [Object.from_string(obj_str) for obj_str in msg.objects]
        init = [Predicate.from_string(pred) for pred in msg.init]
        goal = [Predicate.from_string(pred) for pred in msg.goal]
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
            objs.append(Object(obj, supertype))
        return objs

    @classmethod
    def _parse_init(cls, items):
        """ Extract predicates from a defined list. """
        return [Predicate(item[0], [Object(name) for name in item[1:]]) for item in items]

    @classmethod
    def _parse_goal(cls, items):
        preds = []
        for item in items:
            if item == 'AND':
                continue
            elif item[0] == 'NOT':
                preds.append(Predicate(item[1][0], [Object(name) for name in item[1][1:]], True))
            else:
                preds.append(Predicate(item[0], [Object(name) for name in item[1:]]))
        return preds

    def to_file(self, filename=None):
        """ Write a PDDL Problem file based on a Problem instance. """
        filename = '.'.join([self.name, 'problem']) if filename is None else filename
        with open(filename, 'w') as prob_file:
            string = str(self)
            prob_file.write(string)


class Situation(object):
    def __init__(self, domain, problem):
        if not domain.check_problem(problem):
            raise RuntimeError("Problem cannot be applied to this domain.")
        self.domain = domain
        self.problem = problem
        self.objects = self._merge_objects(domain, problem)
        self.solution = None
        self.states = []

    def _get_object_type(self, obj):
        """ Get the type of an object from the combined object list by name."""
        for known_object in self.objects:
            if obj == known_object.name:
                return known_object.type
        return None

    def _get_objects_of_type(self, type_):
        """ Return a list of objects of the given type.  Used for 'FORALL' conditions."""
        objs = []
        for obj in self.objects:
            if self.domain.types[obj.type].is_type(type_):
                objs.append(obj)
        return objs

    @staticmethod
    def _merge_objects(domain, problem):
        """ Merge objects from domain (constants) and problem (objects) to avoid repeated entries."""
        objs = domain.constants
        for obj in problem.objects:
            if obj not in objs:
                objs.append(obj)
        return objs

    def _get_effects(self, effect, state, arg_map):
        """ Get lists of predicates to add or remove from a state based on action effects."""
        add_list = []
        del_list = []
        if effect[0] == 'PREDICATE':
            eff = Predicate(effect[1].name, [Object(arg_map[arg.name].name) for arg in effect[1].args])
            if effect[1].neg:
                del_list.append(eff)
            else:
                add_list.append(eff)
        elif effect[0] == 'WHEN':
            cond = Predicate(effect[1].name, [Object(arg_map[arg.name].name) for arg in effect[1].args], effect[1].neg)
            pos_pred = Predicate(cond.name, cond.args)
            if (cond.neg and pos_pred not in state) or (not cond.neg and pos_pred in state):  # If condition negative and pred not in state, or condition positive and it is, evaluate it
                al, dl = self._get_effects(effect[2], state, arg_map)
                add_list.extend(al)
                del_list.extend(dl)
        elif effect[0] == 'FORALL':
            for obj in self._get_objects_of_type(effect[1].type):
                arg_map[effect[1].name] = obj
                al, dl = self._get_effects(effect[2], state, arg_map)
                add_list.extend(al)
                del_list.extend(dl)
        else:
            for eff in effect:
                al, dl = self._get_effects(eff, state, arg_map)
                add_list.extend(al)
                del_list.extend(dl)
        return (add_list, del_list)

    @staticmethod
    def _apply_changes(state, add_list, del_list):
        """ Add/remove predicates from a state as appropriate (avoids removing states just added by this effect."""
        new_state = copy.deepcopy(state)
        filtered_add_list = [pred for pred in add_list if pred not in state]  # Actually add it if it doesn't exist
        filtered_del_list = [pred for pred in del_list if pred in state]
        for pred in filtered_add_list:
            new_state.add(pred)
        for pred in filtered_del_list:
            pred.negate()
            new_state.add(pred)
        return new_state

    def _apply_effects(self, effect, state, arg_map):
        """ Recursively apply the effects of of an action to a state.  Requires argument map."""
        add_list, del_list = self._get_effects(effect, state, arg_map)
        return self._apply_changes(state, add_list, del_list)

    def _expand_conditions(self, action, arg_map):
        """ Create specific predicates for all preconditions of an action."""
        condition_predicates = []
        for cond in action.preconditions:
            if isinstance(cond, Predicate):
                condition_predicates.append(Predicate(cond.name, [Object(arg_map[arg.name].name) for arg in cond.args], cond.neg))
            else:
                for obj in self._get_objects_of_type(cond[1].type):
                    arg_map[cond[1].name] = obj
                    condition_predicates.append(Predicate(cond[2].name, [Object(arg_map[arg.name].name) for arg in cond[2].args], cond[2].neg))
        return condition_predicates

    def _resolve_args(self, action, args):
        """ Create a map from variable parameters in action defs to specific arguments in an action call."""
        param_arg_map = {}
        for arg, param in zip(args, action.parameters):
            arg_type = self._get_object_type(arg)
            if not self.domain.types[arg_type].is_type(param.type):
                raise ActionException("Action arguments do not match action parameter types")
            param_arg_map[param.name] = Object(arg, arg_type)
        return param_arg_map

    def apply_action(self, action, args, state):
        """ Apply an action to the given state. Returns (success, resulting_state)."""
        arg_map = self._resolve_args(action, args)
        all_preconditions = self._expand_conditions(action, arg_map)
        if not state.satisfies_predicates(all_preconditions):
            raise ActionException("Cannot perform %s(%s) in current state (%s).\nPreconditions: %s"
                                  % (action.name, map(str, args), state, map(str, all_preconditions)))
        result_state = self._apply_effects(action.effects, state, arg_map)
        return result_state

    def get_plan_intermediary_states(self, plan=None):
        plan = self.solution if plan is None else plan
        if plan is None:
            raise RuntimeError("Cannot find intermediary plan states.  No plan provided, and no solution already stored.")
        states = [State(self.problem.init)]
        for step in plan:
            new_state = self.apply_action(self.domain.actions[step.name], step.args, copy.copy(states[-1]))
            states.append(new_state)
        return states

    def _get_arg_sets(self, types):
        obj_types = []
        for type_ in types:
            obj_types.append([obj.name for obj in self._get_objects_of_type(type_)])
        arg_sets = list(it.product(*obj_types))
        return arg_sets

    def test_domain(self, initial_states=None):
        initial_states = [State()] if initial_states is None else initial_states
        full_states_list = []
        for initial_state in initial_states:
            states_list = [initial_state]
            added_state = True
            while added_state:
                added_state = False
                for state in states_list:
                    for action in self.domain.actions.itervalues():
                        arg_sets = self._get_arg_sets(action.get_parameter_types())
                        for arg_set in arg_sets:
                            try:
                                new_state = self.apply_action(action, arg_set, state)
                                if new_state not in states_list:
                                    states_list.append(new_state)
                                    added_state = True
                            except ActionException:
                                continue
            for state in states_list:
                if state not in full_states_list:
                    full_states_list.append(state)
        print "\nComplete states set (%d states):" % len(full_states_list)
        return full_states_list


def find_irreversible_actions(self, solution, states, domain, planner):
    irreversible_actions = []
    for i in range(len(states)-1):
        init = set(copy.copy(states[i+1]))
        goal = set(copy.copy(states[i]))
        negate = init.difference(goal)  # items in init, but not in the goal.  These need to be actively negated
        negations = [Predicate(pred.name, pred.args, True) for pred in list(negate)]
        goal = list(goal)
        goal.extend(negations)
        p = Problem("undo-check-%s" % i, self.problem.domain_name, self.problem.objects, init, goal)
        try:
            planner.solve(domain, p)
        except PlanningException:
            irreversible_actions.append(solution[i])
    return irreversible_actions

#    @staticmethod
#    def _astar_dist(state, goal):
#        """ Compute the distance from the goal in terms of remaining predicates incorrect from goal."""
#        pass
#
#    def solve_Astar(self):
#        """ Solve this problem in this domain using the A* algorithm."""
#        pass
#
#    def solve(self, problem=None):
#        """ Masking function for switching solver implementations."""
#        return self.solve_FF(problem)
#
#    def solve_FF(self, problem=None, ff_executable="../ff"):
#        """ Solve the given problem in this domain using an external FF executable. """
#        problem = self.problem if problem is None else problem
#        solver = FF(self.domain, problem, ff_executable)
#        self.solution = solver.solve()
#        return self.solution

# class Planner(object):
#    """ Base class for planners to solve PDDL problems. """
#    def __init__(self, domain, problem):
#        self.domain = domain
#        self.problem = problem
#        self.solution = None
#
#    def solve(self):
#        raise NotImplementedError()
#
#    def print_solution(self):
#        """ Print solution steps. """
#        if self.solution is None:
#            print "This problem has not been solved yet."
#        elif self.solution == []:
#            print "Result: Initial State satisfies the Goal State"
#        elif not self.solution:
#            print "Result: FF failed to find a solution"
#        else:
#            print "Result:\n\tPlan:"
#            for step in self.solution:
#                args = ', '.join(step['args'])
#                print ''.join(["\t", step['act'], "(", args, ")"])
#

from tempfile import NamedTemporaryFile
from subprocess import check_output, CalledProcessError
from os import remove


class FF(object):
    """ A solver instance based on an FF executable. """
    def __init__(self, ff_executable='./ff'):
        self.ff_executable = ff_executable

    @staticmethod
    def _parse_solution(soln_txt):
        """ Extract list of solution steps from FF output. """
        sol = []
        soln_txt = soln_txt.split('step')[1].strip()
        soln_txt = soln_txt.split('time spent')[0].strip()
        steps = [step.strip() for step in soln_txt.splitlines()]
        for step in steps:
            args = step.split(':')[1].lstrip().split()
            act = args.pop(0)  # Remove action, leave all args
            sol.append(PlanStep(act, args))
        return sol

    def solve(self, domain, problem):
        """ Create a temporary problem file and call FF to solve. """
        original_problem_name = problem.name
        problem.name = "tmpProblemName"  # FF's parser gets confused by special characters, so don't let it seem them...
        with NamedTemporaryFile() as problem_file:
            problem.to_file(problem_file.name)
            with NamedTemporaryFile() as domain_file:
                domain.to_file(domain_file.name)
                try:
                    soln_txt = check_output([self.ff_executable, '-o', domain_file.name, '-f', problem_file.name])
                    print "FF Output:\n", soln_txt
                    if "problem proven unsolvable." in soln_txt:
                        # print "FF Could not find a solution to problem: %s" % self.problem.domain_name
                        raise PlanningException("FF could not solve problem (%s) in domain (%s)" % (problem.name, domain.name))
                except CalledProcessError as cpe:
                    print "FF Output:\n", cpe.output
                    if "goal can be simplified to TRUE." in cpe.output:
                        return []
                    else:
                        # print "FF Could not find a solution to problem: %s" % self.problem.domain_name
                        print "Problem file:\n", str(problem)
                        print "Domain file:\n", str(domain)
                        raise PlanningException("FF could not solve problem (%s) in domain (%s)" % (problem.name, domain.name))
                finally:
                    # clean up the soln file produced by ff (avoids large dumps of files in /tmp)
                    problem.name = original_problem_name
                    try:
                        remove('.'.join([problem_file.name, 'soln']))
                    except OSError as ose:
                        if ose.errno != 2:
                            raise ose
        return self._parse_solution(soln_txt)
