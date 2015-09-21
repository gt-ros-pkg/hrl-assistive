#!/usr/bin/env python


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


class PDDLType(object):
    """ A class describing a type in PDDL."""
    def __init__(self, name, supertype=None):
        self.name = name
        self.supertype = supertype

    def is_subtype(self):
        return bool(self.supertype)

    def is_type(self, check_type):
        if self.name == check_type:
            return True
        elif self.is_subtype():
            return self.supertype.is_type(check_type)
        else:
            return False

    def __str__(self):
        if self.is_subtype():
            return " - ".join([self.name, self.supertype.name])
        else:
            return self.name


class PDDLObject(object):
    """ A class describing an Object in PDDL. """
    def __init__(self, name, type=None):
        self.name = name
        self.type = type

    @classmethod
    def from_string(cls, string):
        """ Create a PDDLObject instance from a formatted string."""
        print "Object from String: ", string
        string = string.strip('( )')
        name, type_ = string.split(' - ')
        return cls(name.strip(), type_.strip())

    def is_variable(self):
        return bool(self.name[0] == '?')

    def __str__(self):
        if self.type is None:
            return self.name.upper()
        else:
            return "%s - %s" % (self.name.upper(), self.type.upper())


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

    def is_abstract(self):
        return any([bool(arg.name[0] == '?') for arg in self.args])

    @classmethod
    def from_string(cls, string):
        """ Create a PDDLPredicate instance from a formatted string."""
        return cls.from_list(lisp_to_list(string))

    @classmethod
    def from_list(cls, pred_list):
        print "Predicate.from_list(%s)" % pred_list
        neg = False
        if pred_list[0] == 'NOT':
            neg = True
            pred_list = pred_list[1]
        name = pred_list.pop(0)
        print "Name-stripped Pred_list: ", pred_list
        if '-' in pred_list:
            name_type_pairs = pred_list.count('-')
            args = []
            for i in range(name_type_pairs):
                args.append(PDDLObject(pred_list[3*i], pred_list[3*i+2]))
        else:
            args = [PDDLObject(arg) for arg in pred_list]
        res = cls(name, args, neg)
        print "Predicate object:", str(res)
        return res

    def __str__(self):
        msg = "(%s %s)" % (self.name, ' '.join(map(str, self.args)))
        if self.neg:
            msg = ''.join(["( NOT ", msg, ")"])
        return msg.upper()


class PDDLPlanStep(object):
    """ A class specifying a PDDL action and the parameters with which to call apply it. """
    def __init__(self, name, args):
        self.name = name
        self.args = args

    @classmethod
    def from_string(cls, string):
        name, args = lisp_to_list(string)
        return cls(name, args)

    def __str__(self):
        return ''.join(["(", self.name, "(", ' '.join(self.args), "))"]).upper()


class PDDLAction(object):
    """ A class describing an action in PDDL. """
    def __init__(self, name, parameters=[], preconditions=[], effects=[]):
        self.name = name
        self.parameters = parameters
        self.preconditions = preconditions
        self.effects = effects

    def _meets_preconditions(self, init_state):
        """ Make sure that the initial state to which the action is being applied meets the required preconditions."""
        for cond in map(str, self.preconditions):
            if cond not in map(str, init_state):
                return False
        return True

    def apply(self, init_state):
        if not self._meets_preconditions(init_state):
            return False

    @classmethod
    def from_string(cls, string):
        act = lisp_to_list(string)
        act = act[1:] if act[0] == ":ACTION" else act
        return cls.from_list(act)

    @classmethod
    def from_list(cls, act):
        print "Action.from_list(%s)" % act
        name = act[0]
        preconditions = []
        params = {}
        effects = []
        try:  # Evaluate parameters passed to action
            param_list = act[act.index(':PARAMETERS') + 1]
            for i in range(len(param_list)/3):
                params[param_list[3*i]] = PDDLObject(param_list[3*i], param_list[3*i+2])
        except ValueError:
            pass
        try:  # Evaluate Preconditions
            precond_list = act[act.index(":PRECONDITION") + 1]
            precond_list = precond_list[1:] if precond_list[0] == 'AND' else precond_list  # Ignore initial AND
            for cond in precond_list:
                if cond[0] == 'FORALL':
                    print "Forall"
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
        print "Effect: ", effect
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
        if effect[0] == 'FORALL':
            return ''.join(["(FORALL (", str(effect[1]),") ", self._effect_str(effect[2])])
        elif effect[0] == 'WHEN':
            return ''.join(["(WHEN ", str(effect[1]), self.effect_str(effect[2])])
        elif effect[0] == 'PREDICATE':
            return str(effect[1])
        else:
            return ''.join(["(AND (", '\n'.join([self._effect_str(eff) for eff in effect[1:]]), ")"])

    def _precondition_str(self, precond):

    def __str__(self):
        string = ''.join(["(:ACTION ", self.name, '\n'])
        string += ":PARAMETERS ( "
        string += ' '.join(map(str, self.parameters.itervalues()))
        string += ' )\n'
        if self.preconditions:
            string += ":PRECONDITION ( "
            if len(self.preconditions) > 1:
                string += 'AND(
            string += self._precondition_str(self.preconditions)
            string += ' )\n'
        # TODO: Complete recursive printing for effects
        string += ":EFFECT ( "
        string += self._effects_str(self.effects)
        string += " )\n)"
        return string


class PDDLDomain(object):
    """ A class describing a domain instance in PDDL."""
    def __init__(self, name, requirements=[], types={}, constants=[], predicates={}, actions={}):
        self.name = name
        self.requirements = requirements
        self.types = types
        self.constants = constants
        self.predicates = predicates
        self.actions = actions

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
    def _parse_predicates(cls, pred_list, types):
        preds = {}
        for pred in pred_list:
            preds[pred[0]] = PDDLPredicate.from_list(pred)
        return preds

    @classmethod
    def from_file(cls, domain_file):
        with open(domain_file, 'r') as f:
            string = f.read()
        return string

    @classmethod
    def from_string(cls, string):
        items = lisp_to_list(string.upper())
        ind = items.index('DEFINE')
        items.pop(ind)
        domain_name = get_sublist(items, "DOMAIN")[1]
        domain_requirements = get_sublist(items, ":REQUIREMENTS")[1:]
        domain_types = cls._parse_types(get_sublist(items, ":TYPES")[1:])
        constants = cls._parse_objects(get_sublist(items, ":CONSTANTS")[1:])
        predicates = cls._parse_predicates(get_sublist(items, ":PREDICATES")[1:], domain_types)
        actions_list = [item[1:] for item in items if item[0] == ':ACTION']
        actions = {}
        for action in actions_list:
            actions[action[0]] = PDDLAction.from_list(action)  # create dict of actions by name
        return cls(domain_name, domain_requirements, domain_types, constants, predicates, actions)

    def to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(str(self))

    def __str__(self):
        string = "(DEFINE (DOMAIN %s)\n\n" % self.name
        string += "(:REQUIREMENTS %s)\n\n" % ' '.join(self.requirements)
        types = [t for t in self.types.itervalues() if t.is_subtype()]  # Put all sub-types up front...
        for t in self.types.iterkeys():  #...and all supertypes at the end of the list
            if t not in types:
                types.append(t)
        string += "(:TYPES\n%s)\n\n" % '\n'.join(map(str, types))
        string += "(:CONSTANTS\n%s)\n\n" % '\n'.join(map(str, self.constants))
        string += "(:PREDICATES\n%s)\n\n" % '\n'.join(map(str, self.predicates.itervalues()))
        string += '\n\n'.join(map(str, self.actions.itervalues()))
        string += ")"
        return string

    def _get_constants_by_type(self):
        # TODO Check to make sure this still works!?
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

    @classmethod
    def from_msg(cls, msg):
        objects = [PDDLObject.from_string(obj_str) for obj_str in msg.objects]
        init = [PDDLObject.from_string(pred) for pred in msg.init]
        goal = [PDDLPredicate.from_string(pred) for pred in msg.goal]
        return cls(msg.name, msg.domain, objects, init, goal)

    @classmethod
    def from_file(cls, filename):
        """ Load a PDDL Problem from a PDDL problem file. """
        with open(filename, 'r') as pfile:
            string = ''.join(pfile.readlines())
        return cls.from_string(string)

    @classmethod
    def from_string(cls, string):
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
