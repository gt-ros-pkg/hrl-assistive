RFH.Smach = function(options) {
    var self = this;
    self.ros = options.ros;
    self.$displayContainer = options.displayContainer || $('#scmah-display');
    self.display = new RFH.SmachDisplay({ros:self.ros,
                                         container: self.$displayContainer});
    self.smach_tasks = {};

    self.solutionSubscriber = new ROSLIB.Topic({
        ros: self.ros,
        name: '/task_solution',
        type: 'hrl_task_planning/PDDLSolution'
    });

    self.planSolutionCB = function(msg) {
        self.display.empty(); // Out with the old
        var actions = self.parseActions(msg.actions);
        var interfaceTasks = self.getInterfaceTasks(msg.domain, actions);
        self.smach_tasks[msg.problem] = {
            domain: msg.domain,
            actions: actions,
            states: msg.states,
            interfaceTasks: interfaceTasks
        };
        self.display.displaySmachStates(self.smach_tasks[msg.problem].actions);
    };
    self.solutionSubscriber.subscribe(self.planSolutionCB);

    self.parseActions = function(actions) {
        var act, i, name_args, name, args;
        var acts_list = [];
        for (i = 0; i < actions.length; i += 1) {
            act = actions[i];
            act = act.slice(1, -2); // First+last(2) char are open/close parens. Remove them.
            name_args = act.split('(');
            name = name_args[0];
            args = name_args.slice(1);
            args = args[0].split(' ');
            acts_list[i] = {
                name: name,
                args: args
            };
        }
        return acts_list;
    };

    self.getInterfaceTasks = function (domain, actions) {
        var task = RFH.taskMenu.tasks[domain];
        var taskNames = [];
        for (var i=0; i < actions.length; i += 1) {
            taskNames.push(task.getInterfaceTask(actions[i]));
        }
        return taskNames;
    };

    self.smachStatusSubscriber = new ROSLIB.Topic({
        ros: self.ros,
        name: "/task_state",
        type: "hrl_task_planning/PDDLState"
    });


    var matchingStateIndex = function(state_msg) {
        var problem_states = self.smach_tasks[state_msg.problem].states;
        for (var idx in problem_states) {
            if (problem_states[idx].predicates.length === state_msg.predicates.length) {
                for (var i = 0; i < state_msg.predicates.length; i += 1) {
                    if (problem_states[idx].predicates[i] == state_msg.predicates[i]) {
                        return idx;
                    }
                }
                continue;
            }
        }
        return -1; // No match found...
    };
    
    var pddlStateCB = function(state_msg) {
        var idx = matchingStateIndex(state_msg);
        if (idx == self.smach_tasks[state_msg.problem].states.length-2) { // -1 for initial state on states list, -1 for index vs length diff.
            self.display.empty();
        } else if (idx >= 0) {
            self.display.setActive(idx); // Update Display
            RFH.taskMenu.startTask(self.smach_tasks[state_msg.problem].interfaceTasks[idx]); // Start corresponding task
        }
    };
    self.smachStatusCBList = [pddlStateCB];

    self.smachStatusCB = function (msg) {
        for (var i=0; i < self.smachStatusCBList.length; i += 1) {
            self.smachStatusCBList[i](msg);
        }
    };
    self.smachStatusSubscriber.subscribe(self.smachStatusCB);


};


RFH.SmachDisplay = function(options) {
    "use strict";
    var self = this;
    self.ros = options.ros;
    self.$container = options.container;

    self.empty = function () {
        self.$container.empty();
    };

    self.displaySmachStates = function(task) {
        for (var i = 0; i < task.length; i += 1) {
            self.$container.append($('<div>', {
                class: "smach-state incomplete",
                text: task[i].name
            }));
            self.$container.append($('<div>', {
                class: "smach-state-separator"
            }));
        }
        self.$container.find('.smach-state-separator').last().remove(); // Don't need extra connector bar hanging off the end.
    };

    self.setActive = function(idx) {
        self.$container.find('.active').removeClass('active').addClass('complete');
        var activate = self.$container.find('.smach-state:eq(' + idx + ')').removeClass('incomplete').addClass('active');
    };


};
