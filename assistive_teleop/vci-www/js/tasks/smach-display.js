RFH.Smach = function(options) {
    var self = this;
    var ros = options.ros;
    self.$displayContainer = options.displayContainer || $('#scmah-display');
    self.display = new RFH.SmachDisplay({ros: ros,
                                         container: self.$displayContainer});
    self.smachTasks = []; // Array of data on tasks. Display only most recent (last index) for ordering of sub-tasks.
    self.activeState = null;
    self.currentActionSubscribers = {};
//    ros.getMsgDetails('hrl_task_planning/PDDLSolution');
//    ros.getMsgDetails('hrl_task_planning/PDDLPlanStep');

    self.solutionSubscriber = new ROSLIB.Topic({
        ros: ros,
        name: '/task_solution',
        type: 'hrl_task_planning/PDDLSolution'
    });

    self.planSolutionCB = function(msg) {
        self.display.empty(); // Out with the old
        var actions = self.parseActions(msg.actions);
//        var interfaceTaskStarts = self.getInterfaceTaskStarts(msg.domain, actions);
        var taskLabels = self.getTaskLabels(msg.domain, actions);
        self.display.displaySmachStates(taskLabels);
        var taskData = {'domain': msg.domain,
                        'problem': msg.problem,
                        'labels': taskLabels,
                        'actions': actions,
                        'states': msg.states};
        self.smachTasks.push(taskData);
        self.setupCurrentActionSubscriber(msg.domain);
    };
    self.solutionSubscriber.subscribe(self.planSolutionCB);

    self.setupCurrentActionSubscriber = function (domain) {
        self.currentActionSubscribers[domain] = new ROSLIB.Topic({
            ros: ros,
            name: '/pddl_tasks/'+domain+'/current_action',
            type: '/hrl_task_planning/PDDLPlanStep'
        });
        self.currentActionSubscribers[domain].subscribe(self.updateCurrentAction);
    };

    self.updateCurrentAction = function (planStepMsg) {
        if (planStepMsg.action === '') {
            self.display.empty(); 
            RFH.taskMenu.startTask('lookingTask');
            return;
        }
        var task;
        // Get the task from the list matching this message
        for (var i=0; i<self.smachTasks.length; i+=1){
            if (self.smachTasks[i].problem == planStepMsg.problem) {
                task = self.smachTasks[i];
            }
        }
        // Get the matching action from the task's actions
        for (i=0; i<task.actions.length; i+=1) {
            var action = task.actions[i];
            if (action.name === planStepMsg.action) {
                if (action.args.length == planStepMsg.args.length) {
                    for (var j=0; j<action.args.length; j+=1) {
                        if (action.args[j] !== planStepMsg.args[j]) {
                            continue;
                        } 
                        // Use the first instance (maybe problematic?), don't need to check the rest
                        // Switch interface to the correct task interface;
                        self.display.setActive(i);  // This is the first matching action, return it's index
                        RFH.taskMenu.tasks[planStepMsg.domain].startAction(planStepMsg);
                        return;
                    }
                }
            }
        }
    };

/*
    var smachStructureCB = function (scs_msg) {
        if (scs_msg.path.indexOf('/') >= 0) return; // Ignore non-base structures
        if (!self.smachTasks[scs_msg.path]) {
            self.smachTasks[scs_msg.path] = {};
        }
        if (self.smachTasks[scs_msg.path].steps !== undefined) return; // We've already got this one...
        var chain = getStateChain(scs_msg.outcomes_from, scs_msg.outcomes_to, scs_msg.internal_outcomes);
        self.smachTasks[scs_msg.path].steps = filterStateChain(chain);
    };

    var getStateChain = function (from_states, to_states, transitions) {
        var states = {};
        var i;
        for (i=0; i < transitions.length; i += 1) {
            if (transitions[i] == 'succeeded') {
                states[from_states[i]] = to_states[i];
            }
        }
        var start_state, end_state;
        for (var ind in states) {
            if (to_states.indexOf(ind) < 0) {
                start_state = ind;
            }
            if (states[ind] == 'succeeded') {
                end_state = ind;
            }
        }
        var stateList = [start_state];
        for (ind in states) {
            stateList.push(states[stateList[stateList.length - 1]]);
            if (stateList[stateList.length - 1] == end_state) return stateList;
        }
    };

    var filterStateChain = function (stateList) {
        var showList = [];
        for (var i=0; i < stateList.length; i += 1) {
            if (stateList[i][0] !== '_') {
                showList.push(stateList[i]);
            }
        }
        return showList;
    };
   
    var smachStructureSub = new ROSLIB.Topic({
        ros: ros,
        name: "/smach_introspection/smach/container_structure",
        type: "smach_msgs/SmachContainerStructure"
    });
    smachStructureSub.subscribe(smachStructureCB);

    var smachContainerStatusCB = function (status_msg) {
        if (status_msg.path.indexOf('/') >= 0) return; // Ignore non-base structures
        if (!self.smachTasks[status_msg.path].steps) return; // Wait until we know the structure...
        if (!self.smachTasks[status_msg.path].interfaceTasks) return; // Wait until we know the interface task data
        if (self.activeState !== status_msg.active_states[0]) {
            self.activeState = status_msg.active_states[0]; // There can be only one...
            if (self.activeState[0] === '_') {
                if (self.activeState.indexOf('FINAL') >= 0 ||
                    self.activeState.indexOf('CLEANUP') >= 0) {
                    self.display.empty();
                }
            } else {
                var idx = self.smachTasks[status_msg.path].steps.indexOf(self.activeState);
                self.display.setActive(idx);
                RFH.taskMenu.startTask(self.smachTasks[status_msg.path].interfaceTasks[idx]); // Start corresponding task
            }
        }
    };

    var smachContainerStatusSub = new ROSLIB.Topic({
        ros: ros,
        name: "/smach_introspection/smach/container_status_cleaned",
        type: "smach_msgs/SmachContainerStatus",
    });
    smachContainerStatusSub.subscribe(smachContainerStatusCB);

    self.smachStatusSubscriber = new ROSLIB.Topic({
        ros: ros,
        name: "/task_state",
        type: "hrl_task_planning/PDDLState"
    });
*/
    // Receives a list of action strings, returns a lists of action objects: [{name:'', args:['','',..]}, ... ]
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

    // Receives: domain/task name and list of action objects, returns task for interface to run for each action
    self.getInterfaceTaskStarts = function (domain, actions) {
        var task = RFH.taskMenu.tasks[domain];
        var taskStartFns = [];
        for (var i=0; i < actions.length; i += 1) {
            taskStartFns.push(task.getInterfaceTask(actions[i]));
        }
        return taskStartFns;
    };

    // Receives domain/task name + list of action objects, returns label for each action to use in guide display
    self.getTaskLabels = function (domain, actions) {
        var task = RFH.taskMenu.tasks[domain];
        var taskLabels = [];
        for (var i=0; i < actions.length; i += 1) {
            taskLabels.push(task.getActionLabel(actions[i]));
        }
        return taskLabels;
    };

};


RFH.SmachDisplay = function(options) {
    "use strict";
    var self = this;
    var ros = options.ros;
    self.$container = options.container;

    self.show = function () {
        self.$container.show();
    };

    self.hide = function () {
        self.$container.hide();
    };

    self.empty = function () {
        self.$container.empty();
        self.$container.hide();
    };

    // Receives a list of label strings, creates a display of actions in sequence with highlighting for done/current/future actions
    self.displaySmachStates = function(stringList) {
        for (var i = 0; i < stringList.length; i += 1) {
            self.$container.append($('<div>', {
                class: "smach-state incomplete",
                text: stringList[i].split('+')[0]
            }));
            self.$container.append($('<div>', {
                class: "smach-state-separator"
            }));
        }
        self.$container.find('.smach-state-separator').last().remove(); // Don't need extra connector bar hanging off the end.
    };

    self.setActive = function(idx) {
        if (idx > self.$container.find('.smach-state').length) { 
            self.empty();
            return;
        }
        self.$container.find('.smach-state:lt('+idx+')').removeClass('incomplete active').addClass('complete');
        self.$container.find('.smach-state:gt('+idx+')').addClass('incomplete active').removeClass('complete');
        self.$container.find('.smach-state:eq(' + idx + ')').removeClass('incomplete complete').addClass('active');
        self.show();
    };
};
