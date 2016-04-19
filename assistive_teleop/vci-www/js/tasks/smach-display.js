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

    var preemptTaskClient = new ROSLIB.Service({ros:ros,
                                                 name:'preempt_pddl_task',
                                                 serviceType: '/hrl_task_planning/PreemptTask'});

    var cancelAction = function (problem) {
        var cancelResultCB = function (resp) {
            if (resp.result) {
                console.log("Cancelled task successfully");
                self.display.empty();
            } else {
                RFH.log("Failed to cancel task");
            }
        };

        var req = new ROSLIB.ServiceRequest({'problem_name':problem});
        preemptTaskClient.callService(req, cancelResultCB);
    };
    self.display.cancelAction = cancelAction;

    var getDomainData = function (domain) {
        for (var i=0; i<self.smachTasks.length; i+= 1){
            if (self.smachTasks[i].domain === domain) {
                return self.smachTasks[i];
            }
        }
        return {domain: domain, problem:'', actionList:[]};
    };

    var updateFullActionList = function (currentActionList, updateActionList) {
        var newActionList = [];
        for (var i=0; i < currentActionList.length; i += 1){
            if (currentActionList[i].completed) {
                newActionList.push(currentActionList[i]);
            } else {
                break;
            }
        };
        // TODO: catch and replace visited states in new list?
        newActionList.push.apply(newActionList, updateActionList);
        return newActionList;
    };

    self.planSolutionCB = function(msg) {
        self.display.empty(); // Out with the old
        var domainData = RFH.taskMenu.tasks[msg.domain];
        var actions = self.parseActionStrings(msg.actions);
        for (var i=0; i<actions.length; i+=1) {
            actions[i].label = domainData.getActionLabel(actions[i].name, actions[i].args);
            actions[i].helpText = domainData.getActionHelpText(actions[i].name, actions[i].args);
            actions[i].startFunction = domainData.getActionFunction(actions[i].name, actions[i].args);
            actions[i].state = msg.states[i];
            actions[i].completed = false;
        }
        var previousTaskData = getDomainData(msg.domain);
        var actionList = updateFullActionList(previousTaskData.actionList, actions);
        var taskData = {'domain': msg.domain,
                        'problem': msg.problem,
                        'currentAction': null,
                        'actionList': actionList};
        self.smachTasks.push(taskData);
                
        self.display.setActionList(actionList);
        self.setupCurrentActionSubscriber(msg.domain);
    };

    var solutionSubscriber = new ROSLIB.Topic({
        ros: ros,
        name: '/task_solution',
        type: 'hrl_task_planning/PDDLSolution'
    });
    solutionSubscriber.subscribe(self.planSolutionCB);

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
            self.smachTasks.pop();
            if (self.smachTasks.length === 0){
                RFH.taskMenu.startTask(RFH.taskMenu.defaultTaskName);
                for (var i=0;i<RFH.regions.
                self.display.empty();
                return;
            }
            var newActions = self.smachTasks[self.smachTasks.length-1].actionList;
            var newCurrentAction = self.smachTasks[self.smachTasks.length-1].currentAction;
            self.display.setActionList(newActions);
            self.display.setCurrentAction(newCurrentAction, planStepMsg.problem);
            self.display.refreshDisplay();
            return;
        }

        // Get the task from the list matching this message
        nowCurrentAction =  {'name':planStepMsg.action, 'args': planStepMsg.args};
        for (var i=0; i<self.smachTasks.length; i+=1){
            if (self.smachTasks[i].problem == planStepMsg.problem) {
                self.smachTasks[i].currentAction = nowCurrentAction;

            }
        }
        self.display.setCurrentAction(nowCurrentAction, planStepMsg.problem);
        self.display.refreshDisplay();
    };

    // Receives a list of action strings, returns a lists of action objects: [{name:'', args:['','',..]}, ... ]
    self.parseActionStrings = function(actions) {
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

};


RFH.SmachDisplay = function(options) {
    "use strict";
    var self = this;
    var ros = options.ros;
    self.$container = options.container;
    var actionList;
    var currentAction;
    var currentProblem;

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

    self.setActionList = function (actions) {
        actionList = actions;
    };

    self.getActionList = function () {
        return actionList;
    };

    self.setCurrentAction = function (action, problem) {
        currentAction = action;
        currentProblem = problem;
    };

    self.getCurrentAction = function () {
        return currentAction;
    };

    var getActionIndex = function (action) {
        var i, j, match;
        for (i=0; i<actionList.length; i += 1){
            if (actionList[i].name == action.name) {
                match = true;
                for (j=0; j < action.args.length; j += 1) {
                    if (actionList[i].args[j] !== action.args[j]) {
                        actionList[i].completed = true;
                        match = false; 
                    } 
                }
                if (match) { return i; };
            } else { 
                actionList[i].completed = true;
            }
        };
    };

    self.refreshDisplay = function (){
        self.empty();
        var bubble;
        for (var i=0; i<actionList.length; i +=1) {
           bubble = $('<div>', {class: "smach-state incomplete",
                                text: actionList[i].label,
                                title: actionList[i].hoverText
                            }
                     ).on('click.rfh', actionList[i].startFunction);
            self.$container.append(bubble);
            self.$container.append($('<div>', { class: "smach-state-separator" }));
        }
        self.$container.find('.smach-state-separator').last().remove(); // Don't need extra connector bar hanging off the end.
        var cancelButton = $('<div>', {class:"smach-state cancel", text:"Cancel"}).on('click.rfh', function (event) {self.cancelAction(currentProblem)});
        self.$container.append(cancelButton);
        self.setActive(getActionIndex(currentAction));
    };

    // Receives a list of label strings, creates a display of actions in sequence with highlighting for done/current/future actions
//    self.displaySmachStates = function(stringList, cancelFn) {
//        for (var i = 0; i < stringList.length; i += 1) {
//            self.$container.append($('<div>', {
//                class: "smach-state incomplete",
//                text: stringList[i].split('+')[0]
//            }));
//            self.$container.append($('<div>', {
//                class: "smach-state-separator"
//            }));
//        }
//        self.$container.find('.smach-state-separator').last().remove(); // Don't need extra connector bar hanging off the end.
//        var cancelButton = $('<div>', {class:"smach-state cancel", text:"Cancel"});
//        cancelButton.on('click.rfh', cancelFn);
//        self.$container.append(cancelButton);
//    };

    self.setActive = function(idx) {
        if (idx > self.$container.find('.smach-state').length) { 
            self.empty();
            return;
        }
        self.$container.find('.smach-state:lt('+idx+')').removeClass('incomplete active').addClass('complete');
        self.$container.find('.smach-state:gt('+idx+')').removeClass('active complete').addClass('incomplete');
        self.$container.find('.smach-state:eq(' + idx + ')').removeClass('incomplete complete').addClass('active');
        actionList[idx].startFunction();
        self.show();
    };
};
