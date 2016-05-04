RFH.Smach = function(options) {
    var self = this;
    var ros = options.ros;
    self.$displayContainer = options.displayContainer || $('#scmah-display');
    self.display = new RFH.SmachDisplay({ros: ros,
                                         container: self.$displayContainer});
    self.smachTasks = []; // Array of data on tasks. Display only most recent (last index) for ordering of sub-tasks.
    self.domains = {};
    self.activeDomains = [];
    self.activeState = null;
    self.currentActionSubscribers = {};
    self.solutionSubscribers = {};

    var preemptTaskClient = new ROSLIB.Service({ros:ros,
                                                 name:'preempt_pddl_task',
                                                 serviceType: '/hrl_task_planning/PreemptTask'});

    self.cancelTask = function (problem) {
        var cancelResultCB = function (resp) {
            if (resp.result) {
                console.log("Cancelled task successfully");
                var i = self.smachTasks.length;
                while (i>0) {
                    if (self.smachTasks[i].problem == problem) {
                        self.smachTasks.pop();
                    }
                    i -= 1;
                }
                self.display.empty();
            } else {
                RFH.log("Failed to cancel task");
            }
        };

        var req = new ROSLIB.ServiceRequest({'problem_name':problem});
        preemptTaskClient.callService(req, cancelResultCB);
    };
    self.display.cancelTask = self.cancelTask;

    self.setupNewDomain = function (domain) {
        self.activeDomains.push(domain);
        self.setupCurrentActionSubscriber(domain);
        self.setupSolutionSubscriber(domain);
    };

    self.cleanupDomain = function (domain) {
        var idx = self.activeDomains.indexOf(domain);
        self.activeDomains.splice(idx, 1);
        self.currentActionSubscribers[domain].unsubscribe(); // Warning: removes all subs in rosjs, may be dangerous
        delete self.currentActionSubscribers[domain];
        self.solutionSubscribers[domain].unsubscribe();
        delete self.solutionSubscribers[domain];
        // TODO: CLEAR VISUALIZATION
    };

    var activeDomainsCB = function (domains_msg) {
        var newDomains = domains_msg.domain_list;
        for (var domain in self.domains) {
            var idx = newDomains.indexOf(domain);
            if (idx < 0) { // Previously active, now gone, so clean up
                self.cleanupDomain(domain);
            } else {
               newDomains.splice(idx, 1);  // If already known, remove from list
            }
        }
        // Set up subscribers for newly active domains
        for (var i=0; i<newDomains.length; i+=1) {
            self.setupNewDomain(newDomains[i]);
        };
    };
    var activeDomainsSubscriber = new ROSLIB.Topic({
        ros: ros,
        name: '/pddl_tasks/active_domains',
        messageType: '/hrl_task_planning/DomainList'
    });
    activeDomainsSubscriber.subscribe(activeDomainsCB);

    self.setupCurrentActionSubscriber = function (domain) {
        self.currentActionSubscribers[domain] = new ROSLIB.Topic({
                                                    ros: ros,
                                                    name: '/pddl_tasks/'+domain+'/current_action',
                                                    messageType: '/hrl_task_planning/PDDLPlanStep'
                                                });
        self.currentActionSubscribers[domain].subscribe(function(msg){self.updateCurrentAction(domain, msg)});
    };

    self.setupSolutionSubscriber = function (domain) {
        self.solutionSubscribers[domain] = new ROSLIB.Topic({
                                                ros: ros,
                                                name: '/pddl_tasks/'+domain+'/solution',
                                                messageType: 'hrl_task_planning/PDDLSolution'
                                            });
        self.solutionSubscribers[domain].subscribe(function(msg){self.updateSolution(domain, msg)});
    };



///////////////// THE LINE ////////////////////////////

    self.getDomainData = function (domain) {
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

    self.updateSolution = function(domain, msg) {
        self.display.empty(); // Out with the old
        var domainData = RFH.taskMenu.tasks[msg.domain];
        var actions = self.parseActionStrings(msg.actions);
        for (var i=0; i<actions.length; i+=1) {
            actions[i].label = domainData.getActionLabel(actions[i].name, actions[i].args);
            actions[i].helpText = domainData.getActionHelpText(actions[i].name, actions[i].args);
            actions[i].startFunction = domainData.getActionFunction(actions[i].name, actions[i].args);
            actions[i].init_state = msg.states[i];
            actions[i].goal_state = msg.states[i+1];
            actions[i].completed = false;
        }
        var previousTaskData = self.getDomainData(msg.domain);
        var actionList = updateFullActionList(previousTaskData.actionList, actions);
        self.display.setActionList(actionList);
        var taskData = {'domain': msg.domain,
                        'problem': msg.problem,
                        'currentAction': null,
                        'actionList': actionList};
        var duplicate = false;
        for (var j=0; j<self.smachTasks.length; j += 1) {
            if (self.smachTasks[j].domain == msg.domain && self.smachTasks[j].problem == msg.problem) {
                self.smachTasks[j] = taskData;
                duplicate=true;
            }
        }
        if (!duplicate) { 
            self.smachTasks.push(taskData);
            self.setupCurrentActionSubscriber(msg.domain);
        };
    };

    self.updateCurrentAction = function (domain, planStepMsg) {
        if (planStepMsg.action === '') { // Empty current action means domain completed successfully
            self.smachTasks.pop();
            if (self.smachTasks.length === 0){ // If last active task is now complete, clear everything
                RFH.taskMenu.startTask(RFH.taskMenu.defaultTaskName);
                self.display.empty();
            } else {  // Otherwise, get the next task up the heirarchy
                var newActions = self.smachTasks[self.smachTasks.length-1].actionList;
                var newCurrentAction = self.smachTasks[self.smachTasks.length-1].currentAction;
                self.display.setActionList(newActions);
                self.display.setCurrentAction(newCurrentAction, planStepMsg.problem);
                self.display.refreshDisplay();
            }            
        } else { // Get the task from the list matching this message
            nowCurrentAction =  {'name':planStepMsg.action, 'args': planStepMsg.args};
            for (var i=0; i<self.smachTasks.length; i+=1){
                if (self.smachTasks[i].domain == planStepMsg.domain && self.smachTasks[i].problem == planStepMsg.problem) {
                    self.smachTasks[i].currentAction = nowCurrentAction;
                }
            }
            self.display.setCurrentAction(nowCurrentAction, planStepMsg.problem);
            self.display.refreshDisplay();
        }
    };

    self.refreshState = function () {
        // TODO: Crawl problem, update display based on best available data
    };

    self.parseActionString = function (action_str) {
            action_str = action_str.slice(1, -2); // First+last(2) char are open/close parens. Remove them.
            name_args = action_str.split('(');
            name = name_args[0];
            args = name_args.slice(1);
            args = args[0].split(' ');
            return {name: name, args: args};
    };

    // Receives a list of action strings, returns a lists of action objects: [{name:'', args:['','',..]}, ... ]
    self.parseActionStrings = function(actions) {
        var act, i, name_args, name, args;
        var acts_list = [];
        for (i = 0; i < actions.length; i += 1) {
            acts_list[i] = self.parseActionString(actions[i]);
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
                                title: actionList[i].helpText
                            }
                     ).on('click.rfh', actionList[i].startFunction);
            self.$container.append(bubble);
            self.$container.append($('<div>', { class: "smach-state-separator" }));
        }
        self.$container.find('.smach-state-separator').last().remove(); // Don't need extra connector bar hanging off the end.
        var cancelButton = $('<div>', {class:"smach-state cancel", text:"Cancel"}).on('click.rfh', function (event) {self.cancelTask(currentProblem)});
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
