var RFH = (function(module) {
    module.Smach = function(options) {
        var self = this;
        var ros = options.ros;
        self.$displayContainer = options.displayContainer || $('#scmah-display');
        self.display = new RFH.SmachDisplay({ros: ros,
                                             container: self.$displayContainer});
        self.domains = {};
        self.currentActionSubscribers = {};
        self.solutionSubscribers = {};

        var preemptTaskClient = new ROSLIB.Service({ros:ros,
                                                     name:'preempt_pddl_task',
                                                     serviceType: '/hrl_task_planning/PreemptTask'});

        // Send a preempt request to task_smacher for the problem name
        self.cancelTask = function (problem) {
            var cancelResultCB = function (resp) {
                if (resp.result) {
                    RFH.actionMenu.startAction(RFH.actionMenu.defaultAction);
                    console.log("Cancelled task successfully");
                } else {
                    RFH.log("Failed to cancel task");
                }
            };

            var req = new ROSLIB.ServiceRequest({'problem_name':problem});
            preemptTaskClient.callService(req, cancelResultCB);
        };
        self.display.cancelTask = self.cancelTask;

        // Add a newly active domain to the internal list, and subscribe to its solution and current action topics
        self.setupNewDomain = function (domain) {
            self.setupCurrentActionSubscriber(domain);
            self.setupSolutionSubscriber(domain);
        };

        // Remove a domain from the internal list, close subscribers for its solution and current_action topics
        self.cleanupDomain = function (domain) {
            if (self.currentActionSubscribers[domain]){
                self.currentActionSubscribers[domain].unsubscribe(); // Warning: removes all subs in rosjs, may be dangerous
                delete self.currentActionSubscribers[domain];
            }
            if (self.solutionSubscribers[domain]) {
                self.solutionSubscribers[domain].unsubscribe();
                delete self.solutionSubscribers[domain];
            }
            delete self.domains[domain];
            self.updateInterface();
        };

        // Subscribe to the list of currently active domains, and update internal list
        var activeDomainsCB = function (domains_msg) {
            var newDomains = domains_msg.domains;
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
            }
        };
        var activeDomainsSubscriber = new ROSLIB.Topic({
            ros: ros,
            name: '/pddl_tasks/active_domains',
            messageType: '/hrl_task_planning/DomainList'
        });
        activeDomainsSubscriber.subscribe(activeDomainsCB);

        // Get current problem name from task_smacher (display lowest leaf action in current problem tree)
        self.currentProblem = null;
        var updateCurrentProblem = function (msg) {
            self.currentProblem = msg.data === '' ? null : msg.data;
            self.display.setCurrentProblem(self.currentProblem);
            self.updateInterface();
        };
        var currentProblemSubscriber = new ROSLIB.Topic({
                                            ros: ros, 
                                            name: '/pddl_tasks/current_problem',
                                            messageType: 'std_msgs/String'});
        currentProblemSubscriber.subscribe(updateCurrentProblem);


        // Create a subscriber for the current_action of an active domain, and add to the internal list for future reference
        self.setupCurrentActionSubscriber = function (domain) {
            self.currentActionSubscribers[domain] = new ROSLIB.Topic({
                                                        ros: ros,
                                                        name: '/pddl_tasks/'+domain+'/current_action',
                                                        messageType: '/hrl_task_planning/PDDLPlanStep'
                                                    });
            self.currentActionSubscribers[domain].subscribe(function(msg){self.updateCurrentAction(domain, msg);});
        };

        self.updateCurrentAction = function (domain, planStepMsg) {
            self.domains[domain] = self.domains[domain] || {}; // Initialize if needed
            self.domains[domain].problem = planStepMsg.problem;
            self.domains[domain].currentAction = {'name': planStepMsg.action, 'args':planStepMsg.args};
            self.updateInterface(); // We have some new information, which might effect the interface, so update
        };

        // Create a subscriber for the current solution of an active domain, and add to an internal list for future reference
        self.setupSolutionSubscriber = function (domain) {
            self.solutionSubscribers[domain] = new ROSLIB.Topic({
                                                    ros: ros,
                                                    name: '/pddl_tasks/'+domain+'/solution',
                                                    messageType: 'hrl_task_planning/PDDLSolution'
                                                });
            self.solutionSubscribers[domain].subscribe(function(msg){self.updateSolution(domain, msg);});
        };

        self.updateSolution = function(domain, msg) {
            self.domains[domain] = self.domains[domain] || {}; // Initialize if needed
            // Set the problem this domain currently applies to
            self.domains[domain].problem = msg.problem;
            // Get action meta-data for interface, add to solution information in domain list
            var actions = self.parseActionStrings(msg.actions);
            var domainDetails = RFH.taskMenu.domains[domain];
            for (var i=0; i<actions.length; i+=1) {
                actions[i].label = domainDetails.getActionLabel(actions[i].name, actions[i].args);
                actions[i].helpText = domainDetails.getActionHelpText(actions[i].name, actions[i].args);
                actions[i].startFunction = domainDetails.getActionFunction(actions[i].name, actions[i].args);
                actions[i].init_state = msg.states[i];
                actions[i].goal_state = msg.states[i+1];
            }
            self.domains[domain].solution_steps = actions;
            // We have some new information, which might effect the interface, so update
            self.updateInterface();
        };

        self.updateInterface = function () {
            // If there is no known current problem, clear the display, 
            if (self.currentProblem === null ) {
                self.display.empty();
                return; 
            }

            // Identify active domains relevant to this problem, with a known current action
            var relevantDomains = getProblemDomains(self.currentProblem);
            if (relevantDomains.length === 0) { return; } // Still waiting for domain data
            for (var i=0; i<relevantDomains.length; i += 1) {
                var dom = relevantDomains[i];
                if (!self.domains[dom].currentAction || ! self.domains[dom].solution_steps) { return; } 
            }
            // Find the leaf domain + action of the domain/action tree
            var leafDomainName = getLeafDomain(relevantDomains);
            var leafDomain = self.domains[leafDomainName];
            var actIdx = self.getActionIndex(leafDomain.currentAction, leafDomain.solution_steps);
            if (actIdx < 0) { return; } // Desired action not in existing plan...

            // Send leaf domain data to display, update, run.
            self.display.setActionList(leafDomain.solution_steps);
            self.display.setCurrentAction(leafDomain.currentAction);
            self.display.refreshDisplay();
            leafDomain.solution_steps[actIdx].startFunction();
        };

        self.parseActionString = function (action_str) {
                action_str = action_str.slice(1, -2); // First+last(2) char are open/close parens. Remove them.
                name_args = action_str.split('(');
                name = name_args[0];
                args = name_args.slice(1);
                args = args[0] === "" ? [] : args[0].split(' ');  // Return empty lift if no arg, else a list of args
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

        self.getPriorState = function () {
            var relevantDomains = getProblemDomains(self.currentProblem);
            if (relevantDomains.length === 0) { return; } // Still waiting for domain data
            for (var i=0; i<relevantDomains.length; i += 1) {
                var dom = relevantDomains[i];
                if (!self.domains[dom].currentAction || ! self.domains[dom].solution_steps) { return null; } 
            }
            // Find the leaf action of the domain/action tree
            var leafDomain = getLeafDomain(relevantDomains);
            // Get action index, if 0, repeat on next-higher domain OR return null
            do {
                var currentActionIndex = self.getActionIndex(self.domains[leafDomain].currentAction, self.domains[leafDomain].solution_steps);
                if (currentActionIndex === 0) {
                    leafDomain = getParentDomain(leafDomain);
                } else {
                   return self.domains[leafDomain].solution_steps[currentActionIndex - 1].init_state; 
                }
            } while (leafDomain !== null);
            return null; // Currently in 1st state in top-level domain
        };

        self.getActionIndex = function (action, actionList) {
            for (var idx in actionList) {
                if (action.name !== actionList[idx].name) { continue; }
                var argsMatch = true;
                for (var argInd in actionList[idx].args) {
                    if (action.args[argInd] !== actionList[idx].args[argInd]) { 
                        argsMatch = false;
                        continue;
                    }
                }
                if (argsMatch) { return idx; }
            }
            return -1;
        };

        var getLeafDomain = function (relevantDomains) {
            for (var idx in relevantDomains) {
                var domAction = self.domains[relevantDomains[idx]].currentAction;
                if (relevantDomains.indexOf(domAction.name.toLowerCase()) > 0) {
                    continue; // The currently active action in this domain is it's own domain...
                } else {
                   return relevantDomains[idx]; // No sub-task for the current action here, must be the one to display!
                }
            }
            return null;
        };

        // Get the known domain for which this domain is a sub-action (if any)
        var getParentDomain = function (domain) {
            var problemDomains = getProblemDomains(self.domains[domain].problem); 
            for (var idx in problemDomains) {
                if (!self.domains[problemDomains[idx]].solution_steps) {continue;}
                var solution = self.domains[problemDomains[idx]].solution_steps;
                for (var action in solution) {
                    if (solution[action].name.toLowerCase() === domain) { return problemDomains[idx]; }
                }
            }
            return null; // Found nothing in loop above...
        };

        // Get all of the known domains which are used in the given problem
        var getProblemDomains = function (problem) {
            var relevantDomains = [];
            for (var domain in self.domains) {
                if (self.domains[domain].problem === problem) {
                    relevantDomains.push(domain); 
                }
            }
            return relevantDomains;
        };

    };

    module.SmachDisplay = function(options) {
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

        self.setCurrentAction = function (action) {
            currentAction = action;
        };

        self.getCurrentAction = function () {
            return currentAction;
        };

        self.setCurrentProblem = function (problem) {
            currentProblem = problem;
        };

        self.getCurrentProblem = function () {
            return currentProblem;
        };

        var getCurrentActionIndex = function () {
            for (var idx in actionList) {
                if (currentAction.name !== actionList[idx].name) { continue; }
                var argsMatch = true;
                for (var argInd in actionList[idx].args) {
                    if (currentAction.args[argInd] !== actionList[idx].args[argInd]) { 
                        argsMatch = false;
                        continue;
                    }
                }
                if (argsMatch) { return idx; }
            }
            return -1;
        };

        var makeClickCB = function (idx, fn) {
            var onclickFn = function (event) {
                var clickedDiv = $('#smach-container > .smach-state').get(idx);
                if ( $(clickedDiv).hasClass('incomplete') ) {
                    event.stopPropagation();
                } else {
                   fn(event); 
                }
            };
            return onclickFn;
        };

        self.refreshDisplay = function (){
            var currentIdx = getCurrentActionIndex();
            if (currentIdx < 0) { return; }
            self.empty();
            if (currentProblem === null) { return; }
            var bubble;
            for (var i=0; i<actionList.length; i +=1) {
               bubble = $('<div>', {class: "smach-state incomplete",
                                    text: actionList[i].label,
                                    title: actionList[i].helpText
                                }
                         ).on('click.rfh', makeClickCB(i, actionList[i].startFunction));
                self.$container.append(bubble);
                self.$container.append($('<div>', { class: "smach-state-separator" }));
            }
            self.$container.find('.smach-state-separator').last().remove(); // Don't need extra connector bar hanging off the end.
            var cancelButton = $('<div>', {class:"smach-state cancel", text:"Cancel"}).on('click.rfh', function (event) {self.cancelTask(currentProblem);});
            self.$container.append(cancelButton);
            self.setActive(currentIdx);
        };

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
    return module;

})(RFH || {});
