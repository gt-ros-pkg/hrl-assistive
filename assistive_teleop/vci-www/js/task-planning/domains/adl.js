var RFH = (function(module) {
    module.Domains = RFH.Domains || {};
    module.Domains.ADL = function (options) {
        'use strict';
        var self = this;
        var ros = options.ros;
        var tfClient = options.tfClient;
        var $viewer = options.viewer;
        self.name = options.name || 'adl';
        self.domain = 'adl';
        var $button = $('#start-task-button');
        ros.getMsgDetails('hrl_task_planning/PDDLProblem');
        self.taskPublisher = new ROSLIB.Topic({
            ros: ros,
            name: '/perform_task',
            messageType: 'hrl_task_planning/PDDLProblem'
        });
        self.taskPublisher.advertise();

        ros.getMsgDetails('hrl_task_planning/PDDLState');
        self.pddlStateUpdatePub = new ROSLIB.Topic({
            ros: ros,
            name: '/pddl_tasks/state_updates',
            messageType: '/hrl_task_planning/PDDLState'
        });
        self.pddlStateUpdatePub.advertise();
        
        self.updatePDDLState = function(pred_array){
            var msg = ros.composeMsg('hrl_task_planning/PDDLState');
            msg.predicates = pred_array;
            self.pddlStateUpdatePub.publish(msg);
        };

        self.zoneDisplay = new RFH.DriveZoneDisplay({
            ros: ros,
            tfClient: tfClient,
            viewer: $viewer
        });
        self.showZone = self.zoneDisplay.show;
        self.hideZone = self.zoneDisplay.hide;


        self.getActionFunction = function (name, args) {
            var startFunc;
            switch (name){
                case 'FIND_TAG':
                case 'TRACK_TAG':
                case 'CHECK_OCCUPANCY':
                case 'REGISTER_HEAD':
                case 'CALL_BASE_SELECTION':
                    startFunc = function () {
                        RFH.undo.sentUndoCommands.mode += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                        RFH.actionMenu.startAction('LookingAction');
                    };
                    break;
                case 'MOVE_BACK':
                    startFunc = function () {
                    self.showZone();
                    RFH.actionMenu.startAction('LookingAction');
                    };
                    break;
                case 'MOVE_ROBOT':
                    startFunc = function () {
                        RFH.undo.sentUndoCommands.mode += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                        RFH.actionMenu.startAction('drivingAction');
                    };
                    break;
                case 'CONFIGURE_MODEL_ROBOT':
                    startFunc = function () {
                        self.hideZone();
                        RFH.undo.sentUndoCommands.mode += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                        RFH.actionMenu.startAction('torsoAction');
                    };
                    break;
                case 'MOVE_ARM':
                case 'DO_TASK':
                    startFunc = function () {
                        self.hideZone();
                        RFH.undo.sentUndoCommands.mode += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                        RFH.actionMenu.startAction('lEECartAction');
                    };
                    break;
            }
            return startFunc;
        };

        self.getActionLabel = function (name, args) {
            switch (name){
                case 'FIND_TAG':
                    return "Finding Tag";
                case 'TRACK_TAG':
                    return "Tracking Tag";
                case 'CHECK_OCCUPANCY':
                    return "Bed Occ";
                case 'REGISTER_HEAD':
                    return "Register Head";
                case 'CALL_BASE_SELECTION':
                    return "Base Select";
                case 'CONFIGURE_MODEL_ROBOT':
                    return "Setup Bed & Robot";
                case 'MOVE_BACK':
                    return "Move Back";
                case 'MOVE_ROBOT':
                    return "Moving Base";
                case 'STOP_TRACKING':
                    return "Stop Tracking";
                case 'MOVE_ARM':
                    return "Moving Arm";
                case 'DO_TASK':
                    return "Manual Task";
            }
        };

        self.getActionHelpText = function (name, args) {
            switch (name){
                case 'FIND_TAG':
                    return "Use the controls to look at the AR Tag attached to the bed.";
                case 'TRACK_TAG':
                    return "Currently Tracking AR Tag.";
                case 'CHECK_OCCUPANCY':
                    return "Checking if the bed is occupied. Please occupy Autobed to proceed.";
                case 'REGISTER_HEAD':
                    return "Trying to find your head in the mat. Please rest your head on the bed.";
                case 'CALL_BASE_SELECTION':
                    return "Please wait while the PR2 finds a good location to perform task...";
                case 'CONFIGURE_MODEL_ROBOT':
                    return "Please wait while we finish repositioning your bed and the robot's height...";
                case 'MOVE_BACK':
                    return "Move back, you must!";
                case 'MOVE_ROBOT':
                    return "Please wait while the robot moves towards you. Please keep RUN STOP handy...";
                case 'STOP_TRACKING':
                    return "Stopping AR Tag Tracking";
                case 'MOVE_ARM':
                    return "Robot moving its arm towards your face. Please wait patiently...";
                case 'DO_TASK':
                    return "Use the controls to move the arm and perform the task.";
            }
        };

        self.clearParams = function (paramList) {
            for (var i=0; i<paramList.length; i+=1) {
                var param = new ROSLIB.Param({
                    ros: ros,
                    name: paramList[i]
                });
                param.delete();
            }
        };

        self.setModelName = function (model_name) {
            var paramName = '/pddl_tasks/'+self.domain+'/model_name';
            var modelParam = new ROSLIB.Param({
                ros: ros,
                name: paramName
            });
            modelParam.set(model_name);
        };


        self.setDefaultGoal = function (goal_pred_list) {
            var paramName = '/pddl_tasks/'+self.domain+'/default_goal';
            var goalParam = new ROSLIB.Param({
                ros: ros,
                name: paramName
            });
            goalParam.set(goal_pred_list);
        };

        var waitForParamUpdate = function (paramName, value, delayMS) {
            var param = new ROSLIB.Param({
                ros: ros,
                name: paramName
            });
            var flag = false;
            var checkFN = function () {
                if (param.get() === value) { 
                    flag = true;
                } else {
                    setTimeout(checkFN, delayMS);
                }
            };
            setTimeout(checkFN, delayMS);
        };

        self.sendTaskGoal = function (options) {
            var task = options.task;
            var model = options.model;
            var goal = options.goal || []; // Empty goal will use default for task
            var model_upper = model.toUpperCase();
            var task_upper = task.toUpperCase();
            self.clearParams([]);
            var msg = ros.composeMsg('hrl_task_planning/PDDLProblem');
            msg.name = 'adl' + '-' + new Date().getTime().toString();
            msg.domain = 'adl';
            self.setModelName(model);
            self.setDefaultGoal(['(TASK-COMPLETED '.concat(task_upper, ' ',model_upper, ')')]);
            self.updatePDDLState(['(NOT (CONFIGURED BED '.concat(task_upper, ' ',model_upper, '))'), 
                                  '(NOT (BASE-SELECTED '.concat(task_upper, ' ', model_upper, '))'),
                                  '(NOT (IS-TRACKING-TAG '.concat(model_upper,'))'),
                                  '(NOT (CONFIGURED SPINE '.concat(task_upper, ' ', model_upper,'))'),
                                  '(NOT (HEAD-REGISTERED '.concat(model_upper,'))'),
                                  '(NOT (OCCUPIED '.concat(model_upper, '))'),
                                  '(NOT (FOUND-TAG '.concat(model_upper, '))'),
                                  '(NOT (BASE-REACHED '.concat(task_upper, ' ', model_upper,'))'),
                                  '(NOT (ARM-REACHED '.concat(task_upper, ' ', model_upper, '))'),
                                  '(NOT (ARM-HOME '.concat(task_upper, ' ', model_upper, '))'),
                                  '(TOO-CLOSE '.concat(model_upper, ')'),
                                  '(NOT (TASK-COMPLETED '.concat(task_upper, ' ', model_upper, '))')]);
            msg.goal = []; 
            setTimeout(function(){self.taskPublisher.publish(msg);}, 1000); // Wait for everything else to settle first...
        };
    };
    return module;

})(RFH || {});
