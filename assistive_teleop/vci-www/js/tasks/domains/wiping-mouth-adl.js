RFH.Domains = RFH.Domains || {};
RFH.Domains.WipingMouthADL = function (options) {
    'use strict';
    var self = this;
    var ros = options.ros;
    self.name = options.name || 'wiping_mouth_adl';
    self.domain = 'wiping_mouth_adl'
    var $button = $('#wiping-mouth-adl-button');
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

    self.getActionFunction = function (name, args) {
        var startFunc;
        switch (name){
            case 'FIND_TAG':
            case 'TRACK_TAG':
            case 'CHECK_OCCUPANCY':
            case 'REGISTER_HEAD':
            case 'CALL_BASE_SELECTION':
                startFunc = function () {
                    RFH.undo.sentUndoCommands['mode'] += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                    RFH.taskMenu.startTask('LookingTask');
                }
                break;
            case 'MOVE_BACK':
            case 'MOVE_ROBOT':
                startFunc = function () {
                    RFH.undo.sentUndoCommands['mode'] += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                    RFH.taskMenu.startTask('drivingTask');
                }
                break;
            case 'CONFIGURE_MODEL_ROBOT':
                startFunc = function () {
                    RFH.undo.sentUndoCommands['mode'] += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                    RFH.taskMenu.startTask('torsoTask');
                }
                break;
            case 'MOVE_ARM':
            case 'DO_TASK':
                startFunc = function () {
                    RFH.undo.sentUndoCommands['mode'] += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                    RFH.taskMenu.startTask('lEECartTask');
                }
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
                return "Moving Arm"
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

    var waitForParamUpdate = function (param, value, delayMS) {
        var param = new ROSLIB.Param({
            ros: ros,
            name: param
        });
        var flag = false;
        var checkFN = function () {
            if (param.get() === value) { 
                flag = true;
            } else {
                setTimeout(checkFN, delayMS);
            }
        }
        setTimeout(checkFN, delayMS);
    };

    self.sendTaskGoal = function (side, goal) {
        goal = goal || []; // Empty goal will use default for task
        self.clearParams([]);
        var msg = ros.composeMsg('hrl_task_planning/PDDLProblem');
        msg.name = 'wiping_mouth_adl' + '-' + new Date().getTime().toString();
        msg.domain = 'wiping_mouth_adl';
        var model = 'autobed';
        var task = 'wiping_mouth';
        self.setModelName(model);
        self.setDefaultGoal(['(TASK-COMPLETED WIPING_MOUTH AUTOBED)']);
        self.updatePDDLState(['(NOT (CONFIGURED BED WIPING_MOUTH AUTOBED))', 
                              '(NOT (BASE-SELECTED WIPING_MOUTH AUTOBED))',
                              '(NOT (IS-TRACKING-TAG AUTOBED))',
                              '(NOT (CONFIGURED SPINE WIPING_MOUTH AUTOBED))',
                              '(NOT (HEAD-REGISTERED AUTOBED))',
                              '(NOT (OCCUPIED AUTOBED))',
                              '(NOT (FOUND-TAG AUTOBED))',
                              '(NOT (BASE-REACHED WIPING_MOUTH AUTOBED))',
                              '(NOT (ARM-REACHED WIPING_MOUTH AUTOBED))',
                              '(NOT (ARM-HOME WIPING_MOUTH AUTOBED))',
                              '(NOT (TOO-CLOSE AUTOBED))',
                              '(NOT (TASK-COMPLETED WIPING_MOUTH AUTOBED))']);
        msg.goal = []; 
        setTimeout(function(){self.taskPublisher.publish(msg);}, 1000); // Wait for everything else to settle first...
    };
    $button.button().on('click', self.sendTaskGoal);

};
