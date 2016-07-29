RFH.Domains = RFH.Domains || {};
RFH.Domains.WipingMouthADL = function (options) {
    'use strict';
    var self = this;
    var ros = options.ros;
    self.name = options.name || 'wiping_mouth_adl';
    self.domain = 'wiping_mouth_adl'
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
            case 'CHECK_BED_OCCUPANCY':
            case 'REGISTER_HEAD':
            case 'CALL_BASE_SELECTION':
                startFunc = function () {
                    RFH.undo.sentUndoCommands['mode'] += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                    RFH.taskMenu.startTask('LookingTask');
                }
                break;
            case 'MOVE_ROBOT':
                startFunc = function () {
                    RFH.undo.sentUndoCommands['mode'] += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                    RFH.taskMenu.startTask('drivingTask');
                }
                break;
            case 'CONFIGURE_BED_ROBOT':
                startFunc = function () {
                    RFH.undo.sentUndoCommands['mode'] += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                    RFH.taskMenu.startTask('torsoTask');
                }
                break;
            case 'MOVE_ARM':
                startFunc = function () {
                    RFH.undo.sentUndoCommands['mode'] += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                    RFH.taskMenu.startTask('rEECartTask');
                }
                break;
        }
        return startFunc;
    };


    self.getActionLabel = function (name, args) {
        switch (name){
            case 'FIND_TAG':
                return "Finding AR Tag.";
            case 'TRACK_TAG':
                return "Tracking AR Tag.";
            case 'CHECK_BED_OCCUPANCY':
                return "Checking Bed Occupancy.";
            case 'REGISTER_HEAD':
                return "Registering Occupant Head.";
            case 'CALL_BASE_SELECTION':
                return "Calling Base Selection.";
            case 'CONFIGURE_BED_ROBOT':
                return "Configuring Bed and Robot.";
            case 'MOVE_ROBOT':
                return "Moving PR2 Base.";
            case 'MOVE_ARM':
                return "Moving PR2 Arm."
            case 'DO_TASK':
                return "Manual Task.";
        }
    };

    self.getActionHelpText = function (name, args) {
        switch (name){
            case 'FIND_TAG':
                return "Use the controls to look at the AR Tag attached to the bed.";
            case 'TRACK_TAG':
                return "Currently Tracking AR Tag.";
            case 'CHECK_BED_OCCUPANCY':
                return "Checking if the bed is occupied. Please occupy Autobed to proceed.";
            case 'REGISTER_HEAD':
                return "Trying to find your head in the mat. Please rest your head on the bed.";
            case 'CALL_BASE_SELECTION':
                return "Please wait while the PR2 finds a good location to perform task...";
            case 'CONFIGURE_BED_ROBOT':
                return "Please wait while we finish repositioning your bed and the robot's height...";
            case 'MOVE_ROBOT':
                return "Please wait while the robot moves towards you. Please keep RUN STOP handy...";
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
        self.setDefaultGoal(['(TASK-COMPLETED)']);
        self.updatePDDLState(['(NOT (CONFIGURED BED)']);
        msg.goal = []; 
        setTimeout(function(){self.taskPublisher.publish(msg);}, 1000); // Wait for everything else to settle first...
    };

};
