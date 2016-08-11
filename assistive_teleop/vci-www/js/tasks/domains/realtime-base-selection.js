RFH.Domains = RFH.Domains || {};
RFH.Domains.RealtimeBaseSelection = function (options) {
    'use strict';
    var self = this;
    var ros = options.ros;
    self.name = options.name || 'realtime_base_selection';
    self.domain = 'realtime_base_selection';
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
        msg.domain = self.domain;
        msg.predicates = pred_array;
        self.pddlStateUpdatePub.publish(msg);
    };

    self.getActionFunction = function (name, args) {
        var startFunc;
        switch (name){
            case 'GET_EE_GOAL':
                startFunc = function () {
                    RFH.taskMenu.tasks.paramLocationTask.setOffset({}); // No offset
                    RFH.taskMenu.tasks.paramLocationTask.setOrientationOverride(null); // No override
                    RFH.taskMenu.tasks.paramLocationTask.setPositionOverride(null); // No override
                    RFH.taskMenu.tasks.paramLocationTask.setParam('/pddl_tasks/'+self.domain+'/KNOWN/'+args[0]);
                    RFH.undo.sentUndoCommands['mode'] += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                    RFH.taskMenu.startTask('paramLocationTask');
                }
                break;
            case 'GET_FRAME':
            case 'SCAN_ENVIRONMENT':
                startFunc = function () {
                    RFH.undo.sentUndoCommands['mode'] += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                    RFH.taskMenu.startTask('LookingTask');
                }
                break;
            case 'SERVO_OPEN_LOOP':
                startFunc = function () {
                    RFH.undo.sentUndoCommands['mode'] += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                    RFH.taskMenu.startTask('DrivingTask');
                }
                break;
            case 'ADJUST_TORSO':
                startFunc = function () {
                    RFH.undo.sentUndoCommands['mode'] += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                    RFH.taskMenu.startTask('torsoTask');
                }
                break;
            case 'CLEAR_ENVIRONMENT':
            case 'CLEAR_TORSO_SET':
            case 'CLEAR_EE_GOAL':
            case 'CLEAR_BASE_GOAL':
            case 'CLEAR_FRAME':
            case 'CLEAR_AT_GOAL':
            case 'CALL_BASE_SELECTION':
            default:
                startFunc = function () {};
        }
        return startFunc;
    };

    self.getActionLabel = function (name, args) {
        switch (name){
            case 'GET_EE_GOAL':
                return "Select Object";
            case 'GET_FRAME':
                return "Choose Hand";
            case 'SCAN_ENVIRONMENT':
                return 'Scan Area';
            case 'SERVO_OPEN_LOOP':
                return 'Move Robot';
            case 'ADJUST_TORSO':
                return 'Adjust Torso';
            case 'CLEAR_ENVIRONMENT':
                return 'Clear Scan';
            case 'CLEAR_TORSO_SET':
                return 'Clear Torso Status';
            case 'CLEAR_EE_GOAL':
                return "Clear Object Selection";
            case 'CLEAR_BASE_GOAL':
                return 'Clear Result';
            case 'CLEAR_FRAME':
                return 'Clear Hand Selection';
            case 'CALL_BASE_SELECTION':
                return 'Get Good Position';
        };
    }

    self.getActionHelpText = function (name, args) {
        switch (name){
            case 'GET_EE_GOAL':
                return "Click on the position or object that you want to manipulate with the gripper.";
            case 'GET_FRAME':
                return "Choose which hand to use.";
            case 'SCAN_ENVIRONMENT':
                return 'Please wait while the robot looks around to find a good position.';
            case 'SERVO_OPEN_LOOP':
                return 'Carefully monitor the robot while it moves into position.';
            case 'ADJUST_TORSO':
                return 'Carefully monitor the robot while it adjusts the torso height.';
            case 'CLEAR_ENVIRONMENT':
                return 'Please wait while the robot clears the scanned environment from its memory.';
            case 'CLEAR_TORSO_SET':
                return 'Please wait while the robot clears the desired torso location from its memory.';
            case 'CLEAR_EE_GOAL':
                return "Please wait while the robot clears the desired position or object from its memory.";
            case 'CLEAR_BASE_GOAL':
                return 'Please wait while the robot clears the identified goal base position from its memory.';
            case 'CLEAR_AT_GOAL':
                return 'Please wait while the robot clears the previous result from its memory.';
            case 'CLEAR_FRAME':
                return 'Please wait while the robot clears the choice of hand from its memory';
            case 'CALL_BASE_SELECTION':
                return 'Please wait while the robot determines a good location to position its base.';
        }
    };

    self.setPoseToParam = function (ps_msg, location_name) {
        self.setParam('/pddl_tasks/'+self.domain+'/KNOWN/'+location_name, ps_msg);
    };

    self.setParam = function (param, data) {
        var poseParam = new ROSLIB.Param({
            ros: ros,
            name: param
        });
        console.log("Setting Param " + param + ":", ps_msg);
        poseParam.set(data);
    }

    self.clearParams = function (known_list) {
        for (var i=0; i<known_list.length; i+=1) {
            var param = new ROSLIB.Param({
                ros: ros,
                name: '/pddl_tasks/'+self.domain+'/KNOWN/'+known_list[i]
            });
            param.delete();
        }
    };

//    self.setDefaultGoal = function (goal_pred_list) {
//        var paramName = '/pddl_tasks/'+self.domain+'/default_goal';
//        var goalParam = new ROSLIB.Param({
//            ros: ros,
//            name: paramName
//        });
//        goalParam.set(goal_pred_list);
//    };

    self.sendTaskGoal = function (side, goal) {
        self.clearParams(['EE_GOAL','BASE_GOAL','EE_FRAME']);
        var ee_frame = side[0] === 'r' ? 'r_gripper_tool_frame' : 'l_gripper_tool_frame'
        self.setParam('/pddl_tasks/'+self.domain+'/KNOWN/EE_FRAME', ee_frame);
//        self.setDefaultGoal(['(AT BASE_GOAL)']);
        self.updatePDDLState(['(NOT (AT BASE_GOAL))', 
                              '(NOT (SCAN_COMPLETE))',
                              '(NOT (TORSO_SET BASE_GOAL))',
                              '(NOT (KNOWN BASE_GOAL))',
                              '(NOT (KNOWN EE_GOAL))',
                              '(KNOWN EE_FRAME)']);
        var msg = ros.composeMsg('hrl_task_planning/PDDLProblem');
        msg.name = self.domain + '-' + new Date().getTime().toString();
        msg.domain = self.domain;
        msg.goal = goal || [];  // Empty goal will use default for task
        setTimeout(function(){self.taskPublisher.publish(msg);}, 750); // Wait for everything else to settle first...
    };
};
