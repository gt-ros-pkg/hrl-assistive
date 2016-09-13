var RFH = (function (module) {
    module.Domains = RFH.Domains || {};
    module.Domains.ToolPick = function (options) {
        'use strict';
        var self = this;
        var ros = options.ros;
        self.arms = {'right': options.r_arm,
            'left': options.l_arm};
        self.grippers = {'right': options.r_gripper,
            'left': options.l_gripper};
        self.name = options.name || 'tool_pick';
        self.domain = 'tool_pick';
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
                case 'PLACE':
                case 'RESET-AUTO-TRIED':
                case 'FIND-TAG':
                    startFunc = function () {
                        RFH.undo.sentUndoCommands.mode += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                        RFH.actionMenu.startAction('LookingAction');
                    };
                    break;
                case 'AUTO-TOOL-GRASP':
                case 'MANUAL-TOOL-GRASP':
                    if (args[0] === 'RIGHT_HAND') {
                        startFunc = function () {
                            RFH.undo.sentUndoCommands.mode += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                            RFH.actionMenu.startAction('rEECartAction');
                        };
                    } else if (args[0] === 'LEFT_HAND') {
                        startFunc = function () {
                            RFH.undo.sentUndoCommands.mode += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                            RFH.actionMenu.startAction('lEECartAction');
                        };
                    }
                    break;
            }
            return startFunc;
        };

        self.getActionLabel = function (name, args) {
            switch (name){
                case 'AUTO-GRASP-TOOL':
                    return "Auto-grasp";
                case 'MANUAL-GRASP-TOOL':
                    return "Grasp Manually";
                case 'FIND-TAG':
                    return "Find Tool Tag";
                case "RESET-AUTO-TRIED":
                    return "Reset auto-grasp";
                case 'PLACE':
                    return "Set Down Item";
            }
        };

        self.getActionHelpText = function (name, args) {
            switch (name){
                case 'AUTO-GRASP-TOOL':
                    return "Please wait while the robot tries to grasp the tool.";
                case 'MANUAL-GRASP-TOOL':
                    return "Grasp the desired tool manually using the arm controls.";
                case 'FIND-TAG':
                    return "Move the robot so that it has a clear view of the AR Tag on the desired tool.";
                case "PLACE":
                    return "Guide the robot in setting down the object it's currently holding.";
                case 'RESET-AUTO-TRIED':
                    return "Please Wait while the robot forgets that it already tried";
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

        self.sendTaskGoal = function (options) {
            var hand = options.arm.toUpperCase() === 'L' ? 'LEFT_HAND' : 'RIGHT_HAND';
            var tool = options.tool.toUpperCase();
            var goal = [];
            self.setDefaultGoal(['(GRASPING '+tool+' '+hand+')']);
            self.updatePDDLState(['(NOT (AUTO-GRASP-DONE))']);
            var msg = ros.composeMsg('hrl_task_planning/PDDLProblem');
            msg.name = self.domain + '-' + new Date().getTime().toString();
            msg.objects = [tool + ' - TOOL'];
            msg.domain = self.domain;
            msg.goal = goal;
            setTimeout(function(){self.taskPublisher.publish(msg);}, 1000); // Wait for everything else to settle first...
        };
    };
    return module;
})(RFH || {});
