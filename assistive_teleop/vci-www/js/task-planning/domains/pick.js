var RFH = (function (module) {
    module.Domains = RFH.Domains || {};
    module.Domains.Pick = function (options) {
        'use strict';
        var self = this;
        var ros = options.ros;
        self.arms = {'right': options.r_arm,
            'left': options.l_arm};
        self.grippers = {'right': options.r_gripper,
            'left': options.l_gripper};
        self.name = options.name || 'pick';
        self.domain = 'pick';
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
                case 'CHOOSE-OBJECT':
                    startFunc = function () {
                        var getPoseCB = function (pose) {
                            self.setParam('/pddl_tasks/'+self.domain+'/KNOWN/'+args[0], pose);
                        };
                        RFH.actionMenu.tasks.getClickedPoseAction.registerPoseCB(getPoseCB, true);
                        RFH.undo.sentUndoCommands.mode += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                        RFH.actionMenu.startAction('getClickedPoseAction');
                    };
                    break;
                case 'FORGET-OBJECT':
                    startFunc = function () {
                        RFH.undo.sentUndoCommands.mode += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                        RFH.actionMenu.startAction('LookingAction');
                    };
                    break;
                case 'AUTO-GRASP':
                case 'MANUAL-GRASP':
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
                case 'CHOOSE-OBJECT':
                    return "Select Object";
                case 'FORGET-OBJECT':
                    return "Clear Object Selection";
                case "AUTO-GRASP":
                    return "Automatic Grasp";
                case 'MANUAL-GRASP':
                    return "Manual Grasp";
            }
        };

        self.getActionHelpText = function (name, args) {
            switch (name){
                case 'CHOOSE-OBJECT':
                    return "Click on the object to grasp.";
                case 'FORGET-OBJECT':
                    return "Clear saved object location.";
                case "AUTO-GRASP":
                    return "Wait for the automated grasping to retrieve the object.";
                case 'MANUAL-GRASP':
                    return "Use the controls to grasp the object.";
            }
        };

        self.setPoseToParam = function (ps_msg, location_name) {
            var poseParam = new ROSLIB.Param({
                ros: ros,
                name: '/pddl_tasks/'+self.domain+'/CHOSEN-OBJ/'+location_name
            });
            console.log("Setting " + location_name + " pose as:", ps_msg);
            poseParam.set(ps_msg);
        };

        self.clearLocationParams = function (loc_list) {
            for (var i=0; i<loc_list.length; i+=1) {
                var param = new ROSLIB.Param({
                    ros: ros,
                    name: '/pddl_tasks/'+self.domain+'/CHOSEN-OBJ/'+loc_list[i]
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

        self.sendTaskGoal = function (side, goal) {
            goal = goal || [];  // Empty goal will use default for task
            var hand = side.toUpperCase()+'_HAND';
            var otherHand = hand === 'LEFT_HAND' ? 'RIGHT_HAND' : 'LEFT_HAND';
            var object = hand + '_OBJECT';
            self.clearLocationParams([object]);
            self.setDefaultGoal(['(GRASPING '+hand+' '+object+')']);
            self.updatePDDLState(['(NOT (GRASPING '+hand+' '+object+'))','(CAN-GRASP '+hand+')', '(NOT (CAN-GRASP '+otherHand+'))','(NOT (AUTO-GRASP-DONE))','(NOT (CHOSEN-OBJ '+object+'))']);
            var msg = ros.composeMsg('hrl_task_planning/PDDLProblem');
            msg.name = 'pick' + '-' + new Date().getTime().toString();
            msg.domain = 'pick';
            msg.goal = goal;
            setTimeout(function(){self.taskPublisher.publish(msg);}, 1000); // Wait for everything else to settle first...
        };
    };
    return module;
})(RFH || {});
