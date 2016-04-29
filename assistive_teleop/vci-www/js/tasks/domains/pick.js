RFH.Domains = RFH.Domains || {};
RFH.Domains.Pick = function (options) {
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
            case 'ID-LOCATION':
                startFunc = function () {
                    RFH.taskMenu.tasks.paramLocationTask.setOffset({}); // No offset
                    RFH.taskMenu.tasks.paramLocationTask.setOrientationOverride(null); // No override
                    RFH.taskMenu.tasks.paramLocationTask.setPositionOverride(null); // No override
                    RFH.taskMenu.tasks.paramLocationTask.setParam('/pddl_tasks/'+self.domain+'/KNOWN/'+args[0]);
                    RFH.taskMenu.startTask('paramLocationTask');
                }
                break;
            case 'FORGET-LOCATION':
                startFunc = function () {
                    RFH.taskMenu.startTask('LookingTask');
                }
                break;
            case 'AUTO-GRASP':
            case 'MANUAL-GRASP':
                if (args[0] === 'RIGHT_HAND') {
                    startFunc = function () {
                        RFH.taskMenu.startTask('rEECartTask');
                    }
                } else if (args[0] === 'LEFT_HAND') {
                    startFunc = function () {
                        RFH.taskMenu.startTask('lEECartTask');
                    }
                };
                break;
        }
        return startFunc;
    };

    self.getActionLabel = function (name, args) {
        switch (name){
            case 'ID-LOCATION':
                return "Select Object";
            case 'FORGET-LOCATION':
                return "Clear Object Selection";
            case "AUTO-GRASP":
                return "Automatic Grasp";
            case 'MANUAL-GRASP':
                return "Manual Grasp";
        };
    }

    self.getActionHelpText = function (name, args) {
        switch (name){
            case 'ID-LOCATION':
                return "Click on the object to grasp.";
            case 'FORGET-LOCATION':
                return "Clear saved object location.";
            case "AUTO-GRASP":
                return "Wait for the automated grasping to retrieve the object."
            case 'MANUAL-GRASP':
                return "Use the controls to grasp the object.";
        }
    };

    self.setPoseToParam = function (ps_msg, location_name) {
        var poseParam = new ROSLIB.Param({
            ros: ros,
            name: '/pddl_tasks/'+self.domain+'/KNOWN/'+location_name
        });
        console.log("Setting " + location_name + " pose as:", ps_msg);
        poseParam.set(ps_msg);
    };

    self.clearLocationParams = function (loc_list) {
        for (var i=0; i<loc_list.length; i+=1) {
            var param = new ROSLIB.Param({
                ros: ros,
                name: '/pddl_tasks/'+self.domain+'/KNOWN/'+loc_list[i]
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
        var otherHand = hand === 'LEFT_HAND' ? 'RIGHT_HAND' : 'LEFT_HAND'
        var object = hand + '_OBJECT';
        self.clearLocationParams([object]);
        self.setDefaultGoal(['(GRASPING '+hand+' '+object+')']);
        self.updatePDDLState(['(NOT (GRASPING '+hand+' '+object+'))','(CAN-GRASP '+hand+')', '(NOT (CAN-GRASP '+otherHand+'))','(NOT (AUTO-GRASP-DONE))','(NOT (KNOWN '+object+'))']);
        var msg = ros.composeMsg('hrl_task_planning/PDDLProblem');
        msg.name = 'pick' + '-' + new Date().getTime().toString();
        msg.domain = 'pick';
        msg.goal = goal;
        setTimeout(function(){self.taskPublisher.publish(msg);}, 1000); // Wait for everything else to settle first...
    };

};
