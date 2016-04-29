RFH.Domains = RFH.Domains || {};
RFH.Domains.PickAndPlace = function (options) {
    'use strict';
    var self = this;
    var ros = options.ros;
    self.name = options.name || 'pick_and_place';
    self.domain = 'pick_and_place'
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
        name: '/pddl_tasks/pick_and_place/state_updates',
        messageType: '/hrl_task_planning/PDDLState'
    });
    self.pddlStateUpdatePub.advertise();
    
    self.updatePDDLState = function(pred_array, domain){
        var msg = ros.composeMsg('hrl_task_planning/PDDLState');
        msg.domain = domain || self.domain;
        msg.predicates = pred_array;
        self.pddlStateUpdatePub.publish(msg);
    };

    self.getActionFunction = function (name, args) {
        if (args[0] === 'RIGHT_HAND') {
            return function () {RFH.taskMenu.startTask('rEECartTask');}
        } else if (args[0] === 'LEFT_HAND') {
            return function () {RFH.taskMenu.startTask('lEECartTask');}
        };
    };

    self.getActionLabel = function (name, args) {
        switch (name){
            case 'PICK':
                return "Pickup Object";
            case 'PLACE':
                return "Place Object";
        }
    };

    self.getActionHelpText = function (name, args) {
        switch (name){
            case 'PICK':
                return "Use the controls to pick up the needed object.";
            case 'PLACE':
                return "Use the controls to set down the object";
        }
    };

    self.clearLocationParams = function (loc_list) {
        for (var i=0; i<loc_list.length; i+=1) {
            var param = new ROSLIB.Param({
                ros: ros,
                name: '/pddl_tasks/'+self.domain+'/KNOWN/'+loc_list[i]
            });
            param.delete();
            if (RFH.regions[param.name] !== undefined) {
                RFH.regions[param.name].remove();
            }
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
        self.clearLocationParams(['PICK_LOC']);
        var msg = ros.composeMsg('hrl_task_planning/PDDLProblem');
        msg.name = 'pick_and_place' + '-' + new Date().getTime().toString();
        msg.domain = 'pick_and_place';
        var hand = side.toUpperCase()+'_HAND';
        var otherHand = hand === 'LEFT_HAND' ? 'RIGHT_HAND' : 'LEFT_HAND'
        var object = hand + '_OBJECT';
        self.setDefaultGoal(['(PLACED '+object+')']);
        self.updatePDDLState(['(NOT (PLACED '+object+'))','(CAN-GRASP '+hand+')', '(NOT (CAN-GRASP '+otherHand+'))']);
        self.updatePDDLState(['(CAN-GRASP '+hand+')', '(NOT (CAN-GRASP '+otherHand+'))'], 'pick');
        msg.goal = []; 
        setTimeout(function(){self.taskPublisher.publish(msg);}, 1000); // Wait for everything else to settle first...
    };

};
