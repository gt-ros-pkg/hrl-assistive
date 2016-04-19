RFH.PickAndPlace = function (options) {
    'use strict';
    var self = this;
    var ros = options.ros;
    self.arms = {'right': options.r_arm,
                 'left': options.l_arm};
    self.grippers = {'right': options.r_gripper,
                     'left': options.l_gripper};
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
                if (args[0] == 'PLACE_LOC' || args[0] == 'PICK_LOC' || args[0] == 'ELSEWHERE'){
                    startFunc = function () {
                        RFH.taskMenu.tasks.paramLocationTask.setOffset({'position':{'x':0, 'y':0, 'z':0.1}});
                        //RFH.taskMenu.tasks.paramLocationTask.setOrientationOverride({'x':0, 'y':0, 'z':0.38, 'w':0.925});
                        RFH.taskMenu.tasks.paramLocationTask.setOrientationOverride(null);
                        RFH.taskMenu.tasks.paramLocationTask.setParam('/pddl_tasks/'+self.domain+'/KNOWN/'+args[0]);
                        RFH.taskMenu.startTask('paramLocationTask');
                    }
                } else {
                    startFunc = function () {
                        RFH.taskMenu.tasks.paramLocationTask.setOffset({}); // No offset
                        RFH.taskMenu.tasks.paramLocationTask.setOrientationOverride(null); // No override
                        RFH.taskMenu.tasks.paramLocationTask.setPositionOverride(null); // No override
                        RFH.taskMenu.tasks.paramLocationTask.setParam('/pddl_tasks/'+self.domain+'/KNOWN/'+args[0]);
                        RFH.taskMenu.startTask('paramLocationTask');
                    }
                }
                break;
            case 'FORGET-LOCATION':
                startFunc = function () {
                    RFH.taskMenu.startTask('LookingTask');
                }
                break;
            case 'MOVE-ARM':
            case 'GRAB':
            case 'RELEASE':
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
        var loc;
        switch (name){
            case 'ID-LOCATION':
                if (args[0].indexOf('PICK') >= 0) {
                    loc = 'Pickup';
                } else if (args[0].indexOf('PLACE') >= 0) {
                    loc = 'Place';
                } else {
                    loc = 'Empty';
                }
                return "Indicate %loc Location".replace('%loc', loc);
            case 'FORGET-LOCATION':
                switch (args[0]) {
                    case 'PICK_LOC':
                        loc = 'Pickup';
                        break;
                    case 'PLACE_LOC':
                        loc = 'Place';
                        break;
                    case 'HAND_START_LOC':
                        loc = 'Original Hand';
                        break;
                    case 'ELSEWHERE':
                        loc = 'any available';
                        break;
                }
                return "Clear Saved %loc Location".replace('%loc', loc);
            case 'MOVE-ARM':
                switch (args[1]) {
                    case 'PICK_LOC':
                        loc = 'pickup';
                        break;
                    case 'PLACE_LOC':
                        loc = 'place';
                        break;
                    case 'HAND_START_LOC':
                        loc = 'original hand';
                        break;
                    case 'ELSEWHERE':
                        loc = 'any available';
                        break;
                }
                return "Approach %loc location".replace('%loc', loc);
            case "GRAB":
                return "Grab Item";
            case 'RELEASE':
                return "Set Item Down";
        }
    };

    self.getActionHelpText = function (name, args) {
        var loc
        switch (name){
            case 'ID-LOCATION':
                if (args[0].indexOf('PICK') >= 0) {
                    loc = 'the item you wish to pick up';
                } else if (args[0].indexOf('PLACE') >= 0) {
                    loc = 'the spot where you want to place the item';
                } else {
                    loc = 'a clear spot on a surface';
                }
                return "Click on %loc. If not visible, click the gray edges to look around.".replace('%loc', loc);
            case 'FORGET-LOCATION':
                switch (args[0]) {
                    case 'PICK_LOC':
                        loc = 'the pickup';
                        break;
                    case 'PLACE_LOC':
                        loc = 'the place';
                        break;
                    case 'HAND_START_LOC':
                        loc = 'the original hand';
                        break;
                    case 'ELSEWHERE':
                        loc = 'the extra';
                        break;
                }
                return "Clears %loc location".replace('%loc', loc);
            case 'MOVE-ARM':
                switch (args[1]) {
                    case 'PICK_LOC':
                        loc = 'pickup';
                        break;
                    case 'PLACE_LOC':
                        loc = 'place';
                        break;
                    case 'HAND_START_LOC':
                        loc = 'original hand';
                        break;
                    case 'ELSEWHERE':
                        loc = 'extra empty';
                        break;
                }
                return "Use the arm controls to move the arm near the %loc location".replace('%loc', loc);
            case "GRAB":
                return "Use the arm and gripper controls to grasp the desired item.";
            case 'RELEASE':
                return "Use the arm and gripper controls to release the item.";
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
//        waitForParamUpdate(paramName, goal_pred_list, 100);
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

    self.sendTaskGoal = function (side) {
        self.clearLocationParams(['HAND_START_LOC', 'PICK_LOC', 'PLACE_LOC', 'ELSEWHERE']);
        var msg = ros.composeMsg('hrl_task_planning/PDDLProblem');
        msg.name = 'pick_and_place' + '-' + new Date().getTime().toString();
        msg.domain = 'pick_and_place';
        var hand = side.toUpperCase()+'_HAND';
        var otherHand = hand === 'LEFT_HAND' ? 'RIGHT_HAND' : 'LEFT_HAND'
        var object = hand + '_OBJECT';
        self.setDefaultGoal(['(AT '+object+' PLACE_LOC)']);
        self.updatePDDLState(['(NOT (AT '+object+' PLACE_LOC))','(CAN-GRASP '+hand+')', '(NOT (CAN-GRASP '+otherHand+'))']);
        self.setPoseToParam(self.arms[side].getState(), 'HAND_START_LOC');
        if (!self.grippers[side].getGrasping()) {
            self.updatePDDLState(['(AT '+object+' PICK_LOC)']);
        } else {
            self.updatePDDLState(['(GRASPING '+hand+' '+object+')']);
        }
        msg.goal = [];  // Empty goal will use default for task
        self.taskPublisher.publish(msg);
    };

};
