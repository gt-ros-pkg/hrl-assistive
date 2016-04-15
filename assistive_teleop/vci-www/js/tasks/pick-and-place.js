RFH.PickAndPlace = function (options) {
    'use strict';
    var self = this;
    var ros = options.ros;
    self.arm = options.arm;
    self.gripper = options.gripper;
    self.side = self.gripper.side;
    self.name = options.name || 'pick_and_place_'+self.side;
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

    self.startAction = function (planStepMsg) {
        switch (planStepMsg.action){
            case 'ID-LOCATION':
                if (planStepMsg.args[0] == 'PLACE_LOC' ||
                    planStepMsg.args[0] == 'PICK_LOC' ||
                    planStepMsg.args[0] == 'ELSEWHERE'){
                    RFH.taskMenu.tasks.paramLocationTask.setOffset({'position':{'x':0, 'y':0, 'z':0.1}});
                    RFH.taskMenu.tasks.paramLocationTask.setOrientationOverride({'x':0, 'y':0, 'z':0.38, 'w':0.925});
                } else {
                    RFH.taskMenu.tasks.paramLocationTask.setOffset({}); // No offset
                    RFH.taskMenu.tasks.paramLocationTask.setOrientationOverride(null); // No override
                    RFH.taskMenu.tasks.paramLocationTask.setPositionOverride(null); // No override
                }
                RFH.taskMenu.tasks.paramLocationTask.setParam('/pddl_tasks/'+self.domain+'/KNOWN/'+planStepMsg.args[0]);
                RFH.taskMenu.startTask('paramLocationTask');
                break;
            case 'FORGET-LOCATION':
                RFH.taskMenu.startTask('LookingTask');
                break;
            case 'MOVE-ARM':
                RFH.taskMenu.startTask(self.side.substring(0,1)+'EECartTask');
                break;
            case 'GRAB':
                RFH.taskMenu.startTask(self.side.substring(0,1)+'EECartTask');
                break;
            case 'RELEASE':
                RFH.taskMenu.startTask(self.side.substring(0,1)+'EECartTask');
                break;
        }
    };

    self.getActionLabel = function (action) {
        var loc;
        switch (action.name){
            case 'ID-LOCATION':
                if (action.args[0].indexOf('PICK') >= 0) {
                    loc = 'Pickup';
                } else if (action.args[0].indexOf('PLACE') >= 0) {
                    loc = 'Place';
                } else {
                    loc = 'Empty';
                }
                return "Indicate %loc Location".replace('%loc', loc);
            case 'FORGET-LOCATION':
                switch (action.args[0]) {
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
                switch (action.args[1]) {
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


    self.getActionHelpText = function (action) {
        switch (action.name){
            case 'ID-LOCATION':
                if (action.args[0].indexOf('PICK') >= 0) {
                    loc = 'the item you wish to pick up';
                } else if (action.args[0].indexOf('PLACE') >= 0) {
                    loc = 'the spot where you want to place the item';
                } else {
                    loc = 'a clear spot on a surface';
                }
                return "Click on %loc. If not visible, click the gray edges to look around.".replace('%loc', loc);
            case 'FORGET-LOCATION':
                switch (action.args[0]) {
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
                switch (action.args[1]) {
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
        }
    };

    self.sendTaskGoal = function () {
        self.clearLocationParams(['HAND_START_LOC', 'PICK_LOC', 'PLACE_LOC', 'ELSEWHERE']);
        var msg = ros.composeMsg('hrl_task_planning/PDDLProblem');
        msg.name = 'pick_and_place'+'-'+ new Date().getTime().toString();
        msg.domain = 'pick_and_place';
        var hand = self.side.toUpperCase()+'_HAND';
        msg.objects = [hand +' - GRIPPER'];
        self.setPoseToParam(self.arm.getState(), 'HAND_START_LOC');
        self.updatePDDLState(['(NOT (AT TARGET PLACE_LOC))', '(AT TARGET PICK_LOC)']);
        if (!self.gripper.getGrasping()) {
            msg.init.push('(AT TARGET PICK_LOC)');
        }
        msg.goal = [];  // Empty goal will use default for task
        self.taskPublisher.publish(msg);
    };

};
