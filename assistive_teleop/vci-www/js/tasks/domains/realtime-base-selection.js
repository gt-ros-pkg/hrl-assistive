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
    var offset = {position: {x:0.06, y:0.0, z:0.0},
                  rotation: {x:0.0, y:Math.PI, z:0.0}};

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
                    var getPoseCB = function (pose) {
                        var eeGoalPose = applyOffset(pose);
                        self.setParam('/pddl_tasks/'+self.domain+'/KNOWN/'+args[0], pose);
                    };
                    RFH.taskMenu.tasks.getClickedPoseTask.registerPoseCB(getPoseCB, true);
                    RFH.undo.sentUndoCommands['mode'] += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                    RFH.taskMenu.startTask('getClickedPoseTask');
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
                return 'Scannig Area';
            case 'SERVO_OPEN_LOOP':
                return 'Moving Robot';
            case 'ADJUST_TORSO':
                return 'Adjusting Torso';
            case 'CLEAR_ENVIRONMENT':
                return 'Clearing Scan';
            case 'CLEAR_TORSO_SET':
                return 'Clearing Torso Status';
            case 'CLEAR_EE_GOAL':
                return "Clearing Object Selection";
            case 'CLEAR_BASE_GOAL':
                return 'Clearing Result';
            case 'CLEAR_FRAME':
                return 'Clearing Hand Selection';
            case 'CALL_BASE_SELECTION':
                return 'Getting Good Position';
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
        var param = new ROSLIB.Param({
            ros: ros,
            name: param
        });
        console.log("Setting Param " + param + ":", data);
        param.set(data);
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

    self.sendTaskGoal = function (side, goal) {
        self.clearParams(['EE_GOAL','BASE_GOAL','EE_FRAME']);
        var ee_frame = side[0] === 'r' ? 'r_gripper_tool_frame' : 'l_gripper_tool_frame'
        self.setParam('/pddl_tasks/'+self.domain+'/KNOWN/EE_FRAME', ee_frame);
        self.updatePDDLState(['(NOT (AT BASE_GOAL))', 
                              '(NOT (SCAN_COMPLETE))',
                              '(NOT (TORSO_SET BASE_GOAL))',
                              '(NOT (KNOWN BASE_GOAL))',
                              '(NOT (KNOWN EE_GOAL))',
                              '(NOT (KNOWN EE_FRAME))']);
        var msg = ros.composeMsg('hrl_task_planning/PDDLProblem');
        msg.name = self.domain + '-' + new Date().getTime().toString();
        msg.domain = self.domain;
        msg.goal = goal || [];  // Empty goal will use default for task
        setTimeout(function(){self.taskPublisher.publish(msg);}, 750); // Wait for everything else to settle first...
    };

    var applyOffset = function (pose_msg) {
        var pose = pose_msg.pose;
        var quat = new THREE.Quaternion(pose.orientation.x,
                                        pose.orientation.y,
                                        pose.orientation.z,
                                        pose.orientation.w);
        var poseRotMat = new THREE.Matrix4().makeRotationFromQuaternion(quat);
        var offsetVec = new THREE.Vector3(offset.position.x, 
                                          offset.position.y,
                                          offset.position.z); //Get to x dist from point along normal
        offsetVec.applyMatrix4(poseRotMat);
        var desRotMat = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(offset.rotation.x,
                                                                                  offset.rotation.y,
                                                                                  offset.rotation.z));
        poseRotMat.multiply(desRotMat);
        poseRotMat.setPosition(new THREE.Vector3(pose.position.x + offsetVec.x,
                                                 pose.position.y + offsetVec.y,
                                                 pose.position.z + offsetVec.z));
        var trans = new THREE.Matrix4();
        var scale = new THREE.Vector3();
        poseRotMat.decompose(trans, quat, scale);
        pose.position.x = trans.x;
        pose.position.y = trans.y;
        pose.position.z = trans.z;
        pose.orientation.x = quat.x;
        pose.orientation.y = quat.y;
        pose.orientation.z = quat.z;
        pose.orientation.w = quat.w;
        return pose;
    };

};
