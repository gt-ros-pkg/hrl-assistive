var RFH = (function (module) {
    module.Domains = RFH.Domains || {};
    module.Domains.RealtimeBaseSelection = function (options) {
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
        var offset = {position: {x:0.14, y:0.0, z:0.0},
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
                            self.setParam('/pddl_tasks/'+self.domain+'/KNOWN/'+args[0], eeGoalPose);
                        };
                        RFH.actionMenu.actions.getClickedPoseAction.registerPoseCB(getPoseCB, true);
                        RFH.undo.sentUndoCommands.mode += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                        RFH.actionMenu.startAction('getClickedPoseAction');
                    };
                    break;
                case 'GET_FRAME':
                case 'SCAN_ENVIRONMENT':
                    startFunc = function () {
                        RFH.undo.sentUndoCommands.mode += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                        RFH.actionMenu.startAction('lookingAction');
                    };
                    break;
                case 'SERVO_OPEN_LOOP':
                    startFunc = function () {
                        RFH.undo.sentUndoCommands.mode += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                        RFH.actionMenu.startAction('drivingAction');
                    };
                    break;
                case 'ADJUST_TORSO':
                    startFunc = function () {
                        RFH.undo.sentUndoCommands.mode += 1; // Increment so this switch isn't grabbed by undo queue...(yes, ugly hack)
                        RFH.actionMenu.startAction('torsoAction');
                    };
                    break;
                    // case 'CLEAR_ENVIRONMENT':
                    // case 'CLEAR_TORSO_SET':
                    // case 'CLEAR_EE_GOAL':
                    // case 'CLEAR_BASE_GOAL':
                    // case 'CLEAR_FRAME':
                    // case 'CLEAR_AT_GOAL':
                    // case 'CALL_BASE_SELECTION':
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
                    return 'Scanning Area';
                case 'SERVO_OPEN_LOOP':
                    return 'Move Robot';
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
            }
        };

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

        self.setParam = function (paramName, data) {
            var param = new ROSLIB.Param({
                ros: ros,
                name: paramName
            });
            console.log("Setting Param " + param + ":", data);
            param.set(data);
        };

        self.clearParams = function (known_list) {
            for (var i=0; i<known_list.length; i+=1) {
                var param = new ROSLIB.Param({
                    ros: ros,
                    name: '/pddl_tasks/'+self.domain+'/KNOWN/'+known_list[i]
                });
                param.delete();
            }
        };

        self.sendTaskGoal = function (options) {
            var side = options.arm;
            var goal = options.goal;
            self.clearParams(['EE_GOAL','BASE_GOAL','EE_FRAME']);
            var ee_frame = side[0] === 'r' ? 'r_gripper_tool_frame' : 'l_gripper_tool_frame';
            self.updatePDDLState(['(NOT (AT BASE_GOAL))', 
                '(NOT (SCAN_COMPLETE))',
                '(NOT (TORSO_SET TORSO_GOAL))',
                '(NOT (KNOWN BASE_GOAL))',
                '(NOT (KNOWN TORSO_GOAL))',
                '(NOT (KNOWN EE_GOAL))',
                '(NOT (KNOWN EE_FRAME))']);
            self.setParam('/pddl_tasks/'+self.domain+'/KNOWN/EE_FRAME', ee_frame);
            var msg = ros.composeMsg('hrl_task_planning/PDDLProblem');
            msg.name = self.domain + '-' + new Date().getTime().toString();
            msg.domain = self.domain;
            msg.goal = goal || [];  // Empty goal will use default for task
            setTimeout(function(){self.taskPublisher.publish(msg);}, 750); // Wait for everything else to settle first...
        };

        var applyOffset = function (pose_msg) {
            var quat = new THREE.Quaternion(pose_msg.pose.orientation.x,
                pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z,
                pose_msg.pose.orientation.w);
            var poseMat = new THREE.Matrix4().makeRotationFromQuaternion(quat);
            poseMat.setPosition(new THREE.Vector3(pose_msg.pose.position.x,
                pose_msg.pose.position.y,
                pose_msg.pose.position.z));

            var offsetMat = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(offset.rotation.x,
                offset.rotation.y,
                offset.rotation.z));
            offsetMat.setPosition(new THREE.Vector3(offset.position.x, offset.position.y, offset.position.z));


            var trans = new THREE.Matrix4();
            var scale = new THREE.Vector3();
            var quat_out = new THREE.Quaternion();
            poseMat.multiply(offsetMat);
            poseMat.decompose(trans, quat_out, scale);
            var result_pose = ros.composeMsg('geometry_msgs/PoseStamped');
            result_pose.header = pose_msg.header;
            result_pose.pose.position.x = trans.x;
            result_pose.pose.position.y = trans.y;
            result_pose.pose.position.z = trans.z;
            result_pose.pose.orientation.x = quat_out.x;
            result_pose.pose.orientation.y = quat_out.y;
            result_pose.pose.orientation.z = quat_out.z;
            result_pose.pose.orientation.w = quat_out.w;
            return result_pose;
        };
    };
    return module;
})(RFH || {});
