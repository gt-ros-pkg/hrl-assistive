RFH.UndoEntry = function (options) {
    'use strict';
    options = options || {};
    var self = this;   
    self.time = options.time || new Date();
    self.type = options.type;
    self.command = options.command;
    self.stateGoal = options.stateGoal;
};

RFH.Undo = function (options) {
    'use strict';
    var self = this;
    options = options || {};
    var ros = options.ros;
    var torso = options.torso || new PR2Torso(ros);
    var rGripper = options.rGripper || new PR2GripperSensor({side:'right', ros:ros});
    var lGripper = options.lGripper || new PR2GripperSensor({side:'left', ros:ros});
    var head = options.head || new PR2Head({ros: ros,
                                            limits: [[-2.85, 2.85], [1.18, -0.38]],
                                            joints: ['head_pan_joint', 'head_tilt_joint'],
                                            pointingFrame: 'head_mount_kinect_rgb_optical_frame',
                                            undoSetActiveService: 'undo/move_head/set_active'});
    var rArm = options.rArm || new PR2ArmMPC({side:'right',
                                              ros: ros,
                                              stateTopic: 'right_arm/haptic_mpc/gripper_pose',
                                              poseGoalTopic: 'right_arm/haptic_mpc/goal_pose',
                                              ee_frame:'r_gripper_tool_frame',
                                              trajectoryGoalTopic: '/right_arm/haptic_mpc/joint_trajectory',
                                              plannerServiceName:'/moveit_plan/right_arm'});
    var lArm = options.rArm || new PR2ArmMPC({side:'left',
                                              ros: ros,
                                              stateTopic: 'left_arm/haptic_mpc/gripper_pose',
                                              poseGoalTopic: 'left_arm/haptic_mpc/goal_pose',
                                              ee_frame:'l_gripper_tool_frame',
                                              trajectoryGoalTopic: '/left_arm/haptic_mpc/joint_trajectory',
                                              plannerServiceName:'/moveit_plan/left_arm'});

//    self.undoTopic = options.undoTopic || '/undo';
    var $undoButton = $('#'+options.buttonDiv).button();
    // Set up states registry
    var eventQueue = [];
    eventQueue.pushUndoEntry = function (undoEntry) {
        $undoButton.show();
        eventQueue.push(undoEntry);
    };
    eventQueue.popUndoEntry = function () {
        if (eventQueue.length == 1) { $undoButton.hide(); }
        return eventQueue.pop();
    };

    self.states = {};
    self.states.mode = null;
    var previewFunctions = {};
    var undoFunctions = {};
    var sentUndoCommands = {};

    // Preview the undo action that will take place
    var startPreview = function (undoEvent) {
        previewFunctions[undoEvent.type]['start'](undoEvent);
    };

    var stopPreview = function (undoEvent) {
        previewFunctions[undoEvent.type]['stop'](undoEvent);
    };

    var undoHoverStartCB = function(event) {
        startPreview(eventQueue[eventQueue.length-1]);
    }

    var undoHoverStopCB = function(event) {
        if (eventQueue.length > 0) {
            stopPreview(eventQueue[eventQueue.length-1]);
        }
    }

    $undoButton.hover(undoHoverStartCB, undoHoverStopCB).hide();

 /*   ros.getMsgDetails('std_msgs/Int32');
    var undoPub =  new ROSLIB.Topic({
        ros: ros,
        name: self.undoTopic,
        messageType: 'std_msgs/Int32'
    });
    undoPub.advertise();

    var sendUndoCommand = function (numSteps) {
        numSteps = numSteps === undefined ? 1 : numSteps;
        console.log("Sending command to undo " + numSteps.toString() + " step(s).");
        self.undoPub.publish({'data': numSteps});
    };
*/

    var undoButtonCB = function ( ) {
        var eventToReverse = eventQueue.popUndoEntry(); // Get event to undo
        stopPreview(eventToReverse);  // Stop Preview
        undoFunctions[eventToReverse.type](eventToReverse); // Send command to reverse state
        if (eventQueue.length > 0) {
            var nextUndo = eventQueue[eventQueue.length-1]  // Get next event in list
            startPreview(nextUndo); // Preview for next button click
        }
    };
    $undoButton.on('click.rfh', undoButtonCB); 


    // Handle Task Goals

    // rArm 
    previewFunctions['rArm'] = function (goal){
        // Display preview of goal 
    };

    // lArm
    previewFunctions['lArm'] = function (goal){
        // Display preview of goal 
    };


    // Handle Gripper Goals
    //TODO: Add preview functions for undo mouseover
    previewFunctions['rGripper'] = function (goal){
        // Display preview of goal 
    };
    ros.getMsgDetails('pr2_controllers_msgs/Pr2GripperCommandGoal');

    var gripperStateToGoal = function (state_msg) {
        // TODO: Fill out based on gripper-sensor-action interface/options
    }

    self.setPosition = function (pos, effort) {
        if (grabGoal !== null) { grabGoal.cancel(); }
        var msg = ros.composeMsg('pr2_controllers_msgs/Pr2GripperCommandGoal');
        msg.command.position = pos;
        msg.command.max_effort = effort || -1;
        var goal = new ROSLIB.Goal({
            actionClient: positionActionClient,
            goalMessage: msg
        });
        goal.send();
    };
         
    var rGripperCmdSub = new ROSLIB.Topic({
        ros: ros,
        name: 'r_gripper_sensor_controller/gripper_action/goal',
        messageType: "pr2_controllers_msgs/Pr2GripperCommandGoal"
    });
    var rGripperCmdCB = function (cmd_msg) {
        var stateGoal = gripperStateToGoal(self.rGripper.getState());
        undoEntry.command = cmd_msg;
        undoEntry.stateGoal = {};// TODO FILL IN CORRECTLY
        var undoEntry = new RFH.UndoEntry({type: 'rGripper',
                                           command: cmd_msg,
                                           stateGoal: stateGoal
                                           });
        eventQueue.pushUndoEntry(undoEntry);
    };
    rGripperCmdSub.subscribe(rGripperCmdCB);

    
    /*//////////////////// Handle Looking goals /////////////////////////*/
    previewFunctions['look'] = {
        start: function (undoEntry) {
            var px = RFH.mjpeg.cameraModel.projectPoint(undoEntry.stateGoal.x,
                                                        undoEntry.stateGoal.y,
                                                        undoEntry.stateGoal.z,
                                                        'base_link');
            var lrClass = 'center';
            lrClass = px[0] < 0 ? "left" : lrClass;
            lrClass = px[0] > 1 ? "right" : lrClass;
            var udClass = "middle";
            udClass = px[1] < 0 ? "top" : udClass;
            udClass = px[1] > 1 ? "bottom" : udClass;
            if (lrClass === "center" && udClass === "middle" ) {
                $previewEyes.css({left:100*px[0]+'%', top:100*px[1]+'%'}).show();
            } else {
                $('.map-look.'+lrClass+'.'+udClass).addClass('preview-undo');
            }
        },
        stop: function (undoEntry) {
            $('.map-look.preview-undo').removeClass('preview-undo');
            $previewEyes.hide();
        }
    };

    sentUndoCommands['look'] = 0;
    undoFunctions['look'] = function (undoEntry) {
        sentUndoCommands['look'] += 1;
        head.pointHead(undoEntry.stateGoal.x,
                       undoEntry.stateGoal.y,
                       undoEntry.stateGoal.z,
                       '/base_link');
//        head.setPosition(undoEntry.stateGoal[0], undoEntry.stateGoal[1]);
    };

    var $previewEyes = $('<div/>', {id:"look-preview"})
    $('#video-main').append($previewEyes);

    var headCmdSub = new ROSLIB.Topic({
        ros: ros,
        name: "/head_traj_controller/command",
        messageType: "trajectory_msgs/JointTrajectory"
    });
    var headCmdCB = function (traj_msg) {
        if (self.states.mode == 'rEECartTask' || 
            self.states.mode == 'lEECartTask') {
                return; 
        };
        if (sentUndoCommands['look'] > 0 ) {
            sentUndoCommands['look'] -= 1;
            return;
        }
        var camModel = RFH.mjpeg.cameraModel;
        var pt3d = new THREE.Vector3(0, 0, 2);
        var camQuat = new THREE.Quaternion(camModel.transform.rotation.x,
                                           camModel.transform.rotation.y,
                                           camModel.transform.rotation.z,
                                           camModel.transform.rotation.w);
        var camPos = new THREE.Vector3(camModel.transform.translation.x,
                                       camModel.transform.translation.y,
                                       camModel.transform.translation.z);
        var camTF = new THREE.Matrix4().makeRotationFromQuaternion(camQuat);
        camTF.setPosition(camPos);
        pt3d.applyMatrix4(camTF);
        var undoEntry = new RFH.UndoEntry({
            type: 'look',
            stateGoal: pt3d,
            command: traj_msg
        });
        eventQueue.pushUndoEntry(undoEntry);
    };
    headCmdSub.subscribe(headCmdCB);

    /* //////////////////// END LOOKING UNDO /////////////////////// */

   /*///////////////// Interface Mode Switching Management /////////////////////*/
    previewFunctions['mode'] = {
        start: function (undoEntry){
                   $('.menu-item.'+undoEntry.stateGoal).addClass('preview-undo');
                    // Display preview of goal 
                },
        stop: function (undoEntry) {
                   $('.menu-item.'+undoEntry.stateGoal).removeClass('preview-undo');
        }
    };
    
    sentUndoCommands['mode'] = 0; // Initialize on list
    undoFunctions['mode'] = function (undoEntry) {
        sentUndoCommands['mode'] += 1  //Increment counter so we know to expect incoming commands
        RFH.taskMenu.startTask(undoEntry.stateGoal);
    };

    var modeChangeSub = new ROSLIB.Topic({
        ros: ros,
        name: '/web_teleop/current_mode',
        messageType: 'std_msgs/String'
    });
    var modeChangeCB = function (state_msg) {
        if (self.states.mode === null) { 
            self.states.mode = state_msg.data;
            return;
        };
        if (sentUndoCommands['mode'] > 0) {  
            sentUndoCommands['mode'] -= 1; // Ignore commands from this module undoing previous commands..
            self.states.mode = state_msg.data; // Keep updated state for later reference
            return
        }
        // Handle standard case, recording command to undo later
        var undoEntry = new RFH.UndoEntry({
            stateGoal: self.states.mode,
            command: state_msg.data,
            type: 'mode'
        });
        eventQueue.pushUndoEntry(undoEntry);
        self.states.mode = state_msg.data; // Update current mode
    };
    modeChangeSub.subscribe(modeChangeCB);
    /////////////////// END MODE SWITCHING //////////////////////////////////

    /*////////////////////// Torso Commands ///////////////////////////////////*/
    var $torsoPreviewHandle = $('<span>').addClass('preview-undo ui-corner-all ui-slider-handle ui-state-default').hide();
    $('#torsoSlider').append($torsoPreviewHandle)
    var torsoMin = $('#torsoSlider').slider('option', 'min');
    var torsoMax = $('#torsoSlider').slider('option', 'max');
    var torsoRange = torsoMax - torsoMin;
    previewFunctions['torso'] = {
        start: function (undoEntry) {
            var offsetPct = 100 * (undoEntry.stateGoal - torsoMin) / (torsoRange);
            $torsoPreviewHandle.css('bottom', offsetPct+'%');
            $torsoPreviewHandle.show();
        },
        stop: function (undoEntry) {
            $torsoPreviewHandle.hide();
        }
    };

    sentUndoCommands['torso'] = 0; // Initialize counter
    undoFunctions['torso'] = function (undoEntry) {
        sentUndoCommands['torso'] += 1;
        torso.setPosition(undoEntry.stateGoal);
    };

    var torsoCmdCB = function (cmdMsg) {
        if (sentUndoCommands['torso'] > 0) {
            sentUndoCommands['torso'] -= 1;
            return;
        }
        var undoEntry = new RFH.UndoEntry({
            type: 'torso',
            stateGoal: torso.getState(),
            command: cmdMsg
        });
        eventQueue.pushUndoEntry(undoEntry);
    };

    var torsoCmdSub = new ROSLIB.Topic({
        ros: ros,
        name: '/torso_controller/command',
        messageType:'trajectory_msgs/JointTrajectory'
    });
    torsoCmdSub.subscribe(torsoCmdCB);
    /////////////  END TORSO UNDO ///////////////////////////////////////
}


