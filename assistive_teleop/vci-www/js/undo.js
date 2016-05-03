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
    var rEEDisplay = options.rightEEDisplay;
    var lEEDisplay = options.leftEEDisplay;
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

/*
    self.undoTopic = options.undoTopic || '/undo';
   ros.getMsgDetails('std_msgs/Int32');
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

    /*/////////////  START TASK-PLANNING UNDO FUNCTIONS ////////////////////*/
    previewFunctions['task'] = {
        start: function (undoEntry) {
            // TODO: Display undo by highlighting target state in smach display
            $('#smach-container').css('background-color','orange');
        },
        stop: function(undoEntry) {
            $('#smach-container').css('background-color','inherit');
        }
    };

    self.states['task'] = {};
    var domainStateCB = function (state) {
        self.states['task'][state.domain] = state.predicates;
    };

    var domainStepCB = function (step) {
        if (sentUndoCommands['task'][step.domain] > 0) {
            sentUndoCommands['task'][step.domain] -= 1;
            return;
        };
        var lastStepIdx = eventQueue.length - 1;
        for (lastStepIdx; lastStepIdx>=0; lastStepIdx-=1) {
            if (eventQueue[lastStepIdx].type === 'task') {
                if (eventQueue[lastStepIdx].command.problem === step.problem) {
                    break;
                }
            }
        }
        if (lastStepIdx >= 0){ // Otherwise, this is the first state, just add and leave everything before it in place
            eventQueue.splice(lastStepIdx+1, eventQueue.length-lastStepIdx); // Remove from index forward
        }

        var undoEntry = new RFH.UndoEntry({
            type: 'task',
            stateGoal: self.states['task'][step.domain],
            command: step
        });
        eventQueue.pushUndoEntry(undoEntry);
    };

    var taskStateSubs = {};
    var taskStepSubs = {};
    var setupDomainSubs = function (domain) {
        var stateSub = new ROSLIB.Topic({
            ros: ros,
            name: 'pddl_tasks/'+domain+'/state',
            messageType: 'hrl_task_planning/PDDLState'
        });
        stateSub.subscribe(domainStateCB);
        taskStateSubs[domain] = stateSub;

        var stepSub = new ROSLIB.Topic({
            ros: ros,
            name: 'pddl_tasks/'+domain+'/current_action',
            messageType: 'hrl_task_planning/PDDLPlanStep'
        });
        stepSub.subscribe(domainStepCB);
        taskStepSubs[domain] = stepSub;
    };

    sentUndoCommands['task'] = {};
    undoFunctions['task'] = function (undoEntry) {
        var domainActions = RFH.smach.getDomainData(undoEntry.command.domain).actionList;
        for (var i=0; i < domainActions.length; i += 1) {
            if (domainActions[i].name == undoEntry.command.action) {
                if (i == 0) {
                    RFH.smach.cancelTask(undoEntry.command.problem);
                    return;
                } else {
                    var goal = domainActions[i-1].goal_state;
                    continue;
                }
            }
        }
        var pddlCmd = ros.composeMsg('hrl_task_planning/PDDLProblem');
        pddlCmd.domain = undoEntry.command.domain;
        pddlCmd.name = undoEntry.command.problem;
        pddlCmd.objects = undoEntry.command.objects;
        pddlCmd.goal = goal;
        sentUndoCommands['task'][undoEntry.command.domain] += 1;
        taskCmdPub.publish(pddlCmd);
    };
    
    var taskCmdPub = new ROSLIB.Topic({
        ros: ros,
        name: 'perform_task',
        messageType: 'hrl_task_planning/PDDLProblem'
    });

    var taskActiveDomains = [];
    var activeDomainsCB = function (domains_msg) {
        var domains = domains_msg.domains;
        var newDomains = [];
        for (var i=0; i<domains.length; i+=1) {
            if (taskActiveDomains.indexOf(domains[i]) < 0){
                newDomains.push(domains[i]);
            }
        }
        for (var j=0; j<newDomains.length; j+=1) {
            setupDomainSubs(newDomains[j]);
        }
    };
    var activeDomainsSub = new ROSLIB.Topic({
        ros: ros,
        name: 'pddl_tasks/active_domains',
        messageType: 'hrl_task_planning/DomainList'
    });
    activeDomainsSub.subscribe(activeDomainsCB);

    var domainSolutions = {};
    var taskSolutionCB = function (sol_msg) {
        domainSolutions[sol_msg.domain] = {'actions': sol_msg.actions,
                                           'states': sol_msg.states}
    };
    var taskSolutionSub = new ROSLIB.Topic({
        ros: ros,
        name: '/task_solution',
        messageType: 'hrl_task_planning/PDDLSolution'
    });
    taskSolutionSub.subscribe(taskSolutionCB);

//    var taskCmdSub = new ROSLIB.Topic({
//        ros: ros,
//        name: 'perform_task',
//        messageType: 'hrl_task_planning/PDDLProblem'
//    });
//
//    var taskCmdCB = function (problem_msg) {
//        if (sentUndoCommands['task'][problem_msg.domain] > 0) {
//            sentUndoCommands['task'][problem_msg.domain] -= 1;
//            return;
//        };
//        var undoEntry = new RFH.UndoEntry({
//            type: 'task',
//            stateGoal: self.states['task'][problem_msg.domain], // TODO: Make sure this always exists...
//            command: problem_msg
//        });
//       eventQueue.pushUndoEntry(undoEntry); 
//    };
//    taskCmdSub.subscribe(taskCmdCB);

    /*/////////////  END TASK-PLANNING UNDO FUNCTIONS ////////////////////*/

    /*/////////////  START LEFT ARM UNDO FUNCTIONS ////////////////////*/
    previewFunctions['lArm'] = {
        start: function (undoEntry){
            // Display preview of goal 
            lEEDisplay.showPreviewGripper(undoEntry.stateGoal);
        }, 
        stop: function (undoEntry) {
            // Stop Display
            lEEDisplay.hidePreviewGripper();
        }
    };

    sentUndoCommands['lArm'] = 0;
    undoFunctions['lArm'] = function (undoEntry) {
        sentUndoCommands['lArm'] += 1;
        var gp = undoEntry.stateGoal;
        rArm.sendPoseGoal({position: gp.pose.position,
                           orientation: gp.pose.orientation,
                           frame_id: gp.header.frame_id
        });
    };

    var lArmCmdSub = new ROSLIB.Topic({
        ros: ros,
        name: 'left_arm/haptic_mpc/goal_pose',
        messageType: 'geometry_msgs/PoseStamped'
    });

    var lArmCmdCB = function (cmd_msg) {
        if (sentUndoCommands['lArm'] > 0) {
            sentUndoCommands['lArm'] -= 1;
            return;
        }
        var armInTorso = lArm.getState(); // Received in torso_lift_link
        armInTorso.header.frame_id = 'base_link'
        armInTorso.pose.position.x -= 0.05
        armInTorso.pose.position.z += (0.75 + torso.getState());
        var undoEntry = new RFH.UndoEntry({type: 'lArm',
                                           command: cmd_msg,
                                           stateGoal: lArm.getState()
                                           });
        eventQueue.pushUndoEntry(undoEntry);
    };

    lArmCmdSub.subscribe(lArmCmdCB);
    /*/////////////  END LEFT ARM UNDO FUNCTIONS ////////////////////*/

    /*/////////////  START RIGHT ARM UNDO FUNCTIONS ////////////////////*/
    previewFunctions['rArm'] = {
        start: function (undoEntry){
            // Display preview of goal 
            rEEDisplay.showPreviewGripper(undoEntry.stateGoal);
        }, 
        stop: function (undoEntry) {
            // Stop Display
            rEEDisplay.hidePreviewGripper();
        }
    };

    sentUndoCommands['rArm'] = 0;
    undoFunctions['rArm'] = function (undoEntry) {
        sentUndoCommands['rArm'] += 1;
        var gp = undoEntry.stateGoal;
        rArm.sendPoseGoal({position: gp.pose.position,
                           orientation: gp.pose.orientation,
                           frame_id: gp.header.frame_id
        });
    };

    var rArmCmdSub = new ROSLIB.Topic({
        ros: ros,
        name: 'right_arm/haptic_mpc/goal_pose',
        messageType: 'geometry_msgs/PoseStamped'
    });

    var rArmCmdCB = function (cmd_msg) {
        if (sentUndoCommands['rArm'] > 0) {
            sentUndoCommands['rArm'] -= 1;
            return;
        }
        var armInTorso = rArm.getState(); // Received in torso_lift_link
        armInTorso.header.frame_id = 'base_link'
        armInTorso.pose.position.x -= 0.05
        armInTorso.pose.position.z += (0.75 + torso.getState());
        var undoEntry = new RFH.UndoEntry({type: 'rArm',
                                           command: cmd_msg,
                                           stateGoal: rArm.getState()
                                           });
        eventQueue.pushUndoEntry(undoEntry);
    };

    rArmCmdSub.subscribe(rArmCmdCB);

    /*/////////////  END RIGHT ARM UNDO FUNCTIONS ////////////////////*/

    /*/////////////  START LEFT GRIPPER UNDO FUNCTIONS ////////////////////*/
    // Keep separate variable for grabbing/position.
    self.states.lGripper = lGripper.getState() < 0.01 ? 'grab' : 'position';

    var $lGripperPreviewHandleLeft = $('<span>').addClass('preview-undo ui-corner-all ui-slider-handle ui-state-default').hide();
    var $lGripperPreviewHandleRight = $('<span>').addClass('preview-undo ui-corner-all ui-slider-handle ui-state-default').hide();
    var $lGripperSlider = $('#lGripperCtrlContainer > .gripper-slider');
    $lGripperSlider.append([$lGripperPreviewHandleLeft, $lGripperPreviewHandleRight]);
    var lGripperMin = $lGripperSlider.slider('option','min');
    var lGripperMax = $lGripperSlider.slider('option','max');
    var lGripperRange = lGripperMax - lGripperMin;
    var lGripperMid = lGripperMin + lGripperRange/2;
    var lGripperHandleWidthPct = $lGripperPreviewHandleLeft.css('width').slice(0,-1);


    previewFunctions['lGripper'] = {
        start: function (undoEntry){
            var halfOpenDist = undoEntry.stateGoal === 'grab' ? lGripperMin : undoEntry.stateGoal/2;
            var halfWidthDist = lGripperRange * lGripperHandleWidthPct / 200;
            var offsetL = ((lGripperMid - halfOpenDist - halfWidthDist)/lGripperRange)*100;
            var offsetR = ((lGripperMid + halfOpenDist + halfWidthDist)/lGripperRange)*100;
            $lGripperPreviewHandleLeft.css('left', offsetL+'%').show();
            $lGripperPreviewHandleRight.css('left', offsetR+'%').show();
        }, 
        stop: function (undoEntry) {
            $lGripperPreviewHandleLeft.hide();
            $lGripperPreviewHandleRight.hide();
        }
    };

    sentUndoCommands.lGripper = 0;
    undoFunctions.lGripper = function (undoEntry) {
        sentUndoCommands.lGripper += 1;
        if (undoEntry.stateGoal === 'grab') {
            lGripper.grab();
        } else {
            lGripper.setPosition(undoEntry.stateGoal);
        }
    };

    var lGripperGenerateUndoEntry = function (cmd_msg) {
        if (sentUndoCommands.lGripper > 0) { 
            sentUndoCommands.lGripper -= 1;
            return;
        }
        var state = self.states.lGripper === 'grab' ? 'grab' : lGripper.getState();
        var undoEntry = new RFH.UndoEntry({type: 'lGripper',
                                           command: cmd_msg,
                                           stateGoal: state
                                           });
        eventQueue.pushUndoEntry(undoEntry);
    };

    var lGripperPositionCmdSub = new ROSLIB.Topic({
        ros: ros,
        name: 'l_gripper_sensor_controller/gripper_action/goal',
        messageType: "pr2_controllers_msgs/Pr2GripperCommandActionGoal"
    });
    var lGripperPositionCmdCB = function (cmd_msg) {
        lGripperGenerateUndoEntry(cmd_msg);
        if (cmd_msg.goal_id.id.indexOf("grab") == -1) { // Grab action sends position command with '....grab...' in the id
            self.states.lGripper = 'position';
        } else {
            self.states.lGripper = 'grab';
        }
    };
    lGripperPositionCmdSub.subscribe(lGripperPositionCmdCB);
         
    /*/////////////  END LEFT GRIPPER UNDO FUNCTIONS ////////////////////*/

    /*/////////////  RIGHT GRIPPER UNDO FUNCTIONS ////////////////////*/
    // Keep separate variable for grabbing/position.
    self.states.rGripper = 'position';

    var $rGripperPreviewHandleLeft = $('<span>').addClass('preview-undo ui-corner-all ui-slider-handle ui-state-default').hide();
    var $rGripperPreviewHandleRight = $('<span>').addClass('preview-undo ui-corner-all ui-slider-handle ui-state-default').hide();
    var $rGripperSlider = $('#rGripperCtrlContainer > .gripper-slider');
    var rGripperHandles = $rGripperSlider.find('.ui-slider-handle');
    $rGripperSlider.append([$rGripperPreviewHandleLeft, $rGripperPreviewHandleRight]);
    var rGripperMin = $rGripperSlider.slider('option','min');
    var rGripperMax = $rGripperSlider.slider('option','max');
    var rGripperRange = rGripperMax - rGripperMin;
    var rGripperMid = rGripperMin + rGripperRange/2;
    var rGripperHandleWidthPct = $rGripperPreviewHandleLeft.css('width').slice(0,-1);
    var rGripperStopPreview = true;

    previewFunctions['rGripper'] = {
        start: function (undoEntry){
            var currentOffsets = [$(rGripperHandles[0]).css('left'), $(rGripperHandles[1]).css('left')];
            var halfOpenDist = undoEntry.stateGoal === 'grab' ? rGripperMin : undoEntry.stateGoal/2;
            var halfWidthDist = rGripperRange * rGripperHandleWidthPct / 200;
            var offsetL = ((rGripperMid - halfOpenDist - halfWidthDist)/rGripperRange)*100;
            var offsetR = ((rGripperMid + halfOpenDist + halfWidthDist)/rGripperRange)*100;
//            $rGripperPreviewHandleLeft.css('left', offsetL+'%').show();
 //           $rGripperPreviewHandleRight.css('left', offsetR+'%').show();
            rGripperStopPreview = false;
            var leftAnimation = function () {
                if (rGripperStopPreview) {return};
                $rGripperPreviewHandleLeft.css('left', currentOffsets[0]).show();
                $rGripperPreviewHandleLeft.animate({'left': offsetL+'%'}, {duration:1400, easing: 'linear', done:leftAnimation});
            };
            var rightAnimation = function () {
                if (rGripperStopPreview) {return};
                $rGripperPreviewHandleRight.css('left', currentOffsets[1]).show();
                $rGripperPreviewHandleRight.animate({'left': offsetR+'%'}, {duration:1400, easing: 'linear', done:rightAnimation});
            };
            leftAnimation();
            rightAnimation();
        }, 
        stop: function (undoEntry) {
            rGripperStopPreview = true;
            $rGripperPreviewHandleLeft.hide().stop();
            $rGripperPreviewHandleRight.hide().stop();

        }
    };

    sentUndoCommands.rGripper = 0;
    undoFunctions.rGripper = function (undoEntry) {
        sentUndoCommands.rGripper += 1;
        if (undoEntry.stateGoal === 'grab') {
            rGripper.grab();
        } else {
            rGripper.setPosition(undoEntry.stateGoal);
        }
    };

    var rGripperGenerateUndoEntry = function (cmd_msg) {
        if (sentUndoCommands.rGripper > 0) { 
            sentUndoCommands.rGripper -= 1;
            return;
        }
        var state = self.states.rGripper === 'grab' ? 'grab' : rGripper.getState();
        var undoEntry = new RFH.UndoEntry({type: 'rGripper',
                                           command: cmd_msg,
                                           stateGoal: state
                                           });
        eventQueue.pushUndoEntry(undoEntry);
    };

    var rGripperPositionCmdSub = new ROSLIB.Topic({
        ros: ros,
        name: 'r_gripper_sensor_controller/gripper_action/goal',
        messageType: "pr2_controllers_msgs/Pr2GripperCommandActionGoal"
    });
    var rGripperPositionCmdCB = function (cmd_msg) {
        rGripperGenerateUndoEntry(cmd_msg);
        if (cmd_msg.goal_id.id.indexOf("grab") == -1) { // Grab action sends position command with '....grab...' in the id
            self.states.rGripper = 'position';
        } else {
            self.states.rGripper = 'grab';
        }
    };
    rGripperPositionCmdSub.subscribe(rGripperPositionCmdCB);
         
    /*//////////////////// END GRIPPER UNDO  //////////////////////////////////*/
    
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
    };

    var $previewEyes = $('<div/>', {id:"look-preview"}).hide();
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
            return;
        }
        if (self.states.mode === state_msg.data) { // Only record if the mode is actually changing
            return;
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


