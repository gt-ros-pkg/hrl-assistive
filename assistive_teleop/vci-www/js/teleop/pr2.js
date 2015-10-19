var PR2Base = function (ros) {
    'use strict';
    var base = this;
    base.ros = ros;
    base.ros.getMsgDetails('geometry_msgs/Twist');
    base.commandPub = new ROSLIB.Topic({
        ros: base.ros,
        name: 'base_controller/command',
        messageType: 'geometry_msgs/Twist'
    });
    base.commandPub.advertise();
    base.pubCmd = function (x, y, rot) {
       // console.log("Base Command: ("+x+", "+y+", "+rot+").");
        var cmd = base.ros.composeMsg('geometry_msgs/Twist');
        cmd.linear.x = x;
        cmd.linear.y = y;
        cmd.angular.z = rot;
        base.commandPub.publish(cmd);
    };

    base.drive = function (selector, x, y, rot) {
        if ($(selector).hasClass('ui-state-active')){
            base.pubCmd(x,y,rot);
            setTimeout(function () {
                base.drive(selector, x, y, rot);
                }, 100);
        } else {
            console.log('End driving for '+selector);
        }
    };
};

var PR2GripperSensor = function (options) {
    'use strict';
    var self = this;
    var ros = options.ros;
    self.side = options.side;
    var state = 0.0;

    // Subscribe to state msgs
    var stateSub = new ROSLIB.Topic({
        ros: ros,
        name: self.side.substring(0, 1) + '_gripper_controller/state_throttled',
        messageType: 'pr2_controllers_msgs/JointControllerState'
    });

    var setState = function (msg) {
        state = msg.process_value;
    };
    self.stateCBList = [setState];
    var stateCB = function (msg) {
        for (var i=0; i<self.stateCBList.length; i++) {
            self.stateCBList[i](msg);
        }
    };
    stateSub.subscribe(stateCB);

    // Set Position through gripper_sensor gripper_action
    var positionActionClient = new ROSLIB.ActionClient({
        ros: ros,
        serverName: self.side.substring(0, 1) + '_gripper_sensor_controller/gripper_action',
        actionName: 'pr2_controllers_msgs/Pr2GripperCommandAction'
    });

    ros.getMsgDetails('pr2_controllers_msgs/Pr2GripperCommandGoal');
    self.setPosition = function (pos, effort) {
        var msg = ros.composeMsg('pr2_controllers_msgs/Pr2GripperCommandGoal');
        msg.command.position = pos;
        msg.command.max_effort = effort || -1;
        var goal = new ROSLIB.Goal({
            actionClient: positionActionClient,
            goalMessage: msg
        });
        goal.send();
    };

    self.open = function () {
        self.setPosition(0.09);
    };

    self.close = function () {
        self.setPosition(-0.001);
    };

    // Grab action
    var graspActionClient = new ROSLIB.ActionClient({
        ros: ros,
        serverName: self.side.substring(0, 1) + '_gripper_sensor_controller/grab',
        actionName: 'pr2_gripper_sensor_msgs/PR2GripperGrabAction'
    });

    ros.getMsgDetails('pr2_gripper_sensor_msgs/PR2GripperGrabGoal');
    self.grab = function (hardness_gain) {
        var msg = ros.composeMsg('pr2_gripper_sensor_msgs/PR2GripperGrabGoal');
        msg.command.hardness_gain = hardness_gain || 0.03; // Default value recommended from msg def file
        var goal = new ROSLIB.Goal({
            actionClient: graspActionClient,
            goalMessage: msg
        });
        goal.send();
    };

    // Release action
    var releaseActionClient = new ROSLIB.ActionClient({
        ros: ros,
        serverName: self.side.substring(0, 1) + '_gripper_sensor_controller/release',
        actionName: 'pr2_gripper_sensor_msgs/PR2GripperReleaseAction'
    });

    ros.getMsgDetails('pr2_gripper_sensor_msgs/PR2GripperReleaseGoal');
    self.release = function () {
        var msg = ros.composeMsg('pr2_gripper_sensor_msgs/PR2GripperReleaseGoal');
        msg.command.event.trigger_conditions = 2; // Slip OR finger contact OR accelerometer
        msg.command.event.acceleration_trigger_magnitude = 4.0; // Msg def file recommends 2.0 for small motions, 5.0 for large, rapid motion-planned motions
        msg.command.event.slip_trigger_magnitude = 0.01; // Default value recommended in msg def file
        var goal = new ROSLIB.Goal({
            actionClient: releaseActionClient,
            goalMessage: msg
        });
        goal.send();
    };

};

var PR2Gripper = function (options) {
    'use strict';
    var self = this;
    self.side = options.side;
    var ros = options.ros;
    var state = 0.0;
    ros.getMsgDetails('pr2_controllers_msgs/Pr2GripperCommandActionGoal');
    var stateSub = new ROSLIB.Topic({
        ros: ros,
        name: self.side.substring(0, 1) + '_gripper_controller/state_throttled',
        messageType: 'pr2_controllers_msgs/JointControllerState'
    });

    var setState = function (msg) {
        state = msg.process_value;
    };
    self.stateCBList = [setState];
    var stateCB = function (msg) {
        for (var i=0; i<self.stateCBList.length; i++) {
            self.stateCBList[i](msg);
        }
    };
    stateSub.subscribe(stateCB);
    
    var goalPub = new ROSLIB.Topic({
        ros: ros,
        name: self.side.substring(0, 1) + '_gripper_controller/gripper_action/goal',
        messageType: 'pr2_controllers_msgs/Pr2GripperCommandActionGoal'
    });
    goalPub.advertise();

    self.setPosition = function (pos, effort) {
        var goalMsg = ros.composeMsg('pr2_controllers_msgs/Pr2GripperCommandActionGoal');
        goalMsg.goal.command.position = pos;
        goalMsg.goal.command.max_effort = effort || -1;
        goalPub.publish(goalMsg);
    };

    self.open = function () {
        self.setPosition(0.09);
    };

    self.close = function () {
        self.setPosition(-0.001);
    };
};

var PR2Head = function (options) {
    'use strict';
    var self = this;
    var ros = options.ros;
    var state = [0.0, 0.0];
    var limits = options.limits || [[-2.85, 2.85], [1.18, -0.38]];
    var joints = options.joints || ['head_pan_joint', 'head_tilt_joint'];
    self.pointingFrame = options.pointingFrame || 'head_mount_kinect_rgb_optical_frame';
    var trackingInterval = null;
    var undoSetActiveService = options.undoSetActiveService;
    self.ros.getMsgDetails('trajectory_msgs/JointTrajectory');
    self.ros.getMsgDetails('trajectory_msgs/JointTrajectoryPoint');
    self.jointPub = new ROSLIB.Topic({
        ros: self.ros,
        name: 'head_traj_controller/command',
        messageType: 'trajectory_msgs/JointTrajectory'
    });
    self.jointPub.advertise();

    self.ros.getMsgDetails('pr2_controllers_msgs/PointHeadGoal');
    self.pointHeadActionClient = new ROSLIB.ActionClient({
        ros: self.ros,
        serverName: "/head_traj_controller/point_head_action",
        actionName: "pr2_controllers_msgs/PointHeadAction"
    });

    self.stateSub = new ROSLIB.Topic({
        ros: self.ros,
        name: '/head_traj_controller/state_throttled',
        messageType: 'pr2_controllers_msgs/JointTrajectoryControllerState'
    });
    self.setState = function (msg) {
        self.state = msg.actual.positions;    
    };
    self.stateCBList = [self.setState];
    self.stateCB = function (msg){
        for (var i=0; i<self.stateCBList.length; i++){
            self.stateCBList[i](msg);
        }
    };
    self.stateSub.subscribe(self.stateCB);

    self.enforceLimits = function (pan, tilt) {
        pan  = pan > self.limits[0][0] ? pan : self.limits[0][0];
        pan  = pan < self.limits[0][1] ? pan : self.limits[0][1];
        tilt  = tilt < self.limits[1][0] ? tilt : self.limits[1][0];
        tilt  = tilt > self.limits[1][1] ? tilt : self.limits[1][1];
        return [pan, tilt];
    };

    self.setPosition = function (pan, tilt) {
        var dPan = Math.abs(pan - self.state[0]);
        var dTilt = Math.abs(tilt - self.state[1]);
        var trajPointMsg = self.ros.composeMsg('trajectory_msgs/JointTrajectoryPoint');
        trajPointMsg.positions = self.enforceLimits(pan, tilt);
        trajPointMsg.velocities = [0.0, 0.0];
        trajPointMsg.time_from_start.secs = Math.max(dPan+dTilt, 1);
        var goalMsg = self.ros.composeMsg('trajectory_msgs/JointTrajectory');
        goalMsg.joint_names = self.joints;
        goalMsg.points.push(trajPointMsg);
        self.jointPub.publish(goalMsg);
    };

    self.delPosition = function (delPan, delTilt) {
        var pan = self.state[0] += delPan;
        var tilt = self.state[1] += delTilt;
        self.setPosition(pan, tilt);
    };

    var getPointHeadGoal = function (x, y, z, frame) {
        var headPointMsg = self.ros.composeMsg('pr2_controllers_msgs/PointHeadGoal');
        headPointMsg.pointing_axis = {
            x: 0,
            y: 0,
            z: 1
        };
        headPointMsg.target.header.frame_id = frame;
        headPointMsg.target.point = {
            x: x,
            y: y,
            z: z
        };
        headPointMsg.pointing_frame = self.pointingFrame;
        headPointMsg.max_velocity = 0.25;
        return headPointMsg;
    };

    self.pointHead = function (x, y, z, frame) {
        var headPointMsg = getPointHeadGoal(x, y, z, frame);
        var actionGoal = new ROSLIB.Goal({
            actionClient: self.pointHeadActionClient,
            goalMessage: headPointMsg
        });
        actionGoal.send();
    };


    self.undoToggleService = new ROSLIB.Service({
        ros: ros,
        name: undoSetActiveService,
        serviceType: 'hrl_undo/SetActive'
    });

    self.trackPoint = function (x, y, z, frame) {
        trackingInterval = setInterval(function() {self.pointHead(x, y, z, frame);}, 1500);
    };

    self.stopTracking = function () {
        clearInterval(trackingInterval);
    };
};

var PR2Torso = function (ros) {
    'use strict';
    var self = this;
    self.ros = ros;
    self.state = null;
    self.ros.getMsgDetails('trajectory_msgs/JointTrajectory');
    self.ros.getMsgDetails('trajectory_msgs/JointTrajectoryPoint');
    self.jointNames = ['torso_lift_joint'];

    self.goalPub = new ROSLIB.Topic({
        ros: self.ros,
        name: 'torso_controller/command',
        messageType: 'trajectory_msgs/JointTrajectory'
    });
    self.goalPub.advertise();

    self.stateSub = new ROSLIB.Topic({
        ros: self.ros,
        name: 'torso_controller/state_throttled',
        messageType: 'pr2_controllers_msgs/JointTrajectoryControllerState'
    });
    self.setState = function (msg) {
        self.state = msg.actual.positions[0];
    };
    self.stateCBList = [self.setState];
    self.stateCB = function (msg) {
        for (var i=0; i<self.stateCBList.length; i++){
            self.stateCBList[i](msg);
        }
    };
    self.stateSub.subscribe(self.stateCB);

    self.setPosition = function (z) {
        console.log('Commanding torso' + ' from z=' + self.state.toString() + ' to z=' + z.toString());
        var goal_msg = self.ros.composeMsg('trajectory_msgs/JointTrajectory');
        var traj_point = self.ros.composeMsg('trajectory_msgs/JointTrajectoryPoint');
        traj_point.positions = [z];
        traj_point.time_from_start.secs = 1;
        goal_msg.joint_names = self.jointNames;
        goal_msg.points = [traj_point];
        self.goalPub.publish(goal_msg);
    };
};

//var PR2ArmJoints = function (ros) {
//    'use strict';
//    var self = this;
//    this.state = null;
//    self.getMsgDetails('p
//};

var PR2ArmMPC = function (options) {
    'use strict';
    var self = this;
    self.ros = options.ros;
    self.side = options.side;
    self.ee_frame = options.ee_frame;
    self.stateTopic = options.stateTopic || 'haptic_mpc/gripper_pose';
    self.poseGoalTopic = options.poseGoalTopic || 'haptic_mpc/goal_pose';
    self.trajectoryGoalTopic = options.trajectoryGoalTopic || 'haptic_mpc/joint_trajectory';
    self.plannerServiceName = options.plannerServiceName;
    self.state = null;
    self.ros.getMsgDetails('geometry_msgs/PoseStamped');
    self.ros.getMsgDetails('trajectory_msgs/JointTrajectory');

    self.setState = function (msg) {
        self.state = msg;
    };
    self.stateCBList = [self.setState];
    self.stateCB = function (msg) {
        for (var i=0; i<self.stateCBList.length; i += 1) {
            self.stateCBList[i](msg);
        }
    };
    self.stateSubscriber = new ROSLIB.Topic({
        ros: self.ros,
        name: self.stateTopic,
        messageType: 'geometry_msgs/PoseStamped'
    });
    self.stateSubscriber.subscribe(self.stateCB);

    self.goalPosePublisher = new ROSLIB.Topic({
        ros: self.ros,
        name: self.poseGoalTopic,
        messageType: 'geometry_msgs/PoseStamped'
    });
    self.goalPosePublisher.advertise();

    self.sendPoseGoal = function (options) {
        var position =  options.position || self.state.pose.position;
        var orientation =  options.orientation || self.state.pose.orientation;
        var frame_id =  options.frame_id || self.state.header.frame_id;
        var msg = self.ros.composeMsg('geometry_msgs/PoseStamped');
        msg.header.frame_id = frame_id;
        msg.pose.position = position;
        msg.pose.orientation = orientation;
        console.log("Sending Goal:", msg);
        self.goalPosePublisher.publish(msg);
    };

    self.trajectoryGoalPublisher = new ROSLIB.Topic({
        ros: self.ros,
        name: self.trajectoryGoalTopic,
        messageType: 'trajectory_msgs/JointTrajectory'
    });
    self.trajectoryGoalPublisher.advertise();

    self.sendTrajectoryGoal = function (trajectory) {
        self.trajectoryGoalPublisher.publish(trajectory);
    };

    self.moveitPlannerClient = new ROSLIB.Service({
        ros: self.ros,
        name: self.plannerServiceName,
        serviceType: 'assistive_teleop/MoveItPlan'
    });
    
    self.planTrajectory = function (options) {
        var position =  options.position || self.state.pose.position;
        var orientation =  options.orientation || self.state.pose.orientation;
        var frame_id =  options.frame_id || self.state.header.frame_id;
        var cb = options.cb || function () {}; // Default to no-op on return
        var pose = self.ros.composeMsg('geometry_msgs/PoseStamped');
        pose.header.frame_id = frame_id;
        pose.pose.position = position;
        pose.pose.orientation = orientation;
        console.log("Requesting trajectory to ", pose);
        var failureCB = function (msg) {console.log("Service Faluire!", msg);};
        var req = new ROSLIB.ServiceRequest({'pose_target': pose});
        self.moveitPlannerClient.callService(req, cb, failureCB);
    };



};

var PR2 = function (ros) {
    'use strict';
    var self = this;
    self.torso = new PR2Torso(ros);
    self.r_gripper = new PR2GripperSensor({side: 'right', ros: ros});
    self.l_gripper = new PR2GripperSensor({side: 'left', ros: ros});
    //self.r_gripper = new PR2Gripper({side: 'right', ros: ros});
    //self.l_gripper = new PR2Gripper({side: 'left', ros: ros});
    self.base = new PR2Base(ros);
    self.head = new PR2Head({ros: ros,
                             
    
    });
    self.head.stopTracking(); // Cancel left-over tracking goals from before page refresh...
    self.r_arm_cart = new PR2ArmMPC({side:'right',
                                     ros: ros,
                                     stateTopic: 'right_arm/haptic_mpc/gripper_pose',
                                     poseGoalTopic: 'right_arm/haptic_mpc/goal_pose',
                                     ee_frame:'r_gripper_tool_frame',
                                     trajectoryGoalTopic: '/right_arm/haptic_mpc/joint_trajectory',
                                     plannerServiceName:'/moveit_plan/right_arm'});
    self.l_arm_cart = new PR2ArmMPC({side:'left',
                                     ros: ros,
                                     stateTopic: 'left_arm/haptic_mpc/gripper_pose',
                                     poseGoalTopic: 'left_arm/haptic_mpc/goal_pose',
                                     ee_frame:'l_gripper_tool_frame',
                                     trajectoryGoalTopic: '/left_arm/haptic_mpc/joint_trajectory',
                                     plannerServiceName:'/moveit_plan/left_arm'});
};
