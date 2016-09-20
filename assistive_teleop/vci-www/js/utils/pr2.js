var RFH = (function (module) {
    'use strict';
    module.PR2Base = function (ros) {
        var self = this;
        self.timer = null;

        ros.getMsgDetails('geometry_msgs/Twist');
        var commandPub = new ROSLIB.Topic({
            ros: ros,
            name: 'base_controller/command',
            messageType: 'geometry_msgs/Twist'
        });
        commandPub.advertise();

        self.pubCmd = function (x, y, rot) {
            // console.log("Base Command: ("+x+", "+y+", "+rot+").");
            var cmd = ros.composeMsg('geometry_msgs/Twist');
            cmd.linear.x = x;
            cmd.linear.y = y;
            cmd.angular.z = rot;
            commandPub.publish(cmd);
        };
    };

    module.PR2GripperSensor = function (options) {
        var self = this;
        var ros = options.ros;
        self.side = options.side;
        var state = 0.0;
        var grasping = null;
        var grabGoal = null;

        // Subscribe to grasping (bool) state messages
        var graspSub = new ROSLIB.Topic({
            ros: ros,
            name: '/grasping/'+self.side+'_gripper',
            messageType: 'std_msgs/Bool'
        });

        self.getGrasping = function () {
            return grasping;
        };

        self.setGrasping = function (grasp_state) {
            grasping = grasp_state;    
        };

        var graspStateFromMsg = function (msg){
            grasping = msg.data;
        };
        self.graspingCBList = [graspStateFromMsg];
        var graspingCB = function (msg){
            for (var i=0; i<self.graspingCBList.length; i++) {
                self.graspingCBList[i](msg);
            }
        };
        graspSub.subscribe(graspingCB);

        // Subscribe to state msgs
        var stateSub = new ROSLIB.Topic({
            ros: ros,
            name: self.side.substring(0, 1) + '_gripper_controller/state_throttled',
            messageType: 'pr2_controllers_msgs/JointControllerState'
        });

        self.getState = function () {
            return state;
        };

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

        self.open = function () {
            self.setPosition(0.09);
        };

        self.close = function () {
            self.setPosition(-0.001);
        };
        // Open Position Param
        var positionOpenParam = new ROSLIB.Param({
            ros: ros,
            name: self.side[0]+'_gripper_sensor_controller/position_open'
        });

        // Reload params service
        var reloadParamsClient = new ROSLIB.Service({
            ros: ros,
            name: self.side[0]+'_gripper_sensor_controller/reload_params',
            serviceType: 'std_srvs/Empty'
        });

        self.reloadParams = function (cb) {
            cb = cb || function (ret) {};
            var req = new ROSLIB.ServiceRequest({});
            reloadParamsClient.callService(req, cb);
        };

        // Grab action
        var graspActionClient = new ROSLIB.ActionClient({
            ros: ros,
            serverName: self.side.substring(0, 1) + '_gripper_sensor_controller/grab',
            actionName: 'pr2_gripper_sensor_msgs/PR2GripperGrabAction'
        });

        var sendGraspMsg = function () {
            var msg = ros.composeMsg('pr2_gripper_sensor_msgs/PR2GripperGrabGoal');
            msg.command.hardness_gain = 0.03; // Default value recommended from msg def file
            grabGoal = new ROSLIB.Goal({
                actionClient: graspActionClient,
                goalMessage: msg
            });
            grabGoal.send();
        };

        ros.getMsgDetails('pr2_gripper_sensor_msgs/PR2GripperGrabGoal');
        self.grab = function () {
            positionOpenParam.set(state);
            self.reloadParams(sendGraspMsg);
        };

        // Release action
        var releaseActionClient = new ROSLIB.ActionClient({
            ros: ros,
            serverName: self.side.substring(0, 1) + '_gripper_sensor_controller/release',
            actionName: 'pr2_gripper_sensor_msgs/PR2GripperReleaseAction'
        });

        ros.getMsgDetails('pr2_gripper_sensor_msgs/PR2GripperReleaseGoal');
        self.releaseOnContact = function () {
            positionOpenParam.set(0.09);
            self.reloadParams(); // Don't enforce a callback, because the open action shouldn't take palce immediately.  It's a race condition we should always win.
            var msg = ros.composeMsg('pr2_gripper_sensor_msgs/PR2GripperReleaseGoal');
            msg.command.event.trigger_conditions = 2; // Slip OR finger contact OR accelerometer
            msg.command.event.acceleration_trigger_magnitude = 2.0; // Msg def file recommends 2.0 for small motions, 5.0 for large, rapid motion-planned motions
            msg.command.event.slip_trigger_magnitude = 0.008; // Default value recommended in msg def file is 0.01
            var goal = new ROSLIB.Goal({
                actionClient: releaseActionClient,
                goalMessage: msg
            });
            goal.send();
        };

        self.cancelReleaseOnContact = function () {
            releaseActionClient.cancel();
        };
    };

    module.PR2Gripper = function (options) {
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

    module.PR2Head = function (options) {
        var self = this;
        var ros = options.ros;
        var state = [0.0, 0.0];
        var limits = options.limits || [[-2.85, 2.85], [1.18, -0.38]];
        var joints = options.joints || ['head_pan_joint', 'head_tilt_joint'];
        self.pointingFrame = options.pointingFrame || 'head_mount_kinect_rgb_optical_frame';
        var trackingInterval = null;
        ros.getMsgDetails('trajectory_msgs/JointTrajectory');
        ros.getMsgDetails('trajectory_msgs/JointTrajectoryPoint');
        var jointPub = new ROSLIB.Topic({
            ros: ros,
            name: 'head_traj_controller/command',
            messageType: 'trajectory_msgs/JointTrajectory'
        });
        jointPub.advertise();

        self.getState = function () {
            return state;
        };

        ros.getMsgDetails('pr2_controllers_msgs/PointHeadGoal');
        var pointHeadActionClient = new ROSLIB.ActionClient({
            ros: ros,
            serverName: "/head_traj_controller/point_head_action",
            actionName: "pr2_controllers_msgs/PointHeadAction"
        });

        var stateSub = new ROSLIB.Topic({
            ros: ros,
            name: '/head_traj_controller/state_throttled',
            messageType: 'pr2_controllers_msgs/JointTrajectoryControllerState'
        });
        var setState = function (msg) {
            state = msg.actual.positions;    
        };
        self.stateCBList = [setState];
        var stateCB = function (msg){
            for (var i=0; i < self.stateCBList.length; i += 1){
                self.stateCBList[i](msg);
            }
        };
        stateSub.subscribe(stateCB);

        self.enforceLimits = function (pan, tilt) {
            pan  = pan > limits[0][0] ? pan : limits[0][0];
            pan  = pan < limits[0][1] ? pan : limits[0][1];
            tilt  = tilt < limits[1][0] ? tilt : limits[1][0];
            tilt  = tilt > limits[1][1] ? tilt : limits[1][1];
            return [pan, tilt];
        };

        self.setPosition = function (pan, tilt) {
            var state = self.getState();
            var dPan = Math.abs(pan - state[0]);
            var dTilt = Math.abs(tilt - state[1]);
            var trajPointMsg = ros.composeMsg('trajectory_msgs/JointTrajectoryPoint');
            trajPointMsg.positions = self.enforceLimits(pan, tilt);
            trajPointMsg.velocities = [0.0, 0.0];
            trajPointMsg.time_from_start.secs = Math.max(dPan+dTilt, 1);
            var goalMsg = ros.composeMsg('trajectory_msgs/JointTrajectory');
            goalMsg.joint_names = joints;
            goalMsg.points.push(trajPointMsg);
            jointPub.publish(goalMsg);
        };

        self.delPosition = function (delPan, delTilt) {
            var state = self.getState();
            var pan = state[0] + delPan;
            var tilt = state[1] + delTilt;
            self.setPosition(pan, tilt);
        };

        var getPointHeadGoal = function (x, y, z, frame) {
            var headPointMsg = ros.composeMsg('pr2_controllers_msgs/PointHeadGoal');
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
            headPointMsg.max_velocity = 0.45;
            return headPointMsg;
        };

        self.pointHead = function (x, y, z, frame) {
            var headPointMsg = getPointHeadGoal(x, y, z, frame);
            var actionGoal = new ROSLIB.Goal({
                actionClient: pointHeadActionClient,
                goalMessage: headPointMsg
            });
            actionGoal.send();
        };

        self.trackPoint = function (x, y, z, frame) {
            self.stopTracking();
            self.pointHead(x, y, z, frame); // Start looking now
            trackingInterval = setInterval(function() {self.pointHead(x, y, z, frame);}, 1500); // Re-send goal regularly
        };

        self.trackAngles = function (pan, tilt) {
            self.stopTracking();
            self.setPosition(pan, tilt);
            trackingInterval = setInterval(function() {self.setPosition(pan, tilt);}, 1500); // Re-send goal regularly
        };

        self.stopTracking = function () {
            // Stop sending tracking messages
            clearInterval(trackingInterval);
        };
    };

    module.PR2Torso = function (ros) {
        var self = this;
        var state = 0.0;
        ros.getMsgDetails('trajectory_msgs/JointTrajectory');
        ros.getMsgDetails('trajectory_msgs/JointTrajectoryPoint');
        var jointNames = ['torso_lift_joint'];

        var goalPub = new ROSLIB.Topic({
            ros: ros,
            name: 'torso_controller/command',
            messageType: 'trajectory_msgs/JointTrajectory'
        });
        goalPub.advertise();

        self.getState = function () {
            return state;
        };

        var stateSub = new ROSLIB.Topic({
            ros: ros,
            name: 'torso_controller/state_throttled',
            messageType: 'pr2_controllers_msgs/JointTrajectoryControllerState'
        });

        self.setState = function (msg) {
            state = msg.actual.positions[0];
        };

        self.stateCBList = [self.setState];
        var stateCB = function (msg) {
            for (var i=0; i < self.stateCBList.length; i += 1){
                self.stateCBList[i](msg);
            }
        };
        stateSub.subscribe(stateCB);

        self.setPosition = function (z) {
            //console.log('Commanding torso' + ' from z=' + state.toString() + ' to z=' + z.toString());
            var goal_msg = ros.composeMsg('trajectory_msgs/JointTrajectory');
            var traj_point = ros.composeMsg('trajectory_msgs/JointTrajectoryPoint');
            traj_point.positions = [z];
            traj_point.time_from_start.secs = 1;
            goal_msg.joint_names = jointNames;
            goal_msg.points = [traj_point];
            goalPub.publish(goal_msg);
        };
    };

    module.PR2ArmMPC = function (options) {
        var self = this;
        self.ros = options.ros;
        self.side = options.side;
        self.ee_frame = options.ee_frame;
        self.stateTopic = options.stateTopic || 'haptic_mpc/gripper_pose';
        self.jointStateTopic = options.jointStateTopic || self.side[0]+'_arm_controller/state';
        self.poseGoalTopic = options.poseGoalTopic || 'haptic_mpc/goal_pose';
        self.trajectoryGoalTopic = options.trajectoryGoalTopic || 'haptic_mpc/joint_trajectory';
        self.enableMPCServiceName = options.enableMPCServiceName || 'haptic_mpc/enable_mpc';
        self.plannerServiceName = options.plannerServiceName;
        self.state = null;
        self.jointNames = [];
        self.ros.getMsgDetails('geometry_msgs/PoseStamped');
        self.ros.getMsgDetails('trajectory_msgs/JointTrajectory');

        self.getState = function () {
            return self.state;
        };

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
            self.goalPosePublisher.publish(msg);
        };

        self.getJointNames = function (state_msg) {
            self.jointNames = state_msg.joint_names;
            self.jointStateSubscriber.unsubscribe();
        };

        self.jointStateSubscriber = new ROSLIB.Topic({
            ros: self.ros,
            name: self.jointStateTopic,
            messageType: 'pr2_controllers_msgs/JointTrajectoryControllerState'
        });
        self.jointStateSubscriber.subscribe(self.getJointNames);

        self.trajectoryGoalPublisher = new ROSLIB.Topic({
            ros: self.ros,
            name: self.trajectoryGoalTopic,
            messageType: 'trajectory_msgs/JointTrajectory'
        });
        self.trajectoryGoalPublisher.advertise();

        self.sendTrajectoryGoal = function (trajectory) {
            self.trajectoryGoalPublisher.publish(trajectory);
        };

        self.sendJointAngleGoal = function (angleList) {
            var traj = self.ros.composeMsg('trajectory_msgs/JointTrajectory');
            var trajPoint = self.ros.composeMsg('trajectory_msgs/JointTrajectoryPoint');
            trajPoint.positions = angleList;
            traj.joint_names = self.jointNames;
            traj.points.push(trajPoint);
            self.trajectoryGoalPublisher.publish(traj);
        };

        self.enableMPCService = new ROSLIB.Service({
            ros: self.ros,
            name: self.enableMPCServiceName,
            serviceType: 'hrl_haptic_manipulation_in_clutter_srvs/EnableHapticMPC'
        });

        self.enableMPC = function (cb) {
            cb = cb === undefined ? function (){} : cb;
            var traj = self.ros.composeMsg('trajectory_msgs/JointTrajectory');
            self.trajectoryGoalPublisher.publish(traj);
            var pose = self.ros.composeMsg('geometry_msgs/PoseStamped');
            self.goalPosePublisher.publish(pose);
            var enabled_cb = function(resp) {
                console.log(resp);
                cb(resp);
            };
            var req = new ROSLIB.ServiceRequest({'new_state': 'enabled'});
            self.enableMPCService.callService(req, enabled_cb);
        };

        self.disableMPC = function (cb) {
            cb = cb === undefined ? function (){} : cb;
            var disabled_cb = function(resp) {
                console.log(resp);
                cb(resp);
            };
            var req = new ROSLIB.ServiceRequest({'new_state': 'disabled'});
            self.enableMPCService.callService(req, disabled_cb);

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

    module.PR2 = function (ros) {
        var self = this;
        self.torso = new module.PR2Torso(ros);
        self.r_gripper = new module.PR2GripperSensor({side: 'right', ros: ros});
        self.l_gripper = new module.PR2GripperSensor({side: 'left', ros: ros});
        self.base = new module.PR2Base(ros);
        self.head = new module.PR2Head({ros: ros,
            limits: [[-2.85, 2.85], [1.18, -0.38]],
            joints: ['head_pan_joint', 'head_tilt_joint'],
            pointingFrame: 'head_mount_kinect_rgb_optical_frame'});
        self.head.stopTracking(); // Cancel left-over tracking goals from before page refresh...
        self.r_arm_cart = new module.PR2ArmMPC({side:'right',
            ros: ros,
            stateTopic: 'right_arm/haptic_mpc/gripper_pose',
            poseGoalTopic: 'right_arm/haptic_mpc/goal_pose',
            ee_frame:'r_gripper_tool_frame',
            jointStateTopic:'r_arm_controller/state',
            trajectoryGoalTopic: '/right_arm/haptic_mpc/joint_trajectory',
            enableMPCServiceName: '/right_arm/haptic_mpc/enable_mpc',
            plannerServiceName:'/moveit_plan/right_arm'});
        self.l_arm_cart = new module.PR2ArmMPC({side:'left',
            ros: ros,
            stateTopic: 'left_arm/haptic_mpc/gripper_pose',
            poseGoalTopic: 'left_arm/haptic_mpc/goal_pose',
            ee_frame:'l_gripper_tool_frame',
            jointStateTopic:'l_arm_controller/state',
            trajectoryGoalTopic: '/left_arm/haptic_mpc/joint_trajectory',
            enableMPCServiceName: '/left_arm/haptic_mpc/enable_mpc',
            plannerServiceName:'/moveit_plan/left_arm'});
    };
    return module;
})(RFH || {});
