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
        console.log("Base Command: ("+x+", "+y+", "+rot+").");
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
                base.drive(selector, x, y, rot)
                }, 100);
        } else {
            console.log('End driving for '+selector);
        };
    };
};

var PR2Gripper = function (side, ros) {
    'use strict';
    var gripper = this;
    gripper.side = side;
    gripper.ros = ros;
    gripper.state = 0.0;
    gripper.ros.getMsgDetails('pr2_controllers_msgs/Pr2GripperCommandActionGoal');
    gripper.stateSub = new ROSLIB.Topic({
        ros: gripper.ros,
        name: gripper.side.substring(0, 1) + '_gripper_controller/state_throttled',
        messageType: 'pr2_controllers_msgs/JointControllerState'
    });

    gripper.setState = function (msg) {
        gripper.state = msg.process_value;
    };
    gripper.stateCBList = [gripper.setState];
    gripper.stateCB = function (msg) {
        for (var i=0; i<gripper.stateCBList.length; i++) {
            gripper.stateCBList[i](msg);
        };
    };
    gripper.stateSub.subscribe(gripper.stateCB);
    
    gripper.goalPub = new ROSLIB.Topic({
        ros: gripper.ros,
        name: gripper.side.substring(0, 1) + '_gripper_controller/gripper_action/goal',
        messageType: 'pr2_controllers_msgs/Pr2GripperCommandActionGoal'
    });
    gripper.goalPub.advertise();

    gripper.setPosition = function (pos, effort) {
        var eff = effort || -1;
        var goalMsg = gripper.ros.composeMsg('pr2_controllers_msgs/Pr2GripperCommandActionGoal');
        goalMsg.goal.command.position = pos;
        goalMsg.goal.command.max_effort = eff;
        gripper.goalPub.publish(goalMsg);
    };

    gripper.open = function () {
        gripper.setPosition(0.09);
    };

    gripper.close = function () {
        gripper.setPosition(-0.001);
    };
};

var PR2Head = function (ros) {
    'use strict';
    var head = this;
    head.ros = ros;
    head.state = [0.0, 0.0];
    head.limits = [[-2.85, 2.85], [1.18, -0.38]];
    head.joints = ['head_pan_joint', 'head_tilt_joint'];
    head.pointingFrame = 'head_mount_kinect_rgb_optical_frame';
    head.ros.getMsgDetails('pr2_controllers_msgs/JointTrajectoryActionGoal');
    head.jointPub = new ROSLIB.Topic({
        ros: head.ros,
        name: 'head_traj_controller/joint_trajectory_action/goal',
        messageType: 'pr2_controllers_msgs/JointTrajectoryActionGoal'
    });
    head.jointPub.advertise();

    head.ros.getMsgDetails('pr2_controllers_msgs/PointHeadActionGoal');
    head.pointPub = new ROSLIB.Topic({
        ros: head.ros,
        name: 'head_traj_controller/point_head_action/goal',
        messageType: 'pr2_controllers_msgs/PointHeadActionGoal'
    });
    head.pointPub.advertise();

    head.stateSub = new ROSLIB.Topic({
        ros: head.ros,
        name: '/head_traj_controller/state_throttled',
        messageType: 'pr2_controllers_msgs/JointTrajectoryControllerState'
    });
    head.setState = function (msg) {
        head.state = msg.actual.positions;    
    };
    head.stateCBList = [head.setState];
    head.stateCB = function (msg){
        for (var i=0; i<head.stateCBList.length; i++){
            head.stateCBList[i](msg);
        };
    };
    head.stateSub.subscribe(head.stateCB);

    head.enforceLimits = function (pan, tilt) {
        pan  = pan > head.limits[0][0] ? pan : head.limits[0][0];
        pan  = pan < head.limits[0][1] ? pan : head.limits[0][1];
        tilt  = tilt < head.limits[1][0] ? tilt : head.limits[1][0];
        tilt  = tilt > head.limits[1][1] ? tilt : head.limits[1][1];
        return [pan, tilt];
    };

    head.setPosition = function (pan, tilt) {
        var dPan = Math.abs(pan - head.state[0]);
        var dTilt = Math.abs(tilt - head.state[1]);
        var trajPointMsg = head.ros.composeMsg('trajectory_msgs/JointTrajectoryPoint');
        trajPointMsg.positions = head.enforceLimits(pan, tilt);
        trajPointMsg.velocities = [0.0, 0.0];
        trajPointMsg.time_from_start.secs = Math.max(dPan+dTilt, 1);
        var goalMsg = head.ros.composeMsg('pr2_controllers_msgs/JointTrajectoryActionGoal');
        goalMsg.goal.trajectory.joint_names = head.joints;
        goalMsg.goal.trajectory.points.push(trajPointMsg);
        head.jointPub.publish(goalMsg);
    };
    head.delPosition = function (delPan, delTilt) {
        var pan = head.state[0] += delPan;
        var tilt = head.state[1] += delTilt;
        head.setPosition(pan, tilt);
    };
    head.pointHead = function (x, y, z, frame) {
        var headPointMsg = head.ros.composeMsg('pr2_controllers_msgs/PointHeadActionGoal');
        headPointMsg.goal.pointing_axis = {
            x: 0,
            y: 0,
            z: 1
        };
        headPointMsg.goal.target.header.frame_id = frame;
        headPointMsg.goal.target.point = {
            x: x,
            y: y,
            z: z
        };
        headPointMsg.goal.pointing_frame = head.pointingFrame;
        headPointMsg.goal.max_velocity = 0.35;
        head.pointPub.publish(headPointMsg);
    };
};

var PR2Torso = function (ros) {
    'use strict';
    var torso = this;
    torso.ros = ros;
    torso.state = 0.0;
    torso.ros.getMsgDetails('pr2_controllers_msgs/SingleJointPositionActionGoal');

    torso.goalPub = new ROSLIB.Topic({
        ros: torso.ros,
        name: 'torso_controller/position_joint_action/goal',
        messageType: 'pr2_controllers_msgs/SingleJointPositionActionGoal'
    });
    torso.goalPub.advertise();

    torso.stateSub = new ROSLIB.Topic({
        ros: torso.ros,
        name: 'torso_controller/state_throttled',
        messageType: 'pr2_controllers_msgs/JointTrajectoryControllerState'
    });
    torso.setState = function (msg) {
        torso.state = msg.actual.positions[0];
    };
    torso.stateCBList = [torso.setState];
    torso.stateCB = function (msg) {
        for (var i=0; i<torso.stateCBList.length; i++){
            torso.stateCBList[i](msg);
        };
    };
    torso.stateSub.subscribe(torso.stateCB);

    torso.setPosition = function (z) {
        var dir = (z < torso.state) ? 'Lowering' : 'Raising';
        log(dir + " Torso");
        console.log('Commanding torso' + ' from z=' + torso.state.toString() + ' to z=' + z.toString());
        var goal_msg = torso.ros.composeMsg('pr2_controllers_msgs/SingleJointPositionActionGoal');
        goal_msg.goal.position = z;
        goal_msg.goal.max_velocity = 1.0;
        torso.goalPub.publish(goal_msg);
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
    self.goalTopic = options.goalTopic || 'haptic_mpc/goal_pose';
    self.state = null;
    self.ros.getMsgDetails('geometry_msgs/PoseStamped');

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
        name: self.goalTopic,
        messageType: 'geometry_msgs/PoseStamped'
    });
    self.goalPosePublisher.advertise();

    self.sendGoal = function (options) {
        var position =  options.position || self.state.pose.position;
        var orientation =  options.orientation || self.state.pose.orientation;
        var frame_id =  options.frame_id || self.state.header.frame_id;
        var msg = self.ros.composeMsg('geometry_msgs/PoseStamped');
        msg.header.frame_id = frame_id;
        msg.pose.position = position;
        msg.pose.orientation = orientation;
        self.goalPosePublisher.publish(msg);
    }
}

var PR2ArmJTTask = function (options) {
    'use strict';
    var self = this;
    self.ros = options.ros;
    self.side = options.side;
    self.ee_frame = options.ee_frame;
    self.stateTopic = options.stateTopic || options.side[0]+'_cart/state/x';
    self.goalTopic = options.goalTopic || options.side[0]+'_cart/command_pose';
    self.state = null;
    self.ros.getMsgDetails('geometry_msgs/PoseStamped');

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
        name: self.goalTopic,
        messageType: 'geometry_msgs/PoseStamped'
    });
    self.goalPosePublisher.advertise();

    self.sendGoal = function (options) {
        var position =  options.position || self.state.pose.position;
        var orientation =  options.orientation || self.state.pose.orientation;
        var frame_id =  options.frame_id || self.state.header.frame_id;
        var msg = self.ros.composeMsg('geometry_msgs/PoseStamped');
        msg.header.frame_id = frame_id;
        msg.pose.position = position;
        msg.pose.orientation = orientation;
        self.goalPosePublisher.publish(msg);
    }
}
var PR2 = function (ros) {
    'use strict';
    var self = this;
    self.ros = ros;
    self.torso = new PR2Torso(self.ros);
    self.r_gripper = new PR2Gripper('right', self.ros);
    self.l_gripper = new PR2Gripper('left', self.ros);
    self.base = new PR2Base(self.ros);
    self.head = new PR2Head(self.ros);
    self.r_arm_cart = new PR2ArmJTTask({side:'right',
                                        ros: self.ros,
                                        ee_frame:'r_gripper_tool_frame'});
    self.l_arm_cart = new PR2ArmMPC({side:'left',
                                     ros: self.ros,
                                     ee_frame:'l_gripper_tool_frame'});
}
