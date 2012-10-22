var Pr2Base = function (ros) {
    'use strict';
    var base = this;
    base.ros = ros;
    getMsgDetails('geometry_msgs/Twist');
    base.commandPub = new base.ros.Topic({
        name: 'base_controller/command',
        messageType: 'geometry_msgs/Twist'
    });
    base.commandPub.advertise();
    base.pubCmd = function (x, y, rot) {
        var cmd = composeMsg('geometry_msgs/Twist');
        cmd.linear.x = x;
        cmd.linear.y = y;
        cmd.angular.z = rot;
        var cmdMsg = new base.ros.Message(cmd);
        base.commandPub.publish(cmdMsg);
    };

    base.drive = function (selector, x, y, rot) {
        if ($(selector).hasClass('ui-state-active')){
            base.pubCmd(x,y,rot);
            setTimeout(function () {
                base.base.drive(selector, x, y, rot)
                }, 100);
        } else {
            console.log('End driving for '+selector);
        };
    };
};

var Pr2Gripper = function (side, ros) {
    'use strict';
    var gripper = this;
    gripper.side = side;
    gripper.ros = ros;
    gripper.state = 0.0;
    getMsgDetails('pr2_controllers_msgs/Pr2GripperCommandActionGoal');
    gripper.stateSub = new gripper.ros.Topic({
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
    
    gripper.goalPub = new gripper.ros.Topic({
        name: gripper.side.substring(0, 1) + '_gripper_controller/gripper_action/goal',
        messageType: 'pr2_controllers_msgs/Pr2GripperCommandActionGoal'
    });
    gripper.setPosition = function (pos) {
        var goalMsg = composeMsg('pr2_controllers_msgs/Pr2GripperCommandActionGoal');
        goalMsg.goal.command.position = pos;
        goalMsg.goal.command.max_effort = -1;
        var msg = new gripper.ros.Message(goalMsg);
        gripper.goalPub.publish(msg);
    };
    gripper.open = function () {
        gripper.setPosition(0.09);
    };
    gripper.close = function () {
        gripper.setPosition(-0.001);
    };

};

var Pr2Head = function (ros) {
    'use strict';
    var head = this;
    head.ros = ros;
    head.state = [0.0, 0.0];
    head.joints = ['head_pan_joint', 'head_tilt_joint'];
    head.pointingFrame = 'head_mount_kinect_rgb_optical_frame';
    getMsgDetails('pr2_controllers_msgs/JointTrajectoryActionGoal');
    head.jointPub = new head.ros.Topic({
        name: 'head_traj_controller/joint_trajectory_action/goal',
        messageType: 'pr2_controllers_msgs/JointTrajectoryActionGoal'
    });
    head.jointPub.advertise();

    getMsgDetails('pr2_controllers_msgs/PointHeadActionGoal');
    head.pointPub = new head.ros.Topic({
        name: 'head_traj_controller/point_head_action/goal',
        messageType: 'pr2_controllers_msgs/PointHeadActionGoal'
    });
    head.pointPub.advertise();

    head.stateSub = new head.ros.Topic({
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

    head.setPosition = function (pan, tilt) {
        var dPan = Math.abs(pan - head.state[0]);
        var dTilt = Math.abs(tilt - head.state[1]);
        var dist = Math.sqrt(dPan * dPan + dTilt * dTilt);
        var trajPointMsg = composeMsg('trajectory_msgs/JointTrajectoryPoint');
        trajPointMsg.positions = [pan, tilt];
        trajPointMsg.velocities = [0.0, 0.0];
        trajPointMsg.time_from_start.secs = Math.max(dist, 1);
        var goalMsg = composeMsg('pr2_controllers_msgs/JointTrajectoryActionGoal');
        goalMsg.goal.trajectory.joint_names = head.joints;
        goalMsg.goal.trajectory.points.push(trajPointMsg);
        var msg = new head.ros.Message(goalMsg);
        head.jointPub.publish(msg);
    };
    head.delPosition = function (delPan, delTilt) {
        var pan = head.state[0] += delPan;
        var tilt = head.state[1] += delTilt;
        head.setPosition(pan, tilt);
    };
    head.pointHead = function (x, y, z, frame) {
        var headPointMsg = composeMsg('pr2_controllers_msgs/PointHeadActionGoal');
        headPointMsg.goal.target = composeMsg('geometry_msgs/PointStamped');
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
        var msg = new head.ros.Message(headPointMsg);
        head.pointPub.publish(msg);
    };
};

var Pr2Torso = function (ros) {
    'use strict';
    var torso = this;
    torso.ros = ros;
    torso.state = 0.0;
    getMsgDetails('pr2_controllers_msgs/SingleJointPositionActionGoal');

    torso.goalPub = new torso.ros.Topic({
        name: 'torso_controller/position_joint_action/goal',
        messageType: 'pr2_controllers_msgs/SingleJointPositionActionGoal'
    });
    torso.goalPub.advertise();

    torso.stateSub = new torso.ros.Topic({
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
        var goal_msg = composeMsg('pr2_controllers_msgs/SingleJointPositionActionGoal');
        goal_msg.goal.position = z;
        goal_msg.goal.max_velocity = 1.0;
        var msg = new torso.ros.Message(goal_msg);
        torso.goalPub.publish(msg);
    };
};
