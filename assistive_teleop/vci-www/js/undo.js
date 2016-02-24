RFH.Undo = function (options) {
    'use strict';
    var self = this;
    options = options || {};
    var ros = options.ros;
    self.undoTopic = options.undoTopic || '/undo';
    var buttonDiv = options.buttonDiv;
    $('#'+buttonDiv).button();
    var eventQueue = [];

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

    var undoButtonCB = function ( ) {

    };

    $('#'+buttonDiv).on('click.rfh', undoButtonCB); 

    // Handle task switching
    

    // Handle Spine goals
    var torso = new PR2Torso(ros);
    ros.getMsgDetails('trajectory_msgs/JointTrajectory');
    ros.getMsgDetails('trajectory_msgs/JointTrajectoryPoint');

    var torsoStateToGoal = function (stateMsg) {
        var trajPoint = ros.composeMsg('trajectory_msgs/JointTrajectoryPoint');
        trajPoint.positions = state_msg.actual.positions;
        trajPoint.velocities = [0];
        trajPoint.accelerations = [0];
        trajPoint.time_from_start.secs = 1;

        var goalMsg = ros.composeMsg('trajectory_msgs/JointTrajectory');
        goalMsg.joint_names = state_msg.joint_names;
        goalMsg.points.push(trajPoint);
        return goalMsg;
    };

    var torsoCmdCB = function (cmdMsg) {
        var undoEntry = {};
        undoEntry.stateGoal = torsoStateToGoal(torso.getState());
        undoEntry.command = cmdMsg;
        undoEntry.time = new Date();

    };

    var torsoSub = new ROSLIB.Topic({
        ros: ros,
        name: '/torso_controller/command',
        messageType:'trajectory_msgs/JointTrajectory'
    });
    torsoSub.subscribe(torsoCmdCB);
}


