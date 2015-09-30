RFH.MoveObject = function (options) {
    'use strict';
    var self = this;
    self.ros = options.ros;
    self.name = options.name || 'moveObjectTask';
    self.buttonText = 'Move_Object';
    self.buttonClass = 'move-object-button';
    self.ros.getMsgDetails('geometry_msgs/PoseStamped');
    self.taskPublisher = new ROSLIB.Topic({
        ros: self.ros,
        name: '/perform_task',
        messageType: '/hrl_task_planning/PDDLProblem'
    });
    self.taskPublisher.advertise();

    self.publishMoveObject = function () {
        msg = self.ros.composeMsg('/hrl_task_planning/PDDLProblem');
        msg.name = 'move-object-'+ new Date().getTime().toString();
        msg.domain = 'move-object';
        self.taskPublisher.publish(msg);
    };

    self.start = function () {
        self.publishMoveObject();
    };
    
    self.stop = function () {
    };

};
