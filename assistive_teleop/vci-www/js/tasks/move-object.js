RFH.MoveObject = function (options) {
    'use strict';
    var self = this;
    self.ros = options.ros;
    self.name = options.name || 'move-object';
    self.buttonText = 'Move_Object';
    self.buttonClass = 'move-object-button';
    self.ros.getMsgDetails('hrl_task_planning/PDDLProblem');
    self.taskPublisher = new ROSLIB.Topic({
        ros: self.ros,
        name: '/perform_task',
        messageType: 'hrl_task_planning/PDDLProblem'
    });
    self.taskPublisher.advertise();

    self.subtask_mapping = {'id-location': 'idLocationTask',
                            'pick-left-gripper':'lEECartTask',
                            'pick-right-gripper':'rEECartTask',
                            'place-left-gripper':'lEECartTask',
                            'place-right-gripper':'rEECartTask'};

    self.get_subtask = function (plan_step) {
        if (plan_step.name == 'ID-LOCATION'){
            return 'idLocationTask';
        }
        if (plan_step.name == 'pick'){
            if (plan_step.args)
            
        }

    };

    self.publishMoveObject = function () {
        var msg = self.ros.composeMsg('hrl_task_planning/PDDLProblem');
        msg.name = 'move_object+'+ new Date().getTime().toString();
        msg.domain = 'move_object';
        self.taskPublisher.publish(msg);
    };

    self.start = function () {
        self.publishMoveObject();
    };
    
    self.stop = function () {
    };

};
