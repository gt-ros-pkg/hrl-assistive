RFH.PickAndPlace = function (options) {
    'use strict';
    var self = this;
    self.ros = options.ros;
    self.name = options.name || 'pick_and_place';
    self.ros.getMsgDetails('hrl_task_planning/PDDLProblem');
    self.taskPublisher = new ROSLIB.Topic({
        ros: self.ros,
        name: '/perform_task',
        messageType: 'hrl_task_planning/PDDLProblem'
    });
    self.taskPublisher.advertise();
    self.side = 'right';

    self.getInterfaceTask = function (plan_step) {
        switch (plan_step.name){
            case 'ID-LOCATION':
                return 'idLocationTask';
            case 'FORGET-LOCATION':
                return 'LookingTask';
            case 'MOVE-ARM':
                return  self.side.substring(0,1)+'EECartTask';
            case 'GRAB':
                return self.side.substring(0,1)+'EECartTask';
            case 'RELEASE':
                return self.side.substring(0,1)+'EECartTask';
        }
    };

    self.publishPickAndPlace = function (side) {
        self.side = side; // Save most recently requested side here
        var msg = self.ros.composeMsg('hrl_task_planning/PDDLProblem');
        msg.name = 'pick_and_place_'+side+'_'+ new Date().getTime().toString();
        msg.domain = 'pick_and_place';
        self.taskPublisher.publish(msg);
    };

    self.start = function(){};
    self.stop = function(){};
};
