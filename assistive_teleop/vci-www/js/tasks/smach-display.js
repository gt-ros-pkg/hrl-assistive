RFH.SmachDisplay = function (options) {
    "use strict";
    var self = this;
    self.ros = options.ros;
    self.container = options.container;
    self.smach_tasks = {};

    self.ros.getMsgDetails('smach_msgs/SmachContainerStructure');
    self.smachSubscriber = new ROSLIB.Topic({
        ros: self.ros,
        name: '/smach_introspection/smach/container_structure',
        type: 'smach_msgs/SmachContainerStructure'
    });

    self.smachContainerCB = function (msg) {
        if (msg.path.indexOf('/') < 0) {
            self.smach_tasks[msg.path] = {'states': msg.children};
        } else {
            //Ignore non-root state data
        }
    };
    self.smachSubscriber.subscribe(self.smachContainerCB);

    self.displaySmachStates = function (task) {
        for (var i; i<len(task.states); i += 1) {
            
            

        }
    };
};
