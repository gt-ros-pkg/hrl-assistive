RFH.Undo = function (options) {
    'use strict';
    var self = this;
    options = options || {};
    self.ros = options.ros;
    self.undoTopic = options.undoTopic || '/undo';
    self.buttonDiv = options.buttonDiv;
    $('#'+self.buttonDiv).button();

    self.ros.getMsgDetails('std_msgs/Int32');
    self.undoPub =  new ROSLIB.Topic({
        ros: self.ros,
        name: self.undoTopic,
        messageType: 'std_msgs/Int32'
    });
    self.undoPub.advertise();

    self.sendUndoCommand = function (numSteps) {
        numSteps = numSteps === undefined ? 1 : numSteps
        console.log("Sending command to undo " + numSteps.toString() + " step(s).");
        self.undoPub.publish({'data': numSteps});
    }

    $('#'+self.buttonDiv).on('click.rfh', function() { self.sendUndoCommand(1) });
}
