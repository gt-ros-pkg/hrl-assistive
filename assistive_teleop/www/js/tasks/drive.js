RFH.Drive = function (options) {
    "use strict";
    var self = this;
    var options = options || {};
    self.ros = options.ros;
    self.div = options.div || 'markers';
    self.head = options.head;
    self.torso = options.torso;
    self.camera = options.camera;
    self.buttonText = 'Drive';
    self.buttonClass = 'drive-button';

    self.start = function () {
        alert ("Starting Driving task");
    }

    self.stop = function () {
        alert ("Stopping Driving task");
    }

    self.onClick = function (e) {
        
    }

}
