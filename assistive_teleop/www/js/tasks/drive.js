RFH.Drive = function (options) {
    "use strict";
    var options = options || {};
    var self = this;
    self.ros = options.ros;
    self.div = options.div || 'markers';
    self.head = options.head || new Pr2Head(self.ros);
    self.buttonText = 'Drive';
    self.buttonClass = 'drive-button';

    self.start = function () {
        alert ("Starting Driving task");
    }

    self.stop = function () {
        alert ("Stopping Driving task");

    }

}
