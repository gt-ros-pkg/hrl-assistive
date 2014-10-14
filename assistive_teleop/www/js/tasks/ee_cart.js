RFH.CartesianEEControl = function (options) {
    'use strict';
    var self = this;
    self.div = options.div || 'markers';
    self.arm = options.arm;
    self.tfClient = options.tfClient;
    self.buttonText = self.arm.side[0] === 'r' ? 'Right_Hand' : 'Left_Hand';
    self.buttonClass = 'hand-button';
    self.targetId = self.arm.side[0]+"PosControl";
    self.targetControl = jQuery('<div/>', {id: self.targetId,
                                           class: ".position-control-outer"}).appendTo('#'+self.div);
    jQuery('<div/>', {id: self.arm.side[0]+'PosControlInner',
                      class: ".position-control-inner"}).appendTo('#'+self.targetId);
    self.targetControl.hide();

    self.tfTorsoCB = function (transform) {
        self.torsoFrame = transform;
    }
    self.tfClient.subscribe('/torso_lift_link', self.tfTorsoCB);

    self.updateHead = function (transform) { self.headTF = transform; }
    self.tryTFSubscribe = function () {
        if (self.camera.frame_id !== '') {
            self.tfClient.subscribe(self.camera.frame_id, self.updateHead);
            console.log("Got camera data, subscribing to TF Frame: "+self.camera.frame_id);
        } else {
            console.log("No camera data -> no TF Transform");
            setTimeout(self.tryTFSubscribe, 500);
            }
    }
    self.tryTFSubscribe();

    self.start = function () {
        $('#'+self.targetId).show();
    }
    
    self.stop = function () {
        $('#'+self.targetId).hide();
    };


    self.updateTarget = function (msg) {
        self. 
        

    };
    self.arm.stateCBList.push(self.updateTarget);
}
