RFH.CartesianEEControl = function (options) {
    'use strict';
    var self = this;
    self.div = options.div || 'markers';
    self.torso = options.arm
    self.buttonText = self.arm.side[0] === 'r' ? 'Right_Hand' : 'Left_Hand';
    self.buttonClass = 'hand-button';

    self.start = function () {
        //Display interactive markers
        //Display switching controls
    }
    
    self.stop = function () {
        //Hide interactive markers
        //Hide switching controls
    };
}
