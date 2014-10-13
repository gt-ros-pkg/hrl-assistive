RFH.Grippers = function (options) {
    'use strict';
    var self = this;
    self.div = options.div || 'markers';
    self.r_gripper = options.r_gripper; 
    self.l_gripper = options.l_gripper; 
    self.buttonText = "Grippers";
    self.buttonClass = 'gripper-button';
    self.rDisplayDiv = 'rGripperDisplay';
    self.lDisplayDiv = 'lGripperDisplay';
    self.rGripperDisplay = new RFH.GripperDisplay({gripper: self.r_gripper,
                                                   parentId: self.div,
                                                   divId: self.rDisplayDiv});
    $('#rGripperDisplay').css({"position":"absolute",
                               "height":"3%",
                               "width":"40%",
                               "bottom":"3%",
                               "right":"3%"}).hide();

    self.rGripperDisplay = new RFH.GripperDisplay({gripper: self.l_gripper,
                                                   parentId: self.div,
                                                   divId: self.lDisplayDiv});
    $('#lGripperDisplay').css({"position":"absolute",
                               "bottom":"3%",
                               "height":"3%",
                               "width":"40%",
                               "left":"3%"}).hide();

    self.start = function () {
        $('#'+self.rDisplayDiv).show();
        $('#'+self.lDisplayDiv).show();
    }
    
    self.stop = function () {
        $('#'+self.rDisplayDiv).hide();
        $('#'+self.lDisplayDiv).hide();
    };

}


RFH.GripperDisplay = function (options) {
    "use strict";
    var self = this;
    self.gripper = options.gripper;
    self.divId = options.divId;
    self.parentId = options.parentId;
    self.gripperSlider = jQuery('<div/>', {id: self.divId}).appendTo('#'+self.parentId);
    self.gripperSlider.slider({
        range: true,
        min: 0.0,
        max: 0.085,
        step: 0.001,
        orientation: 'horizontal'});

    self.mid = 0.5 * ($('#'+self.divId).slider("option", "max") - 
                      $('#'+self.divId).slider("option", "min")) + 
                      $('#'+self.divId).slider("option", "min");

    self.stopCB = function (event, ui) {
        var values = $('#'+self.divId).slider("option", "values");
        self.gripper.setPosition(values[1] - values[0]);
    };
    self.gripperSlider.off("slidestop").on("slidestop.rfh", self.stopCB);

    self.startCB = function (event, ui) {
        $('#'+event.target.id+' > a').addClass('ui-state-active');
    };
    self.gripperSlider.off("slidestart").on("slidestart.rfh", self.startCB);

    self.slideCB = function (event, ui) {
        if (ui.values.indexOf(ui.value) === 0) {//left/low side
            var high = self.mid + (self.mid - ui.value);
            $('#'+self.divId).slider("option", "values", [ui.value, high]);
        } else { //right/high side
            var low = self.mid - (ui.value - self.mid);
            $('#'+self.divId).slider("option", "values", [low, ui.value]);
        }
    };
    self.gripperSlider.off("slide").on("slide.rfh", self.slideCB);

    self.gripperStateDisplay = function (msg) {
        if (!$('#'+self.divId+' > a').hasClass('ui-state-active')) {
            var high = self.mid + 0.5 * msg.process_value;
            var low = self.mid - 0.5 * msg.process_value;
            $('#'+self.divId).slider('option', 'values', [low, high]);
        }
    };
    self.gripper.stateCBList.push(self.gripperStateDisplay);

}
