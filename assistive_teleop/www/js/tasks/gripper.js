RFH.Gripper = function (options) {
    'use strict';
    var self = this;
    self.div = options.div || 'markers';
    self.gripper = options.gripper; 
    self.side = options.side; 
    self.buttonText = self.side.indexOf("r") >= 0 ? "Right Gripper" : "Left Gripper";
    self.gripperSlider = jQuery('<div/>', {id: self.buttonText+'Slider'}).appendTo('#'+self.div);
    self.gripperSlider.off("slidestop"
                   ).on("slidestop.rfh",
                        function (event, ui) {
                            self.gripper.setPosition(self.gripperSlider.slider("value"));
                        }
                   ).hide(
                   ).css({"position":"absolute",
                          "bottom":"3%",
                          "height":"3%",
                           self.side:"3%"}
                   );

    self.gripperSlider.slider({
        min: 0.0,
        max: 0.085,
        step: 0.001,
        orientation: 'horizontal'});

    self.start = function () {
        $('#'+self.buttonText+'Slider').show();
    }
    
    self.stop = function () {
        $('#'+self.buttonText+'Slider').hide();
    };

    self.gripperStateDisplay = function (msg) {
        if (!$('#'+self.buttonText+'Slider > a').hasClass('ui-state-active')) {
            $('#'+self.buttonText+'Slider').slider('option', 'value', msg.actual.positions[0]);
        }
    };
    self.gripper.stateCBList.push(self.gripperStateDisplay);
}
