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
    
    $('#'+self.divId+' .ui-slider-range').css({"background":"rgba(152,152,152,0.4)",
                                               "text-align":"center"}).html("Gripper");
    $('#'+self.divId+'.ui-slider').css({"background":"rgba(50,50,50,0.12)" });
    $('#'+self.divId+' .ui-slider-handle').css({"height":"160%",
                                                "top":"-30%",
                                                "width":"7%",
                                                "margin-left":"-3.5%",
                                                "background":"rgba(42,42,42,1)",
                                                "border":"2px solid rgba(82,82,82,0.87)"});
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
