RFH.GripperDisplay = function (options) {
    "use strict";
    var self = this;
    self.gripper = options.gripper;
    self.parentId = options.parentId;
    self.gripperSlider = jQuery('<div/>', {id: options.divId}).appendTo('#'+self.parentId);
    self.gripperSlider.slider({
        range: true,
        min: 0.0,
        max: 0.085,
        step: 0.001,
        orientation: 'horizontal'});
    
    self.gripperSlider.css({"background":"rgba(50,50,50,0.72)" });
    self.gripperSlider.find('.ui-slider-range').css({"background":"rgba(22,22,22,0.9)",
                                                     "text-align":"center"}).html("Gripper");
    self.gripperSlider.find('.ui-slider-handle').css({"height":"160%",
                                                      "top":"-30%",
                                                      "width":"7%",
                                                      "margin-left":"-3.5%",
                                                      "background":"rgba(42,42,42,1)",
                                                      "border":"2px solid rgba(82,82,82,0.87)"});
    var min = self.gripperSlider.slider("option", "min");
    var max = self.gripperSlider.slider("option", "max");
    self.mid = min + 0.5 * (max - min);

    self.stopCB = function (event, ui) {
        var values = self.gripperSlider.slider("option", "values");
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
            self.gripperSlider.slider("option", "values", [ui.value, high]);
        } else { //right/high side
            var low = self.mid - (ui.value - self.mid);
            self.gripperSlider.slider("option", "values", [low, ui.value]);
        }
    };
    self.gripperSlider.off("slide").on("slide.rfh", self.slideCB);

    self.gripperStateDisplay = function (msg) {
        if (!self.gripperSlider.find('a').hasClass('ui-state-active')) {
            var high = self.mid + 0.5 * msg.process_value;
            var low = self.mid - 0.5 * msg.process_value;
            self.gripperSlider.slider('option', 'values', [low, high]);
        }
    };
    self.gripper.stateCBList.push(self.gripperStateDisplay);
};
