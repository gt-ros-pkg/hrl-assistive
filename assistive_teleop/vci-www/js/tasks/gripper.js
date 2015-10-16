RFH.GripperDisplay = function (options) {
    "use strict";
    var self = this;
    var gripper = options.gripper;
    var $div = $('#'+options.divId);
    var $gripperSlider = $div.find('.gripper-slider');
    var $grabButton = $div.find('.grab').button();
    var $releaseButton = $div.find('.release').button();
    $gripperSlider.slider({
        range: true,
        min: 0.0,
        max: 0.085,
        step: 0.001,
        orientation: 'horizontal'});

    self.show = function () { $div.show(); };
    self.hide = function () { $div.hide(); };
    self.hide(); // Hide on init

    $grabButton.on('click', function () { gripper.grab(); });
    $releaseButton.on('click', function() { gripper.release(); });

    
    // Set up slider display
    $gripperSlider.css({"background":"rgba(50,50,50,0.72)" });
    $gripperSlider.find('.ui-slider-range').css({"background":"rgba(22,22,22,0.9)",
                                                     "text-align":"center"}).html("Gripper");
    $gripperSlider.find('.ui-slider-handle').css({"height":"160%",
                                                 "top":"-30%",
                                                      "width":"7%",
                                                      "margin-left":"-3.5%",
                                                      "background":"rgba(42,42,42,1)",
                                                      "border":"2px solid rgba(82,82,82,0.87)"});
    var min = $gripperSlider.slider("option", "min");
    var max = $gripperSlider.slider("option", "max");
    self.mid = min + 0.5 * (max - min);

    // Stop/start/slide callbacks for slider (don't move with state updates while the user is controlling)
    var stopCB = function (event, ui) {
        var values = $gripperSlider.slider("option", "values");
        gripper.setPosition(values[1] - values[0]);
    };
    $gripperSlider.off("slidestop").on("slidestop.rfh", stopCB);

    var startCB = function (event, ui) {
        $gripperSlider.find('a').addClass('ui-state-active');
    };
    $gripperSlider.off("slidestart").on("slidestart.rfh", startCB);

    // Make both sides of display open/close together
    var slideCB = function (event, ui) {
        if (ui.values.indexOf(ui.value) === 0) {//left/low side
            var high = self.mid + (self.mid - ui.value);
            $gripperSlider.slider("option", "values", [ui.value, high]);
        } else { //right/high side
            var low = self.mid - (ui.value - self.mid);
            $gripperSlider.slider("option", "values", [low, ui.value]);
        }
    };
    $gripperSlider.off("slide").on("slide.rfh", slideCB);

    // Update state display, add to gripper state CB list
    self.gripperStateDisplay = function (msg) {
        if (!$gripperSlider.find('a').hasClass('ui-state-active')) {
            var high = self.mid + 0.5 * msg.process_value;
            var low = self.mid - 0.5 * msg.process_value;
            $gripperSlider.slider('option', 'values', [low, high]);
        }
    };
    gripper.stateCBList.push(self.gripperStateDisplay);
};
