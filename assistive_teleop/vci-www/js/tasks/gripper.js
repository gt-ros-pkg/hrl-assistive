RFH.GripperDisplay = function (options) {
    "use strict";
    var self = this;
    var gripper = options.gripper;
    var zeroOffset = options.zeroOffset || 0.0;
    var $div = $('#'+options.divId);
    var $gripperSlider = $div.find('.gripper-slider');
    var $grabButton = $div.find('.grab').button();
    var $releaseButton = $div.find('.release').button();
    var handleWidthPct = 7; // Width of the slider handles relative to the slider;
    $gripperSlider.slider({
        range: true,
        min: 0.0,
        max: 0.09,
        step: 0.001,
        orientation: 'horizontal'});

    self.show = function () { $div.show(); };
    self.hide = function () { $div.hide(); };
    self.hide(); // Hide on init

    $grabButton.on('click', function () { gripper.grab(); });

    var releaseOnContactCB = function (event) {
        if ($releaseButton.prop('checked')) {
            gripper.releaseOnContact();
        } else {
            gripper.cancelReleaseOnContact();
        }
    };
    $releaseButton.on('click', releaseOnContactCB );
   
    var updateReleaseOnContact = function (msg) {
        if (!msg.data) {
           $releaseButton.prop('checked', false).button('refresh'); 
        }
    };
    gripper.graspingCBList.push(updateReleaseOnContact);

    // Set up slider display
    $gripperSlider.css({"background":"rgba(50,50,50,0.72)" });
    $gripperSlider.find('.ui-slider-range').css({"background":"rgba(22,22,22,0.9)",
                                                 "text-align":"center"}).html("Gripper");
    $gripperSlider.find('.ui-slider-handle').css({"height":"160%",
                                                  "top":"-30%",
                                                  "width":handleWidthPct.toString()+"%",
                                                  "margin-left":"-"+(handleWidthPct/2).toString()+"%",
                                                  "background":"rgba(42,42,42,1)",
                                                  "border":"2px solid rgba(82,82,82,0.87)"});
    $gripperSlider.find('a:first-of-type').css('border-right', 'none');
    $gripperSlider.find('a:last-of-type').css('border-left', 'none');
    var min = $gripperSlider.slider("option", "min");
    var max = $gripperSlider.slider("option", "max");
    var range = max - min;
    var mid = min + range/2;

    // Stop/start/slide callbacks for slider (don't move with state updates while the user is controlling)
    var stopCB = function (event, ui) {
        var values = $gripperSlider.slider("option", "values");
        var diff = values[1]-values[0];
        gripper.setPosition(diff - (handleWidthPct/100)*range);
    };
    $gripperSlider.off("slidestop").on("slidestop.rfh", stopCB);

    var startCB = function (event, ui) {
        $gripperSlider.find('a').addClass('ui-state-active');
    };
    $gripperSlider.off("slidestart").on("slidestart.rfh", startCB);

    // Make both sides of display open/close together
    var slideCB = function (event, ui) {
        //var aperture = getApertureFromSliders(ui.value);
        //setSlidersFromAperture(aperture);
        if (ui.values.indexOf(ui.value) === 0) {//left/low side
            var high = mid + (mid - ui.value) + (handleWidthPct/200)*range;
            $gripperSlider.slider("option", "values", [ui.value, high]);
        } else { //right/high side
            var low = mid - (ui.value - mid) - (handleWidthPct/200)*range;
            $gripperSlider.slider("option", "values", [low, ui.value]);
        }
    };
    $gripperSlider.off("slide").on("slide.rfh", slideCB);

    var setSlidersFromAperture = function(open_dist) {
        var high = mid + open_dist/2 + (handleWidthPct/200)*range;
        var low = mid - open_dist/2 - (handleWidthPct/200)*range;
            $gripperSlider.slider('option', 'values', [low, high]);
    };

    var getApertureFromSliders = function (handle_position) {
        if (handle_position < mid) {
            return 2*(mid - handle_position);
         } else {
            return 2*(handle_position - mid);
         }

    };

    // Update state display, add to gripper state CB list
    self.gripperStateDisplay = function (msg) {
        if (!$gripperSlider.find('a').hasClass('ui-state-active')) {
            setSlidersFromAperture(msg.process_value-zeroOffset);
        }
    };
    gripper.stateCBList.push(self.gripperStateDisplay);
};
