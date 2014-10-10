RFH.Torso = function (options) {
    'use strict';
    var self = this;
    self.div = options.div || 'markers';
    self.torso = options.torso; 
    self.buttonText = 'Spine';
    self.buttonClass = 'spine-button';
    self.torsoSlider = jQuery('<div/>', {id: self.buttonText+'Slider'}).appendTo('#'+self.div);
    self.torsoSlider.off("slidestop"
                   ).on("slidestop.rfh",
                        function (event, ui) {
                            self.torso.setPosition(self.torsoSlider.slider("value"));
                        }
                   ).hide(
                   ).css({"position":"absolute",
                          "top":"10%",
                          "height":"80%",
                          "left":"3%"}
                   );

    self.torsoSlider.slider({
        min: 0.0,
        max: 0.3,
        step: 0.01,
        orientation: 'vertical'});

    self.start = function () {
        $('#'+self.buttonText+'Slider').show();
    }
    
    self.stop = function () {
        $('#'+self.buttonText+'Slider').hide();
    };

    self.torsoStateDisplay = function (msg) {
        if (!$('#'+self.buttonText+'Slider > a').hasClass('ui-state-active')) {
            $('#'+self.buttonText+'Slider').slider('option', 'value', msg.actual.positions[0]);
        }
    };
    self.torso.stateCBList.push(self.torsoStateDisplay);

}
