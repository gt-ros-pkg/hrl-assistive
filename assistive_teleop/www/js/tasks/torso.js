RFH.Torso = function (options) {
    'use strict';
    var self = this;
    self.containerDiv = options.containerDiv;
    self.sliderDiv = options.sliderDiv;
    self.torso = options.torso; 
    self.buttonText = 'Spine';
    self.buttonClass = 'spine-button';

    $('#'+self.sliderDiv).slider({min: 0.011, max: 0.325,
                             step: 0.01, orientation: 'vertical'})
                          .on("slidestop.rfh", function (event, ui){
                               self.torso.setPosition($('#'+self.sliderDiv).slider("value")); } 
    );

    self.start = function () {
        $('#'+self.sliderDiv+', #tallIcon, #shortIcon').show();
    }
    
    self.stop = function () {
        $('#'+self.sliderDiv+', #tallIcon, #shortIcon').hide();
    };

    self.torsoStateDisplay = function (msg) {
        if (!$('#'+self.sliderDiv +' > a').hasClass('ui-state-active')) {
            $('#'+self.sliderDiv).slider('option', 'value', msg.actual.positions[0]);
        }
    };
    self.torso.stateCBList.push(self.torsoStateDisplay);

}
