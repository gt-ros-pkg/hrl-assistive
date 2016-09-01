var RFH = (function (module) {
    module.Torso = function (options) {
        'use strict';
        var self = this;
        self.name = options.name || 'torsoAction';
        self.showButton = true;
        self.buttonID = options.buttonID || 'torsoModeButton';
        self.toolTipText = "Raise or lower the spine";
        self.containerDiv = options.containerDiv;
        self.sliderDiv = options.sliderDiv;
        self.torso = options.torso; 
        $('#tallIcon').prop('title', "Raise the Spine");
        $('#shortIcon').prop('title', "Lower the Spine");

        self.icons = $('#tallIcon, #shortIcon');
        self.slider = $('#'+self.sliderDiv).slider({min: 0.011,
                                                    max: 0.325,
                                                    step: 0.01,
                                                    orientation: 'vertical'})
                                            .on("slidestop.rfh", function (event, ui){
                                               self.torso.setPosition($('#'+self.sliderDiv).slider("value")); } 
                                            ).prop('title', "Raise or Lower the Spine");

        self.start = function () {
            self.slider.show();
            self.icons.show();
        };
        
        self.stop = function () {
            self.slider.hide();
            self.icons.hide();
        };

        self.torsoStateDisplay = function (msg) {
            if (!self.slider.find('span').hasClass('ui-state-active')) {
                self.slider.slider('option', 'value', msg.actual.positions[0]);
            }
        };
        self.torso.stateCBList.push(self.torsoStateDisplay);
    };
    return module;
})(RFH || {});
