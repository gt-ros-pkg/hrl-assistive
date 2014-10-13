RFH.Torso = function (options) {
    'use strict';
    var self = this;
    self.div = options.div || 'markers';
    self.torso = options.torso; 
    self.buttonText = 'Spine';
    self.buttonClass = 'spine-button';
    self.tallIcon = jQuery('<img/>', {id: "tallIcon", src:"./css/icons/tall.png"}).appendTo('#'+self.div);
    self.tallIcon.css({"height":"12%",
                        "width":"auto",
                        "position":"absolute",
                        "top":"2%",
                        "left":"2%"}).hide();
    self.shortIcon = jQuery('<img/>', {id: "shortIcon", src:"./css/icons/short.png"}).appendTo('#'+self.div);
    self.shortIcon.css({"height":"11%",
                        "width":"auto",
                        "position":"absolute",
                        "bottom":"2%",
                        "left":"2%"}).hide();
    self.torsoSlider = jQuery('<div/>', {id: self.buttonText+'Slider'}).appendTo('#'+self.div);
    self.torsoSlider.off("slidestop"
                   ).on("slidestop.rfh",
                        function (event, ui) {
                            self.torso.setPosition(self.torsoSlider.slider("value"));
                        }
                   ).hide(
                   ).css({"position":"absolute",
                          "top":"16%",
                          "height":"68%",
                          "left":"2%"}
                   );

    self.torsoSlider.slider({
        min: 0.011,
        max: 0.325,
        step: 0.01,
        orientation: 'vertical'});

    self.start = function () {
        $('#'+self.buttonText+'Slider, #tallIcon, #shortIcon').show();
    }
    
    self.stop = function () {
        $('#'+self.buttonText+'Slider, #tallIcon, #shortIcon').hide();
    };

    self.torsoStateDisplay = function (msg) {
        if (!$('#'+self.buttonText+'Slider > a').hasClass('ui-state-active')) {
            $('#'+self.buttonText+'Slider').slider('option', 'value', msg.actual.positions[0]);
        }
    };
    self.torso.stateCBList.push(self.torsoStateDisplay);

}
