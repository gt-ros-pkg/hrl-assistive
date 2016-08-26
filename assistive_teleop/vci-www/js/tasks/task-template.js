// RFH (from Robots for Humanity) is a global variable that holds all of the web interface javascript content
var RFH = (function (module) {
    module.Task = function(options) { // Rename 'Task' for your particular task (should be unique among tasks)
        'use strict';
        var self = this;
        options = options || {}; // Initialize the options object as an empty object if none is given
        self.name = options.name || 'tasknameTask'; // Required: Task.name  (use name passed in, or default name here if not given)
        self.showButton = true; // Required: used to decide to create clickable button in menu or only allow intirect access
        self.buttonText = 'Spine'; // Required: human-readable, descriptive task name
        self.buttonClass = 'spine-button'; // Required: class used for setting css rules 

        // Required: start function with no params.  
        // Called when button is clicked.
        // Performs setup (typically making loaded resources visible).
        self.start = function() {
            $('#' + self.sliderDiv + ', #tallIcon, #shortIcon').show();
        };

        //Required: Stop function with no params.
        // Called when switching away to a different task.
        // Should hide all specific interface elements, and release any shared resources.
        self.stop = function() {
            $('#' + self.sliderDiv + ', #tallIcon, #shortIcon').hide();
        };

        // Initialization/loading of any interface elements.
        $('#' + self.sliderDiv).slider({
            min: 0.011,
            max: 0.325,
            step: 0.01,
            orientation: 'vertical'
        })
            .on("slidestop.rfh", function(event, ui) {
                self.torso.setPosition($('#' + self.sliderDiv).slider("value"));
            });

        //  All other functions required for interacting with the interface.
        self.torsoStateDisplay = function(msg) {
            if (!$('#' + self.sliderDiv + ' > a').hasClass('ui-state-active')) {
                $('#' + self.sliderDiv).slider('option', 'value', msg.actual.positions[0]);
            }
        };
        self.torso.stateCBList.push(self.torsoStateDisplay);
    };
    return module;
})(RFH || {});
