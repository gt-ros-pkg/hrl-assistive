var RFH = (function (module) {
    module.RunStop = function (options) {
        var self = this;
        var ros = options.ros;
        var motorsHalted = null;
        var divId = options.divId || 'runstop-button';
        var $div = $('#'+divId);
        $div.button();
        var $textSpan = $div.find('span');

        var buttonCSSToReset = {
            background: '#00ff00',
            border: "2px #000000"
        };

        var buttonCSSToHalt = {
            background: '#ff0000',
            border: "2px #000000"
        };

        // Get the current motor status
        var updateMotorState = function (bool_msg) {
            if (motorsHalted === bool_msg.data) { return; }
            motorsHalted = bool_msg.data;
            if (motorsHalted) {
                $div.css(buttonCSSToReset);
                $textSpan.text('RESET MOTORS');
            } else {
                $div.css(buttonCSSToHalt);
                $textSpan.text('HALT MOTORS');
            }
        };

        self.motorsHalted = function () {
            return motorsHalted;
        };

        var motorStateSub = new ROSLIB.Topic({
            ros: ros,
            name: "/pr2_etherCAT/motors_halted",
            messageType: 'std_msgs/Bool'
        });
        motorStateSub.subscribe(updateMotorState);

        // Get msgs for handling power_board standby
        var runstopService = new ROSLIB.Service({
            ros: ros,
            name: '/emulate_runstop',
            serviceType: 'hrl_pr2_upstart/SetRunStop'
        });

        self.halt = function () {
            runstopService.callService({'stop':true, 'start':false}, function () {});
        };

        self.reset = function () {
            runstopService.callService({'stop':false, 'start':true}, function () {});
        };

        var clickCB = function (event) {
            if (motorsHalted) {
                self.reset();
            } else {
                self.halt();
            }
        };
        $div.on('click.rfh', clickCB);
    };
    return module;
})(RFH || {});
