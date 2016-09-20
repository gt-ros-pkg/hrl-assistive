var RFH = (function (module) {
    module.DataLogger = function (options) {
        'use strict';
        var self = this;
        var ros = options.ros;
        var logTopic = options.topic || 'interface_log';
        var msgType = 'assistive_teleop/InterfaceLog';
        ros.getMsgDetails(msgType);

        var logPub = new ROSLIB.Topic({
            ros: ros,
            name: logTopic,
            messageType: msgType
        });
        logPub.advertise();

        self.logCustomEvent = function (type, targetId) {
            logEvent({type:type, currentTarget:{id: targetId}});
        };

        var logEvent = function (event, ui) {
            var msg = ros.composeMsg(msgType);
            msg.type = event.type;
            msg.target = event.currentTarget.id;
            // msg.event_time = new Date() // Look up getTime, getSeconds, getMilliseconds, etc to make a decent timestamp
            logPub.publish(msg);
        };

        $('.log-click').on('click.datalogging', logEvent);
        $(".log-slide").on("slidestart.datalogging, slidestop.datalogging", logEvent);
        $('.log-open').on("selectmenuopen.datalogging", logEvent);
        $('.log-select').on("selectmenuselect.datalogging", logEvent);
        $('.log-activate').on("accordionactivate.datalogging", logEvent);
        $(".log-mousehold").on("mousedown.datalogging, mouseup.datalogging, mouseout.datalogging, mouseleave.datalogging, blur.datalogging, ", logEvent);

        /* Special handling of mjpeg image to only log when active */
        var logLookClick = function (event, ui) {
            var $curTar = $(event.currentTarget);
            if ($curTar.hasClass('cursor-eyes') || $curTar.hasClass('cursor-select')) {
                logEvent(event, ui);
            }
        };
        $('#mjpeg-image').off('click.datalogging').on('click.datalogging', logLookClick);


    };
    return module;

})(RFH||{});
