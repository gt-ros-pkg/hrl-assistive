var RFH = (function (module) {
    module.DataLogger = function (options) {
        'use strict';
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

        /*
        var idClassString = function (element) {
            var str = element.tagName + '#'+element.id;
            for (var i=0; i < element.classList.length; i += 1) {
                str += '.'+element.classList[i].toString();
            }
            return str;
        };

        var getParentage = function (element) {
            var str = idClassString(element);
            while (element.parentElement !== null) {
                element = element.parentElement;
                str = idClassString(element) + ' > ' + str;
            }
            return str;
        };
        */

        var logEvent = function (event, ui) {
            var msg = ros.composeMsg(msgType);
            msg.type = event.type;
            msg.target = event.currentTarget.id;
            logPub.publish(msg);
        };

        $('.log-click').on('click.datalogging', logEvent);
        $(".log-slide").on("slidestart.datalogging, slidestop.datalogging", logEvent);
        $('.log-open').on("selectmenuopen.datalogging", logEvent);
        $('.log-select').on("selectmenuselect.datalogging", logEvent);
        $('.log-activate').on("accordionactivate.datalogging", logEvent);
        $(".log-mousehold").on("mousedown.datalogging, mouseup.datalogging, mouseout.datalogging, mouseleave.datalogging, blur.datalogging, ", logEvent);

        // Special handling of mjpeg image to only log when active..
        var logLookClick = function (event, ui) {
            if ($(event.currentTarget).hasClass('cursor-eyes')) {
                logEvent(event, ui);
            }
        };
        $('#mjpeg-image').off('click.datalogging').on('click.datalogging', logLookClick);


    };
    return module;

})(RFH||{});
