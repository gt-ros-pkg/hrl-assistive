RFH = (function(module) {
    module.HeartbeatMonitor = function (options) {
    'use strict';
    var self = this;
    var ros = options.ros;
    var now = new Date();
    var latestReceived = now.setSeconds(now.getSeconds() + 10);
    var lostConnection = false;
    
    var heartbeatPub = new ROSLIB.Topic({
        ros: ros,
        name: "web_heartbeat",
        messageType: 'std_msgs/String'
    });
    heartbeatPub.advertise();

    var processReply = function (hbMsg) {
        var newlyReceived = new Date(hbMsg.data);
        if (newlyReceived - latestReceived > 0) {
            latestReceived = newlyReceived;
        }
    };

    var heartbeatReplySub = new ROSLIB.Topic({
        ros: ros,
        name: "web_heartbeat_reply",
        messageType: 'std_msgs/String'
    });
    heartbeatReplySub.subscribe(processReply);

    var sendHeatbeat = function () {
        heartbeatPub.publish({data: new Date().toJSON()});
    };
    var sendTimer = setInterval(sendHeatbeat, 2500);
    
    var grayVideo = function () {
        var w = $('body').width();
        var h = $('body').height();
        $('#image-cover').css({'height':h, 'width':w}).text("Poor or lost connection").addClass('connection-lost').show();
        lostConnection = true;
    };

    var clearVideo = function () {
        $('#image-cover').hide();
        lostConnection = false;
    };

    var checkHeartbeat = function () {
        var now = new Date();
        var delay = (now - latestReceived) / 1000; // Convert ms to seconds
        if (delay > 5) {
            if (!lostConnection){
                grayVideo();
            }
        } else {
            if (lostConnection) {
                clearVideo();
            }
        }
    };
    var checkTimer = setInterval(checkHeartbeat, 5500);
    
    };
    return module;
})(RFH || {});
