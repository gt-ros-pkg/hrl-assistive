RFH = (function(module) {
    module.HeartbeatMonitor = function (options) {
    'use strict';
    var self = this;
    var ros = options.ros;
    var lastSent = null;
    var latestReceived = null;
    var lostConnection = false;
    
    var heartbeatPub = new ROSLIB.Topic({
        ros: ros,
        name: "web_heartbeat",
        messageType: 'std_msgs/String'
    });
    heartbeatPub.advertise();

    var heartbeatReplySub = new ROSLIB.Topic({
        ros: ros,
        name: "web_heartbeat_reply",
        messageType: 'std_msgs/String'
    });
    heartbeatReplySub.subscribe(processReply);

    var sendHeatbeat = function () {
        lastSent = new Date();
        heartbeatPub.publish({data: lastSent.toJSON()});
    };
    var sendTimer = setInterval(sendHeatbeat, 5000);
    
    var processReply = function (hbMsg) {
        var newlyReceived = new Date(hbMsg.data);
        if (newlyReceived - latestReceived > 0) {
            latestReceived = newlyReceived;
        }
    };

    var grayVideo = function () {
        var w = $('body').width();
        var h = $('body').height();
        $('#image-cover').css({'height':h, 'width':w}).text("Poor or lost connection").addClass('connection-lost').show();
        lostConnection = true;
    };

    var clearVideo = function () {
        $('#image-cover').hide();
        lostConnection = true;
    };

    var checkHeartbeat = function () {
        now = new Date();
        delay = (now - latestReceived) / 1000; // Convert ms to seconds
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
    var checkTimer = setInterval(checkHeartbeat, 5000);
    
    };
    return module;
})(RFH || {});
