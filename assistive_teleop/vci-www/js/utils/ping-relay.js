var RFH = (function(module) {
    module.PingRelay = function (options) {
    'use strict';
    var ros = options.ros;
    var inTopic = options.inTopic;
    var outTopic = options.outTopic;
    
    ros.getMsgDetails('assistive_teleop/PingMsg');
    var relayPub = new ROSLIB.Topic({
        ros: ros,
        name: outTopic,
        messageType: 'assistive_teleop/PingMsg'
    });
    relayPub.advertise();

    var stampAndReturn = function (pingMsg) {
        pingMsg.clientTime = new Date();
        relayPub.publish(pingMsg);
    };

    var relaySub = new ROSLIB.Topic({
        ros: ros,
        name: inTopic,
        messageType: 'assistive_teleop/PingMsg'
    });
    relaySub.subscribe(stampAndReturn);
    };
    return module;
})(RFH || {});
