var RFH = (function(module) {
    module.PingRelay = function (options) {
    'use strict';
    var ros = options.ros;
    var inTopic = options.inTopic;
    var outTopic = options.outTopic;
    
    var relayPub = new ROSLIB.Topic({
        ros: ros,
        name: outTopic,
        messageType: 'std_msgs/Time'
    });
    relayPub.advertise();

    var relay = function (pingMsg) {
        relayPub.publish(pingMsg);
    };

    var relaySub = new ROSLIB.Topic({
        ros: ros,
        name: inTopic,
        messageType: 'std_msgs/Time'
    });
    relaySub.subscribe(relay);
    };
    return module;
})(RFH || {});
