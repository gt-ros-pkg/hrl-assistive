var SkinUtilities = function(ros) {
    'use strict';
    var skutil = this;
    skutil.ros = ros;
    skutil.ros.getMsgDetails('hrl_haptic_manipulation_in_clutter_msgs/HapticMpcWeights');

    //Util for re-zeroing skin
    skutil.zero_pub_topics = [
    '/pr2_fabric_gripper_sensor/zero_sensor',
    '/pr2_fabric_gripper_left_link_sensor/zero_sensor',
    '/pr2_fabric_gripper_right_link_sensor/zero_sensor',
    '/pr2_fabric_gripper_palm_sensor/zero_sensor', 
    '/pr2_fabric_forearm_sensor/zero_sensor', 
    '/pr2_fabric_upperarm_sensor/zero_sensor', 
    '/pr2_pps_sensor/zero_sensor'];
    skutil.zero_pubs = [];
    for (var i=0; i<skutil.zero_pub_topics.length; i += 1) {
        skutil.zero_pubs[i] = new skutil.ros.Topic({
                            name:skutil.zero_pub_topics[i],
                            messageType:'std_msgs/Empty'});
        skutil.zero_pubs[i].advertise();
    }
    skutil.rezeroSkin = function () {
        for (var i=0; i<skutil.zero_pubs.length; i += 1) {
            skutil.zero_pubs[i].publish(new skutil.ros.Message({}))
        };
    };
    
    //Util for setting weights of position/orientation
    skutil.mpcWeightsPub = new skutil.ros.Topic({
        name:'/haptic_mpc/weights',
        messageType:'haptic_msgs/HapticMpcWeights'});
    skutil.mpcWeightsPub.advertise();

    skutil.setMpcWeights = function (pos_weight, orient_weight) {
        var msg = new window.ros.composeMsg('hrl_haptic_manipulation_in_clutter_msgs/HapticMpcWeights');
        msg.pos_weight = pos_weight;
        msg.orient_weight = orient_weight;
        skutil.mpcWeightsPub.publish(msg);
    };
}

var initSkinUtils = function () {
    window.skinUtil = new SkinUtilities(window.ros);
    $('#skinUseOrientation').change(function () {
        if (this.checked) {
            window.skinUtil.setMpcWeights(5.0, 4.0);
            log('Turning On Orientation')
        } else {
            window.skinUtil.setMpcWeights(5.0, 0.0);
            log('Turning Off Orientation');
        };
    })
    $('#rezeroSkinButton').click(function () {
        window.skinUtil.rezeroSkin();        
    });
}
