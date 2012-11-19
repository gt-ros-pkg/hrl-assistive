var SkinUtilities = function(ros) {
    'use strict';
    var skutil = this;
    skutil.ros = ros;
    skutil.pos_weight = undefined;
    skutil.orient_weight = undefined;
    skutil.taxelArrayTopics = [];
    skutil.ros.getMsgDetails('hrl_haptic_manipulation_in_clutter_msgs/HapticMpcWeights');

    //Util for re-zeroing skin
    skutil.zero_pub_topics = [
    '/pr2_fabric_gripper_sensor/zero_sensor',
    '/pr2_fabric_gripper_left_link_sensor/zero_sensor',
    '/pr2_fabric_gripper_right_link_sensor/zero_sensor',
    '/pr2_fabric_gripper_palm_sensor/zero_sensor', 
    '/pr2_fabric_forearm_sensor/zero_sensor', 
    '/pr2_fabric_upperarm_sensor/zero_sensor', 
    '/pr2_pps_left_sensor/zero_sensor',
    '/pr2_pps_right_sensor/zero_sensor'];
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
    skutil.mpcWeightsSub = new skutil.ros.Topic({
        name:'/haptic_mpc/current_weights',
        messageType:'hrl_haptic_manipulation_in_clutter_msgs/HapticMpcWeights'})
    skutil.updateMpcWeights = function (msg) {
        skutil.pos_weight = msg.pos_weight;
        skutil.orient_weight = msg.pos_weight;
    };
    skutil.mpcWeightsSubCBList = [skutil.updateMpcWeights]
    skutil.mpcWeightsSubCB = function (msg) {
        for (var i=0; i<skutil.mpcWeightsSubCBList.length; i += 1) {
            skutil.mpcWeightsSubCBList[i](msg);
        };
    };
    skutil.mpcWeightsSub.subscribe(skutil.mpcWeightsSubCB);

    skutil.mpcWeightsPub = new skutil.ros.Topic({
        name:'/haptic_mpc/weights',
        messageType:'hrl_haptic_manipulation_in_clutter_msgs/HapticMpcWeights'});
    skutil.mpcWeightsPub.advertise();

    skutil.setMpcWeights = function (pos_weight, orient_weight) {
        var msg = new window.ros.composeMsg('hrl_haptic_manipulation_in_clutter_msgs/HapticMpcWeights');
        msg.pos_weight = pos_weight;
        msg.orient_weight = orient_weight;
        skutil.mpcWeightsPub.publish(msg);
    };

    //Add and remove taxel array topics from the skin client
    skutil.addTaxelArrayPub = new skutil.ros.Topic({
        name:'/haptic_mpc/add_taxel_array',
        messageType: 'std_msgs/String'});
    skutil.addTaxelArrayPub.advertise();

    skutil.removeTaxelArrayPub = new skutil.ros.Topic({
        name:'/haptic_mpc/remove_taxel_array',
        messageType: 'std_msgs/String'});
    skutil.removeTaxelArrayPub.advertise();

    skutil.addTaxelArray = function (topic) {
        var msg = new skutil.ros.Message({data:topic});
        skutil.addTaxelArrayPub.publish(msg);
        console.log('Adding '+topic+' topic to skin listeners');
    };

    skutil.removeTaxelArray  = function (topic) {
        var msg = new skutil.ros.Message({data:topic});
        skutil.removeTaxelArrayPub.publish(msg);
        console.log('Removing '+topic+' topic from skin listeners');
    };

    skutil.taxelArrayListSub = new skutil.ros.Topic({
        name: '/haptic_mpc/skin_topics',
        messageType: 'hrl_haptic_manipulation_in_clutter_msgs/StringArray'});
    skutil.updateTaxelArrayTopicList = function (msg) {
        skutil.taxelArrayTopics = msg.strings;
    };
    skutil.taxelArrayListSubCBList = [skutil.updateTaxelArrayTopicList];
    skutil.taxelArrayListSubCB = function (msg) {
        for (var i=0; i<skutil.taxelArrayListSubCBList.length; i += 1) {
            skutil.taxelArrayListSubCBList[i](msg);
        };
    };
    skutil.taxelArrayListSub.subscribe(skutil.taxelArrayListSubCB);
};

var initSkinUtils = function () {
    $('#underVideoBar').append(
      '<td><input type="checkbox" id="skinUseOrientation">Using Orientation</input></td>'+
      '<td><input type="checkbox" id="skinUsePPS">Using PPS Sensors</input></td>'+
      '<td><button id="rezeroSkinButton">Rezero Skin</input></td>');

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

    var updateOrientationCheckbox = function (msg) {
        if (msg.orient_weight !== 0.) {
            $('#skinUseOrientation').attr('checked',true);
            console.log('Received: USING orientation')
        } else {
            $('#skinUseOrientation').attr('checked',false);
            console.log('Received: NOT using orientation')
        };
    }
    window.skinUtil.mpcWeightsSubCBList.push(updateOrientationCheckbox);
        
    $('#rezeroSkinButton').click(function () {
        window.skinUtil.rezeroSkin();        
    });
    
    $('#skinUsePPS').change(function () {
        if (this.checked) {
            window.skinUtil.addTaxelArray('/pr2_pps_left_sensor/taxels/forces');
            window.skinUtil.addTaxelArray('/pr2_pps_right_sensor/taxels/forces');
            log('Turning on PPS Sensors');
        } else {
            window.skinUtil.removeTaxelArray('/pr2_pps_left_sensor/taxels/forces');
            window.skinUtil.removeTaxelArray('/pr2_pps_right_sensor/taxels/forces');
            log('Turning off PPS Sensors');
        };
    });
    var updateUsePPSCheckbox = function (msg) {
        var using_pps = false;
        for (var i=0; i<msg.strings.length; i += 1) {
            console.log(msg.strings[i]);
            if (msg.strings[i] == '/pr2_pps_left_sensor/taxels/forces') {
                using_pps = true;
            } else if (msg.strings[i] == '/pr2_pps_right_sensor/taxels/forces') {
                using_pps = true;
            }
        }
        if (using_pps) {
            console.log('Received: Using PPS Sensors');
            $('#skinUsePPS').attr('checked', true);
        } else {
            console.log('Received: NOT using PPS Sensors');
            $('#skinUsePPS').attr('checked', false);
        };
    };
    window.skinUtil.taxelArrayListSubCBList.push(updateUsePPSCheckbox);
}
