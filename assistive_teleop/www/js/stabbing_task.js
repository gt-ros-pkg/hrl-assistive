function unselectedButton(button_id) {
    $(button_id).css("opacity", "1.0");
}
function selectedButton(button_id) {
    $(button_id).css("opacity", "0.3");
    //$(button_id).css("color", "blue");
}


var StabInterface = function (ros) {
    'use strict';
    var stabTabUI = this;
    stabTabUI.ros = ros;

    stabTabUI.available  = true;
    stabTabUI.handshaked = false;

    // --------------------------------------------------------
    // Publisher
    // --------------------------------------------------------    
    stabTabUI.bowlOffsetPub = new stabTabUI.ros.Topic({
        name: 'hrl_manipulation_task/bowl_highest_point',
        messageType: 'geometry_msgs/Point' });
    stabTabUI.bowlOffsetPub.advertise();

    stabTabUI.statusPub = new stabTabUI.ros.Topic({
        name: "manipulation_task/status",
        messageType: 'std_msgs/String', });
    stabTabUI.statusPub.advertise();
    

    // --------------------------------------------------------
    // Subscriber
    // --------------------------------------------------------    
    stabTabUI.guiStatusSub = new stabTabUI.ros.Topic({
        name: 'manipulation_task/gui_status',
        messageType: 'std_msgs/String'});
    stabTabUI.availableSub = new stabTabUI.ros.Topic({
        name: 'manipulation_task/available',
        messageType: 'std_msgs/String'});

    // --------------------------------------------------------
    // Offset
    // --------------------------------------------------------    
    stabTabUI.bowlOffset = function(id, x, y, z) {

        unselectedButton('#bpd_bowl_offset #b9');
        unselectedButton('#bpd_bowl_offset #b8');
        unselectedButton('#bpd_bowl_offset #b7');

        unselectedButton('#bpd_bowl_offset #b6');
        unselectedButton('#bpd_bowl_offset #b5');
        unselectedButton('#bpd_bowl_offset #b4');

        unselectedButton('#bpd_bowl_offset #b3');
        unselectedButton('#bpd_bowl_offset #b2');
        unselectedButton('#bpd_bowl_offset #b1');

        selectedButton('#bpd_bowl_offset '+id);


        var pointMsg = stabTabUI.ros.composeMsg('geometry_msgs/Point');
        pointMsg.x = x;
        pointMsg.y = y;
        pointMsg.z = 0; //z
        var msg = new stabTabUI.ros.Message(pointMsg);
        stabTabUI.bowlOffsetPub.publish(msg);
    }
    
    // --------------------------------------------------------
    // Feeding Button
    // --------------------------------------------------------
    stabTabUI.stab = function () {
        if (stabTabUI.available) {
            var msg = new stabTabUI.ros.Message({
                data: 'Scooping'
            });
            stabTabUI.statusPub.publish(msg);
            stabTabUI.current_step = 0;
            stabTabUI.max_step = 5;
            stabTabUI.available=false;
            return true;
        } else {
            return false;
        }
    };

    stabTabUI.feed = function () {
        if (stabTabUI.available) {
            var msg = new stabTabUI.ros.Message({
                data: 'Feeding'
            });
            stabTabUI.statusPub.publish(msg);
            stabTabUI.current_step = 0;
            stabTabUI.max_step = 5;
            stabTabUI.available=false;
            return true;
        } else {
            return false;
        }
    };

    stabTabUI.guiStatusSub.subscribe(function(msg) {
        if(msg.data == 'select task' || msg.data == 'stopped') {
            enableButton('#stab_task_Stabbing');
            enableButton('#stab_task_Feeding');
            stabTabUI.available=true;
        } else {
            disableButton('#stab_task_Stabbing');
            disableButton('#stab_task_Feeding');
            stabTabUI.available=false;
        }
        stabTabUI.handshaked = true;
    });

    stabTabUI.availableSub.subscribe(function (msg) {
        if(msg.data=="true") {
            stabTabUI.available=true;
        } else {
            stabTabUI.available=false;
        }
    });
    
}

var initStabInterface = function (tabDivId) {
    assistive_teleop.stabTabUI = new StabInterface(assistive_teleop.ros);
    //var divRef = "#"+tabDivId;

    $('.bpd, .man_task_cont').button();
    $('#stab_task_Stabbing').click(function(){
        if(assistive_teleop.stabTabUI.handshaked) {
            log(assistive_teleop.stabTabUI.available);
            assistive_teleop.stabTabUI.stab(); }
    });
    $('#stab_task_Feeding').click(function(){
        if(assistive_teleop.stabTabUI.handshaked) {
            assistive_teleop.stabTabUI.feed(); }
    });
    
    log('Controlling bowl offset');
    $('#bpd_default, #bpd_default_rot, #cart_frame_select, #cart_frame_select_label, #cart_controller, #cart_cont_state_check').show();
    $('#bpd_bowl_offset').find(':button').unbind('.rfh').text('');

    $('#bpd_bowl_offset :button').unbind('.rfh');


    var offset = 0.015;
    $('#bpd_bowl_offset #b9').show().bind('click.rfh', function (e) {
        assistive_teleop.stabTabUI.bowlOffset('#b9', -offset, -offset, 0); });
    $('#bpd_bowl_offset #b6').show().bind('click.rfh', function (e) {
        assistive_teleop.stabTabUI.bowlOffset('#b6', -offset, 0.0, 0); });
    $('#bpd_bowl_offset #b3').show().bind('click.rfh', function (e) {
        assistive_teleop.stabTabUI.bowlOffset('#b3', -offset, offset, 0); });

    $('#bpd_bowl_offset #b8').show().bind('click.rfh', function (e) {
        assistive_teleop.stabTabUI.bowlOffset('#b8', 0.0, -offset, 0); });
    $('#bpd_bowl_offset #b5').show().bind('click.rfh', function (e) {
        assistive_teleop.stabTabUI.bowlOffset('#b5', 0.0, 0.0, 0); });
    $('#bpd_bowl_offset #b2').show().bind('click.rfh', function (e) {
        assistive_teleop.stabTabUI.bowlOffset('#b2', 0.0, offset, 0); });

    $('#bpd_bowl_offset #b7').show().bind('click.rfh', function (e) {
        assistive_teleop.stabTabUI.bowlOffset('#b7', offset, -offset, 0); });
    $('#bpd_bowl_offset #b4').show().bind('click.rfh', function (e) {
        assistive_teleop.stabTabUI.bowlOffset('#b4', offset, 0.0, 0); });
    $('#bpd_bowl_offset #b1').show().bind('click.rfh', function (e) {
        assistive_teleop.stabTabUI.bowlOffset('#b1', offset, offset, 0); });

}

