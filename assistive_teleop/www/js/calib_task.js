var CalibInterface = function (ros) {
    'use strict';
    var taskUI = this;
    taskUI.ros = ros;

    taskUI.feedingDepthSynched = true;
    taskUI.feedingHorizSynched = true;
    taskUI.feedingVertSynched  = true;
    taskUI.available  = true;
    taskUI.handshaked = false;

    // --------------------------------------------------------
    // Publisher
    // --------------------------------------------------------    
    taskUI.feedingDepthPub = new taskUI.ros.Topic({
        name: 'feeding/manipulation_task/mouth_depth_request',
        messageType: 'std_msgs/Int64' });
    taskUI.feedingDepthPub.advertise();
    taskUI.feedingHorizPub = new taskUI.ros.Topic({
        name: 'feeding/manipulation_task/mouth_horiz_request',
        messageType: 'std_msgs/Int64' });
    taskUI.feedingHorizPub.advertise();
    taskUI.feedingVertPub = new taskUI.ros.Topic({
        name: 'feeding/manipulation_task/mouth_vert_request',
        messageType: 'std_msgs/Int64' });
    taskUI.feedingVertPub.advertise();

    taskUI.statusPub = new taskUI.ros.Topic({
        name: "manipulation_task/status",
        messageType: 'std_msgs/String', });
    taskUI.statusPub.advertise();

    taskUI.resetPub = new taskUI.ros.Topic({
      name: 'manipulation_task/arm_reach_reset',
      messageType: 'std_msgs/String' });
    taskUI.resetPub.advertise();

    // --------------------------------------------------------
    // Subscriber
    // --------------------------------------------------------    
    taskUI.feedingDepthSub = new taskUI.ros.Topic({
        name: 'feeding/manipulation_task/mouth_depth_offset',
        messageType: 'std_msgs/Int64' });
    taskUI.feedingHorizSub = new taskUI.ros.Topic({
        name: 'feeding/manipulation_task/mouth_horiz_offset',
        messageType: 'std_msgs/Int64' });
    taskUI.feedingVertSub = new taskUI.ros.Topic({
        name: 'feeding/manipulation_task/mouth_vert_offset',
        messageType: 'std_msgs/Int64' });
    taskUI.guiStatusSub = new taskUI.ros.Topic({
        name: 'manipulation_task/gui_status',
        messageType: 'std_msgs/String'});
    taskUI.availableSub = new taskUI.ros.Topic({
        name: 'manipulation_task/available',
        messageType: 'std_msgs/String'});



    // --------------------------------------------------------
    // Offset
    // --------------------------------------------------------    
    // 1. feeding depth
    taskUI.feedingDepthRequest = function() {
        if (taskUI.feedingDepthSynched) {
            var new_dist = parseInt(document.getElementById("task_Feeding_depth_offset").value);
            var msg = new taskUI.ros.Message({
                data: new_dist
            });
            taskUI.feedingDepthPub.publish(msg);
            taskUI.feedingDepthSynched = false;
            document.getElementById("task_Feeding_depth_offset").disabled = true;
        }
    }

    taskUI.depth_pos = function() {
        if (taskUI.feedingDepthSynched) {
            var dist = parseInt(document.getElementById("task_Feeding_depth_offset").value)+1;
            if (dist>9) dist = 9;

            var msg = new taskUI.ros.Message({
                data: dist
            });
            taskUI.feedingDepthPub.publish(msg);
            taskUI.feedingDepthSynched = false;
            document.getElementById("task_Feeding_depth_offset").disabled = true;
        }
    }

    taskUI.depth_neg = function() {
        if (taskUI.feedingDepthSynched) {
            var dist = parseInt(document.getElementById("task_Feeding_depth_offset").value)-1;
            if (dist<2) dist = 2;

            var msg = new taskUI.ros.Message({
                data: dist
            });
            taskUI.feedingDepthPub.publish(msg);
            taskUI.feedingDepthSynched = false;
            document.getElementById("task_Feeding_depth_offset").disabled = true;
        }
    };


    taskUI.feedingDepthSub.subscribe(function (msg) {
        document.getElementById("task_Feeding_depth_offset").value = msg.data;
        taskUI.feedingDepthSynched = true;
        document.getElementById("task_Feeding_depth_offset").disabled = false;
    });

    // 2. feeding horizontal offset
    taskUI.feedingHorizRequest = function() {
        if (taskUI.feedingHorizSynched) {
            var new_dist = parseInt(document.getElementById("task_Feeding_horiz_offset").value);
            var msg = new taskUI.ros.Message({
                data: new_dist
            });
            taskUI.feedingHorizPub.publish(msg);
            taskUI.feedingHorizSynched = false;
            document.getElementById("task_Feeding_horiz_offset").disabled = true;
        }
    }

    taskUI.horiz_pos = function() {
        if (taskUI.feedingHorizSynched) {
            var dist = parseInt(document.getElementById("task_Feeding_horiz_offset").value)+1;
            if (dist>3) dist = 3;

            var msg = new taskUI.ros.Message({
                data: dist
            });
            taskUI.feedingHorizPub.publish(msg);
            taskUI.feedingHorizSynched = false;
            document.getElementById("task_Feeding_horiz_offset").disabled = true;
        }
    }

    taskUI.horiz_neg = function() {
        if (taskUI.feedingHorizSynched) {
            var dist = parseInt(document.getElementById("task_Feeding_horiz_offset").value)-1;
            if (dist<-3) dist = -3;

            var msg = new taskUI.ros.Message({
                data: dist
            });
            taskUI.feedingHorizPub.publish(msg);
            taskUI.feedingHorizSynched = false;
            document.getElementById("task_Feeding_horiz_offset").disabled = true;
        }
    }


    taskUI.feedingHorizSub.subscribe(function (msg) {
        document.getElementById("task_Feeding_horiz_offset").value = msg.data;
        taskUI.feedingHorizSynched = true;
        document.getElementById("task_Feeding_horiz_offset").disabled = false;
    });

    // 3. feeding vertical offset
    taskUI.feedingVertRequest = function() {
        if (taskUI.feedingVertSynched) {
            var new_dist = parseInt(document.getElementById("task_Feeding_vert_offset").value);
            var msg = new taskUI.ros.Message({
                data: new_dist
            });
            taskUI.feedingVertPub.publish(msg);
            taskUI.feedingVertSynched = false;
            document.getElementById("task_Feeding_vert_offset").disabled = true;
        }
    }

    taskUI.vert_pos = function() {
        if (taskUI.feedingVertSynched) {
            var dist = parseInt(document.getElementById("task_Feeding_vert_offset").value)+1;
            if (dist>1) dist = 1;

            var msg = new taskUI.ros.Message({
                data: dist
            });
            taskUI.feedingVertPub.publish(msg);
            taskUI.feedingVertSynched = false;
            document.getElementById("task_Feeding_vert_offset").disabled = true;
        }
    }

    taskUI.vert_neg = function() {
        if (taskUI.feedingVertSynched) {
            var dist = parseInt(document.getElementById("task_Feeding_vert_offset").value)-1;
            if (dist<-5) dist = -5;

            var msg = new taskUI.ros.Message({
                data: dist
            });
            taskUI.feedingVertPub.publish(msg);
            taskUI.feedingVertSynched = false;
            document.getElementById("task_Feeding_vert_offset").disabled = true;
        }
    }

    taskUI.feedingVertSub.subscribe(function (msg) {
        document.getElementById("task_Feeding_vert_offset").value = msg.data;
        taskUI.feedingVertSynched = true;
        document.getElementById("task_Feeding_vert_offset").disabled = false;
    });

    // --------------------------------------------------------
    // Reset Button
    // --------------------------------------------------------
    
    taskUI.reset_loc = function () {
      var msg = new taskUI.ros.Message({
        data: 'true'
      });
      taskUI.resetPub.publish(msg);
      assistive_teleop.log("Resetting internal parameters");
    }

    taskUI.reset_offset = function () {
        if (taskUI.feedingDepthSynched && taskUI.feedingHorizSynched && taskUI.feedingVertSynched) {

            var msg = new taskUI.ros.Message({
                data: 4
            });
            taskUI.feedingDepthPub.publish(msg);
            taskUI.feedingDepthSynched = false;
            document.getElementById("task_Feeding_depth_offset").disabled = true;
            
            var msg = new taskUI.ros.Message({
                data: 0
            });
            taskUI.feedingHorizPub.publish(msg);
            taskUI.feedingHorizSynched = false;
            document.getElementById("task_Feeding_horiz_offset").disabled = true;

            var msg = new taskUI.ros.Message({
                data: -3
            });
            taskUI.feedingVertPub.publish(msg);
            taskUI.feedingVertSynched = false;
            document.getElementById("task_Feeding_vert_offset").disabled = true;

            assistive_teleop.log("Resetting feeding offset");
        }
    }

    // --------------------------------------------------------
    // Feeding Button
    // --------------------------------------------------------
    taskUI.feed = function () {
        if (taskUI.available) {
            var msg = new taskUI.ros.Message({
                data: 'Feeding'
            });
            assistive_teleop.log('Please, follow the step 2 to select the action.');
            taskUI.statusPub.publish(msg);
            taskUI.current_step = 0;
            taskUI.max_step = 5;
            taskUI.available=false;
            return true;
        } else {
            return false;
        }
    }

    taskUI.guiStatusSub.subscribe(function(msg) {
        if(msg.data == 'select task' || msg.data == 'stopped') {
            enableButton('#reset');
            enableButton('#task_Feeding');
            taskUI.available=true;
        } else {
            disableButton('#reset');
            disableButton('#task_Feeding');
            taskUI.available=false;
        }
        taskUI.handshaked = true;
    });

    taskUI.availableSub.subscribe(function (msg) {
        if(msg.data=="true") {
            taskUI.available=true;
        } else {
            taskUI.available=false;
        }
    });

}

var initCalibInterface = function (tabDivId) {
    assistive_teleop.taskUI = new CalibInterface(assistive_teleop.ros);
    //var divRef = "#"+tabDivId;

    $('.bpd, .man_task_cont').button();
    $('#task_Feeding_depth_offset').change(function(){
        assistive_teleop.taskUI.feedingDepthRequest(); });
    $('#task_Feeding_horiz_offset').change(function(){
        assistive_teleop.taskUI.feedingHorizRequest(); });
    $('#task_Feeding_vert_offset').change(function(){
        assistive_teleop.taskUI.feedingVertRequest(); });

    $('#offset_reset').click(function(){
        assistive_teleop.taskUI.reset_offset(); });
    $('#loc_reset').click(function(){
        assistive_teleop.taskUI.reset_loc(); });
    $('#task_Feeding').click(function(){
        if(assistive_teleop.taskUI.handshaked) {
            assistive_teleop.taskUI.feed(); }
    });
    
    log('Controlling mouth offset');
    //$('#bpd_default, #bpd_default_rot, #cart_frame_select, #cart_frame_select_label, #cart_controller').hide();
    $('#bpd_default, #bpd_default_rot, #cart_frame_select, #cart_frame_select_label, #cart_controller, #cart_cont_state_check').show();
    $('#bpd_mouth_offset').find(':button').unbind('.rfh').text('');


    $('#bpd_mouth_offset :button').unbind('.rfh');
    $('#bpd_mouth_offset #b9').show().bind('click.rfh', function (e) {
        assistive_teleop.taskUI.depth_neg();
    });
    $('#bpd_mouth_offset #b8').show().bind('click.rfh', function (e) {
        assistive_teleop.taskUI.vert_pos();
    });
    $('#bpd_mouth_offset #b7').hide();
    $('#bpd_mouth_offset #b6').show().bind('click.rfh', function (e) {
        assistive_teleop.taskUI.horiz_pos();
    });
    $('#bpd_mouth_offset #b5').hide();
    //$('#b5').show().bind('click.rfh', function (e) {
    //    assistive_teleop.taskUI.reset_offset();
    //});
    //$('#b5').click(function(){
    //    assistive_teleop.taskUI.reset_offset(); });

    $('#bpd_mouth_offset #b4').show().bind('click.rfh', function (e) {
        assistive_teleop.taskUI.horiz_neg();
    });
    $('#bpd_mouth_offset #b3').hide();
    $('#bpd_mouth_offset #b2').show().bind('click.rfh', function (e) {
        assistive_teleop.taskUI.vert_neg();
    });
    $('#bpd_mouth_offset #b1').show().bind('click.rfh', function (e) {
        assistive_teleop.taskUI.depth_pos();
    });


}

