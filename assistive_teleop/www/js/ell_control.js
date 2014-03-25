var PoweredTool = function (ros) {
    'use strict';
    var tool = this; 
    tool.ros = ros;
    tool.state = false;
    tool.togglePub = new tool.ros.Topic({
        name: 'ros_switch',
        messageType: 'std_msgs/Bool'});
    tool.togglePub.advertise();
    tool.toggle = function () {
        var cmd = !(tool.state);
        var msg = new tool.ros.Message({data:cmd});
        tool.togglePub.publish(msg);
    };
    tool.stateSub = new tool.ros.Topic({
        name: 'ros_switch_state',
        messageType: 'std_msgs/Bool'});
    tool.stateSub.subscribe(function (msg) {
        tool.state = msg.data;
    });
};

var EllipsoidControl = function (ros) {
    'use strict';
    var ellCon = this;
    ellCon.ros = ros;
    
    ellCon.globalPoses = [];
    ellCon.globalPosesSub = new ellCon.ros.Topic({
        name: 'face_adls/global_move_poses',
        messageType: 'hrl_face_adls/StringArray'});
    ellCon.globalPosesSub.subscribe(function (msg) {
        ellCon.globalPoses = [];
        for (var i=0; i<msg.data.length; i += 1) {
            ellCon.globalPoses.push(msg.data[i]);
        }
    });
   
    ellCon.selectedPoseSub = new ellCon.ros.Topic({
        name:'sm_selected_pose',
        messageType:'std_msgs/String'});
    ellCon.selectedPoseSub.subscribe(function (msg) {
        ellCon.selectedPose = msg.data;
    }); 

    ellCon.enabledSub = new ellCon.ros.Topic({
        name: 'face_adls/controller_enabled',
        messageType: 'std_msgs/Bool'});
    //TODO: Separate the interface from the subscriber
    ellCon.enabledSub.subscribe(function (msg) {
        console.log("Ell Controller State Updated to "+msg.data);
        if (msg.data) {
            console.log("Ellipsoid Controller Active")
            $("#ell_controller").attr("checked","checked").button('refresh');
            $('#shave_list').empty();
            for (var i=0; i <window.ellControl.globalPoses.length; i += 1) {
                $('#shave_list').append('<option val="'+window.ellControl.globalPoses[i]+
                                        '">'+ window.ellControl.globalPoses[i]+'</option>');
            };
            $(".ell_control").show();
           } else {
            $("#ell_controller").attr("checked", "").button('refresh');
            console.log("Ellipsoid Controller Inactive")
            $(".ell_control").hide();
           };
    });

    ellCon.globalMovePub = new ellCon.ros.Topic({
        name:'face_adls/global_move',
        messageType: 'std_msgs/String'});
    ellCon.globalMovePub.advertise();
    ellCon.sendGlobalMove = function (key) {
        ellCon.globalMovePub.publish({data:key});
    };

    ellCon.clickedMovePub = new ellCon.ros.Topic({
        name:'face_adls/clicked_move',
        messageType: 'geometry_msgs/PoseStamped'});
    ellCon.clickedMovePub.advertise();
    ellCon.sendClickedMove = function (ps) {
        var msg = new ellCon.ros.Message(ps);
        ellCon.clickedMovePub.publish(msg);
    };

    ellCon.localMovePub = new ellCon.ros.Topic({
        name:'face_adls/local_move',
        messageType: 'std_msgs/String'});
    ellCon.localMovePub.advertise();
    ellCon.sendLocalMove = function (cmd) {
        ellCon.localMovePub.publish({data:cmd});
    };

    ellCon.stopPub = new ellCon.ros.Topic({
        name:'face_adls/stop_move',
        messageType:'std_msgs/Bool'});
    ellCon.stopPub.advertise();
    ellCon.stopMove = function () {
        ellCon.stopPub.publish({data:true});
    };

    ellCon.controllerServiceClient = new ellCon.ros.Service({
        name: '/face_adls/enable_controller',
        serviceType: 'hrl_face_adls/EnableFaceController'});

    ellCon.toggle = function (state) {
    if (typeof state == 'undefined'){  
        state = $("#ell_controller").attr('checked');
    };
    var mode = $('#ell_mode_sel option:selected').val();
    console.log("Sending controller :"+state.toString());
    var req = new ellCon.ros.ServiceRequest({enable:state, mode:mode});
    ellCon.controllerServiceClient.callService(req, function(ret){
            console.log("Switching Ell. Controller Service Returned "+ret.success);
            if (ret.success === false) {
                $("#ell_controller").attr("checked", "").button('refresh');
                console.log("Ellipsoid Controller Inactive")
                $(".ell_control").hide();
            }
        });
    };
};

var MirrorPointer = function (ros) {
    'use strict';
    var mirror = this;
    mirror.ros = ros;
    mirror.pointServiceClient = new mirror.ros.Service({
        name: '/point_mirror',
        serviceType: 'std_srvs/Empty'});
    mirror.point = function () {
        console.log("Pointing Mirror");
        //enable_cart_control('right');
        setTimeout(function () {
           mirror.pointServiceClient.callService({},function () {});
           }, 1000);
    };
}

var initEllControl = function () {
    window.shaver = new PoweredTool(window.ros);
    window.ellControl = new EllipsoidControl(window.ros);
    window.mirrorPointer = new MirrorPointer(window.ros);

    $("#tabs").on("tabsbeforeactivate", function (event, ui) {
      if (ui.newPanel.selector === "#tab_ellipse") {
        window.mjpeg.setCamera('head_registration/confirmation');
      }
    });

    $('#ell_controller').click(function () {window.ellControl.toggle()});
    $('#adj_mirror').click(window.mirrorPointer.point);
    $('#tool_power').click(window.shaver.toggle);
    $('#send_shave_select').click(function () {
        console.log('Sending Command to move to '+$('#shave_list option:selected').text())
        window.ellControl.sendGlobalMove($('#shave_list option:selected').text());
    });
    $('#shave_stop').click(window.ellControl.stopMove);
   
    $('#bpd_ell_trans #b2').click(function () {window.ellControl.sendLocalMove('translate_down')});
    $('#bpd_ell_trans #b4').click(function () {window.ellControl.sendLocalMove('translate_left')});
    $('#bpd_ell_trans #b6').click(function () {window.ellControl.sendLocalMove('translate_right')});
    $('#bpd_ell_trans #b7').click(function () {window.ellControl.sendLocalMove('translate_in')});
    $('#bpd_ell_trans #b8').click(function () {window.ellControl.sendLocalMove('translate_up')});
    $('#bpd_ell_trans #b9').click(function () {window.ellControl.sendLocalMove('translate_out')});
   
    $('#bpd_ell_rot #b1').click(function () {window.ellControl.sendLocalMove('reset_rotation')});
    $('#bpd_ell_rot #b2').click(function () {window.ellControl.sendLocalMove('rotate_y_neg')});
    $('#bpd_ell_rot #b4').click(function () {window.ellControl.sendLocalMove('rotate_z_pos')});
    $('#bpd_ell_rot #b6').click(function () {window.ellControl.sendLocalMove('rotate_z_neg')});
    $('#bpd_ell_rot #b7').click(function () {window.ellControl.sendLocalMove('rotate_x_pos')});
    $('#bpd_ell_rot #b8').click(function () {window.ellControl.sendLocalMove('rotate_y_pos')});
    $('#bpd_ell_rot #b9').click(function () {window.ellControl.sendLocalMove('rotate_x_neg')});

    $(".ell_control").hide();
};
