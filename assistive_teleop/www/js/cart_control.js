var CartControl = function (options) {
    var cCon = this;
    cCon.side = options.side;
    cCon.ros = options.ros;
    cCon.endLink = options.endLink;
    cCon.ctrlParams = options.ctrlParams;
    cCon.ctrlName = options.ctrlName;
    cCon.enabled = false;
    cCon.trans_scale = 0.5;
    cCon.rot_scale = Math.PI/4;
    cCon.ros.getMsgDetails('geometry_msgs/TwistStamped');
    
    cCon.enabledSub = new cCon.ros.Topic({
        name: 'face_adls/'+cCon.side[0]+'_cart_ctrl_enabled',
        messageType: 'std_msgs/Bool'});
    cCon.enabledSubCBList = [];
    cCon.enabledSubCB = function (msg) {
        cCon.enabled = msg.data;
    };
    cCon.enabledSubCBList.push(cCon.enabledSubCB);
    cCon.enabledSub.subscribe(function (msg) {
        for (var i=0; i<cCon.enabledSubCBList.length; i += 1) {
            cCon.enabledSubCBList[i](msg);
        };
    });

    cCon.cmdPub = new cCon.ros.Topic({
        name: cCon.side[0]+'_cart/web_commands',
        messageType: 'geometry_msgs/TwistStamped'});
    cCon.cmdPub.advertise();
    cCon.sendGoal = function (values) {
        var ts = new cCon.ros.composeMsg('geometry_msgs/TwistStamped');
        ts.header.frame_id = values.frame || 'torso_lift_link';
        ts.twist.linear.x = values.linear_x || 0;
        ts.twist.linear.y = values.linear_y || 0;
        ts.twist.linear.z = values.linear_z || 0;
        ts.twist.angular.x = values.angular_x || 0;
        ts.twist.angular.y = values.angular_y || 0;
        ts.twist.angular.z = values.angular_z || 0;
        var tsMsg = new cCon.ros.Message(ts);
        cCon.cmdPub.publish(tsMsg);
    };

    cCon.enableServiceClient = new cCon.ros.Service({
        name: '/face_adls/'+cCon.side[0]+'_enable_cart_ctrl',
        serviceType: 'hrl_face_adls/EnableCartController'});

    cCon.enable = function (state) {
        var enableCartCtrlReq = {};
        enableCartCtrlReq.enable = state; 
        enableCartCtrlReq.end_link = cCon.endLink;
        enableCartCtrlReq.ctrl_params = cCon.ctrlParams;
        enableCartCtrlReq.ctrl_name = cCon.ctrlName;
        enableCartCtrlReq.frame_rot = {x:0, y:0, z:0};
        enableCartCtrlReq.velocity = 0.02;
        var req = new cCon.ros.ServiceRequest(enableCartCtrlReq);
        cCon.enableServiceClient.callService(req, function (resp) {});
    }
};

var initCartControl = function () {
    window.cartControl = [new CartControl({side: 'left',
                                           ros: window.ros,
                                           endLink: "l_gripper_shaver45_frame",
                                           ctrlParams: "$(find hrl_face_adls)/params/l_jt_task_shaver45.yaml",
                                           ctrlName: "l_cart_jt_task_shaver"}),
                          new CartControl({side:'right', 
                                           ros: window.ros,
                                           endLink: "r_gripper_tool_frame",
                                           ctrlParams: "$(find hrl_face_adls)/params/r_jt_task_tool.yaml",
                                           ctrlName: "r_cart_jt_task_tool"})];
                          
    $('#cont_l_arm').bind('click.rfh', function () {window.cartControl[0].enable(true)});
    $('#cont_r_arm').bind('click.rfh', function () {window.cartControl[1].enable(true)});
    $('#default_rot_slider').slider({value : Math.PI/4,
                                     min : 0,
                                     max : Math.PI/2,
                                     step : Math.PI/50,
                                     orientation:'vertical'}); 
    $("#cart_frame_select, #cart_controller, #cart_cont_state_check").hide();
   
    var lCartStateCB = function (msg) {
        console.log("Received L Cart Controller State: "+msg.data.toString());
        if (msg.data){
            showArmControls(window.cartControl[0]);
            $('#cont_l_arm').attr('checked','checked').button('refresh');
        } else {
            $('#cont_l_arm').attr('checked','').button('refresh');
        };
    };
    window.cartControl[0].enabledSubCBList.push(lCartStateCB)

    var rCartStateCB = function (msg) {
        console.log("Received R Cart State: "+msg.data.toString());
        if (msg.data){
            showArmControls(window.cartControl[1]);
            $('#cont_r_arm').attr('checked','checked').button('refresh');
        } else {
            $('#cont_r_arm').attr('checked','').button('refresh');
        };
    };
    window.cartControl[1].enabledSubCBList.push(rCartStateCB)


    var enableCartControl = function (arm) {
        $('#bpd_default :button, #bpd_default_rot :button, #scale_slider, #default_rot_slider').hide();
        log("Requesting "+arm+" arm Cartesian controller.  Controls will appear when controller is active.");
    };

var showArmControls = function (contObj) {
        $('#bpd_default_rot, #cart_frame_select, #cart_frame_select_label, #cart_controller, #cart_cont_state_check').show();
        $('#frame_opt_hand').val('/'+contObj.side[0]+'_wrist_roll_link');

        $('#scale_slider').unbind("slidestop").bind("slidestop", function (event,ui) {
            contObj.trans_scale = $('#scale_slider').slider("value");
        });
        $('#scale_slider').slider("option", "value", contObj.trans_scale).show();
        $('#default_rot_slider').unbind("slidestop").bind("slidestop", function (event,ui) {
            contObj.rot_scale = $('#default_rot_slider').slider("value")
        });
        $('#default_rot_slider').slider("option", "value", contObj.rot_scale).show();

        $('#bpd_default :button').unbind('.rfh');
        $('#bpd_default #b9').show().bind('click.rfh', function (e) {
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              linear_x:contObj.trans_scale/4});
            });
        $('#bpd_default #b8').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              linear_z:contObj.trans_scale/4});
            });
        $('#bpd_default #b7').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              linear_x:-contObj.trans_scale/4});
            });
        $('#bpd_default #b6').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              linear_y:-contObj.trans_scale/4});
            });
        $('#bpd_default #b5').hide()
        $('#bpd_default #b4').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                          linear_y:contObj.trans_scale/4});
            });
        $('#bpd_default #b3').hide();
        $('#bpd_default #b2').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              linear_z:-contObj.trans_scale/4});
            });
        $('#bpd_default #b1').hide();

        $('#bpd_default_rot :button').unbind('.rfh');
        $('#bpd_default_rot #b9').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              angular_x:contObj.rot_scale});
            });
        $('#bpd_default_rot #b8').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              angular_y:-contObj.rot_scale});
            });
        $('#bpd_default_rot #b7').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              angular_x:-contObj.rot_scale});
            });
        $('#bpd_default_rot #b6').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              angular_z:-contObj.rot_scale});
            });
        $('#bpd_default_rot #b5').hide()
        $('#bpd_default_rot #b4').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              angular_z:contObj.rot_scale});
            });
        $('#bpd_default_rot #b3').hide();
        $('#bpd_default_rot #b2').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              angular_y:contObj.rot_scale});
            });
        $('#bpd_default_rot #b1').hide();

    };
};
