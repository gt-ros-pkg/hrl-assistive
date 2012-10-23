var CartControl = function (side, ros) {
    var cCon = this;
    cCon.side = side;
    cCon.ros = ros;
    cCon.enabled = false;
    cCon.scale = 0.5;
    cCon.ros.getMsgDetails('geometry_msgs/TwistStamped');
    
    cCon.enabledSub = new cCon.ros.Topic({
        name: 'face_adls/'+side[0]+'_cart_ctrl_enabled',
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

    cCon.cmdPub = new cCon.ros.Topic({
        name: side[0]+'_cart/web_commands',
        messageType: 'geometry_msgs/TwistStamped'});
    cCon.cmdPub.advertise();
    cCon.sendGoal = function (values) {
        var ts = new cCon.ros.composeMsg('geometry_msgs/TwistStamped');
        ts.header.frame_id = values.frame || 'torso_lift_link';
        ts.linear.x = values.linear_x || 0;
        ts.linear.y = values.linear_y || 0;
        ts.linear.z = values.linear_z || 0;
        ts.angular.x = values.angular_x || 0;
        ts.angular.y = values.angular_y || 0;
        ts.angular.z = values.angular_z || 0;
        var tsMsg = new cCon.ros.Message(ts);
        cCon.cmdPub.publish(tsMsg);
    };

    cCon.enableServiceClient = new cCon.ros.Service({
        name: '',
        requestType: ''});
    cCon.enable = function (state) {
        var req = '';
        cCon.enableServiceClient.callService(req);
    }


};
function cart_init(){
 //   advertise('r_cart/web_commands','geometry_msgs/TwistStamped');
  //  advertise('l_cart/web_commands','geometry_msgs/TwistStamped');
    window.EnableCartControlReq = {enable:true, 
        end_link:'',
        ctrl_params:'',
        ctrl_name:'',
        frame_rot:{x:0, y:0, z:0},
        velocity:0};
};

var initCartControl = function() {
    window.cartControl = [new CartControl('left', window.ros),
                          new CartControl('right', window.ros)]
    $('#cont_l_arm').bind('click.rfh', window.cartControl[0].enable);
    $('#cont_r_arm').bind('click.rfh', window.cartControl[1].enable);
    $('#default_rot_slider').slider({value:0.25*Math.PI,min:0,max:0.5*Math.PI,step:0.02*Math.PI,orientation:'vertical'}); 
    $("#cart_frame_select, #cart_controller, #cart_cont_state_check").hide();
    //$('#default_rot_slider').bind("slidestop", function(event,ui){
    //    scales["rarm"] = $('#scale_slider').slider("value");
    //    });
    //$('#scale_slider').slider("option", "value", scales['rarm']).show();
   
    var lCartStateCB = function (msg) {
        console.log("Received L Cart Controller State: "+msg.data.toString());
        if (msg){
            show_arm_controls('left');
            $('#cont_l_arm').attr('checked','checked').button('refresh');
        } else {
            $('#cont_l_arm').attr('checked','').button('refresh');
        };
    };
    window.cartControl[0].enabledSubCBList.push(lCartStateCB)

    var rCartStateCB = function (msg) {
        console.log("Received R Cart State: "+msg.data.toString());
        if (msg){
            show_arm_controls('right')
                $('#cont_r_arm').attr('checked','checked').button('refresh');
        } else {
            $('#cont_r_arm').attr('checked','').button('refresh');
        };
    };
    window.cartControl[1].enabledSubCBList.push(rCartStateCB)



    tws.header.frame_id = 



    var enableCartControl = function (arm) {
        $('#bpd_default :button, #bpd_default_rot :button, #scale_slider, #default_rot_slider').hide();
        var ecc = window.EnableCartControlReq;
        var service = '';
        if (arm =='right'){
            ecc.end_link = "r_gripper_tool_frame";
            ecc.ctrl_params = "$(find hrl_face_adls)/params/r_jt_task_tool.yaml";
            ecc.ctrl_name = "r_cart_jt_task_tool";
            service = '/face_adls/r_enable_cart_ctrl'
        }else {
            ecc.end_link = "l_gripper_shaver45_frame";
            ecc.ctrl_params = "$(find hrl_face_adls)/params/l_jt_task_shaver45.yaml";
            ecc.ctrl_name = "l_cart_jt_task_shaver";
            service = '/face_adls/l_enable_cart_ctrl'
        };
        ecc.frame_rot = {x:0.0, y:0.0, z:0.0};
        ecc.velocity = 0.02;
        node.rosjs.callService(service, [json(ecc)],function(ret){
                console.log("Enable cart controller returned success: "+ret.success)
                })
        log("Requesting "+arm+" arm Cartesian controller.  Controls will appear when controller is active.");
    };

var showArmControls = function (contObj) {
        $('#bpd_default_rot, #cart_frame_select, #cart_frame_select_label,'+
          '#cart_controller, #cart_cont_state_check').show();
        $('#scale_slider').unbind("slidestop").bind("slidestop", function(event,ui){scales[arm[0]+"arm"] = $('#scale_slider').slider("value")});
        $('#scale_slider').slider("option", "value", scales[arm[0]+'arm']).show();
        $('#default_rot_slider').unbind("slidestop").bind("slidestop", function(event,ui){scales[arm[0]+"rot"] = $('#default_rot_slider').slider("value")});
        $('#default_rot_slider').slider("option", "value", scales[arm[0]+'rot']).show();
        $('#frame_opt_hand').val('/'+arm[0]+'_wrist_roll_link');

        $('#bpd_default :button').unbind('.rfh');
        $('#bpd_default #b9').show().bind('click.rfh', function (e) {
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              linear_x:contObj.scale/4});
        $('#bpd_default #b8').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              linear_z:contObj.scale/4});
            });
        $('#bpd_default #b7').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              linear_x:-contObj.scale/4});
            });
        $('#bpd_default #b6').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              linear_y:-contObj.scale/4});
            });
        $('#bpd_default #b5').hide()
        $('#bpd_default #b4').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                          linear_y:contObj.scale/4});
            });
        $('#bpd_default #b3').hide();
        $('#bpd_default #b2').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              linear_z:-contObj.scale/4});
            });
        $('#bpd_default #b1').hide();

        $('#bpd_default_rot :button').unbind('.rfh');
        $('#bpd_default_rot #b9').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              angular_x:contObj.scale/4});
                pub_cart_twist(arm,[0,0,0],[$('#default_rot_slider').slider('value'),0,0]);
            });
        $('#bpd_default_rot #b8').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              angular_y:-contObj.scale/4});
                });
        $('#bpd_default_rot #b7').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              angular_x:-contObj.scale/4});
                });
        $('#bpd_default_rot #b6').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              angular_z:-contObj.scale/4});
                });
        $('#bpd_default_rot #b5').hide()
        $('#bpd_default_rot #b4').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              angular_z:contObj.scale/4});
            });
        $('#bpd_default_rot #b3').hide();
        $('#bpd_default_rot #b2').show().bind('click.rfh', function(e){
            contObj.sendGoal({frame:$('#cart_frame_select').val(),
                              angular_y:contObj.scale/4});
            });
        $('#bpd_default_rot #b1').hide();
    };

});    


