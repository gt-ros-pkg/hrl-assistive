function cart_init(){
    advertise('r_cart/web_commands','geometry_msgs/TwistStamped');
    advertise('l_cart/web_commands','geometry_msgs/TwistStamped');
    window.TwistStamped = {header:{seq:0,stamp:{secs:0,nsecs:0},frame_id:""},
                               twist:{linear:{x:0,y:0,z:0}, angular:{x:0,y:0,z:0}}};
    window.EnableCartControlReq = {enable:true, 
                                    end_link:'',
                                    ctrl_params:'',
                                    ctrl_name:'',
                                    frame_rot:{x:0, y:0, z:0},
                                    velocity:0};
    node.subscribe('/face_adls/l_cart_ctrl_enabled',
                function(msg){l_cart_state_cb(msg)})
    node.subscribe('/face_adls/r_cart_ctrl_enabled',
                function(msg){r_cart_state_cb(msg)})
    };

$(function(){
    $('#cont_r_arm').bind('click.rfh', function(){enable_cart_control('right')});
    $('#cont_l_arm').bind('click.rfh', function(){enable_cart_control('left')});
	$('#default_rot_slider').slider({value:0.25*Math.PI,min:0,max:0.5*Math.PI,step:0.02*Math.PI,orientation:'vertical'}); 
    $("#cart_frame_select, #cart_controller, #cart_cont_state_check").hide();
	$('#default_rot_slider').bind("slidestop", function(event,ui){
                                scales["rarm"] = $('#scale_slider').slider("value");
                                });
	$('#scale_slider').slider("option", "value", scales['rarm']).show();
    });    

function l_cart_state_cb(msg){
    console.log("Received L Cart State: "+msg.data.toString());
    if (msg){
        show_arm_controls('left');
        $('#cont_l_arm').attr('checked','checked').button('refresh');
    } else {
        $('#cont_l_arm').attr('checked','').button('refresh');
    };
};

function r_cart_state_cb(msg){
    console.log("Received R Cart State: "+msg.data.toString());
    if (msg){
        show_arm_controls('right')
        $('#cont_r_arm').attr('checked','checked').button('refresh');
    } else {
        $('#cont_r_arm').attr('checked','').button('refresh');
    };
};

function enable_cart_control(arm){
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

function pub_cart_twist(arm, trans, rot){
    tws = window.TwistStamped;
    tws.header.frame_id = $('#cart_frame_select').val()
    tws.twist.linear.x = trans[0];
    tws.twist.linear.y = trans[1];
    tws.twist.linear.z = trans[2];
    tws.twist.angular.x = rot[0];
    tws.twist.angular.y = rot[1];
    tws.twist.angular.z = rot[2];
    node.publish(arm[0]+'_cart/web_commands','geometry_msgs/TwistStamped', json(tws));
};

function show_arm_controls(arm){
    $('#bpd_default_rot, #cart_frame_select, #cart_frame_select_label, #cart_controller, #cart_cont_state_check').show();
	$('#scale_slider').unbind("slidestop").bind("slidestop", function(event,ui){scales[arm[0]+"arm"] = $('#scale_slider').slider("value")});
	$('#scale_slider').slider("option", "value", scales[arm[0]+'arm']).show();
	$('#default_rot_slider').unbind("slidestop").bind("slidestop", function(event,ui){scales[arm[0]+"rot"] = $('#default_rot_slider').slider("value")});
	$('#default_rot_slider').slider("option", "value", scales[arm[0]+'rot']).show();
    $('#frame_opt_hand').val('/'+arm[0]+'_wrist_roll_link');

    $('#bpd_default').find(':button').unbind('.rfh');
    $('#bpd_default #b9').show().bind('click.rfh', function(e){
        pub_cart_twist(arm,[scales[arm[0]+'arm']/400,0,0],[0,0,0]);
    });
    $('#bpd_default #b8').show().bind('click.rfh', function(e){
        pub_cart_twist(arm,[0,0,scales[arm[0]+'arm']/400],[0,0,0]);
    });
    $('#bpd_default #b7').show().bind('click.rfh', function(e){
        pub_cart_twist(arm,[-scales[arm[0]+'arm']/400,0,0],[0,0,0]);
    });
    $('#bpd_default #b6').show().bind('click.rfh', function(e){
        pub_cart_twist(arm,[0,-scales[arm[0]+'arm']/400,0],[0,0,0]);
    });
    $('#bpd_default #b5').hide()
    $('#bpd_default #b4').show().bind('click.rfh', function(e){
        pub_cart_twist(arm,[0,scales[arm[0]+'arm']/400,0],[0,0,0]);
    });
    $('#bpd_default #b3').hide();
    $('#bpd_default #b2').show().bind('click.rfh', function(e){
        pub_cart_twist(arm,[0,0,-scales[arm[0]+'arm']/400],[0,0,0]);
    });
    $('#bpd_default #b1').hide();
    
    $('#bpd_default_rot').find(':button').unbind('.rfh');
    $('#bpd_default_rot #b9').show().bind('click.rfh', function(e){
        pub_cart_twist(arm,[0,0,0],[$('#default_rot_slider').slider('value'),0,0]);
    });
    $('#bpd_default_rot #b8').show().bind('click.rfh', function(e){
        pub_cart_twist(arm,[0,0,0],[0,-$('#default_rot_slider').slider('value'),0]);
    });
    $('#bpd_default_rot #b7').show().bind('click.rfh', function(e){
        pub_cart_twist(arm,[0,0,0],[-$('#default_rot_slider').slider('value'),0,0]);
    });
    $('#bpd_default_rot #b6').show().bind('click.rfh', function(e){
        pub_cart_twist(arm,[0,0,0],[0,0,-$('#default_rot_slider').slider('value')]);
    });
    $('#bpd_default_rot #b5').hide()
    $('#bpd_default_rot #b4').show().bind('click.rfh', function(e){
        pub_cart_twist(arm,[0,0,0],[0,0,$('#default_rot_slider').slider('value')]);
    });
    $('#bpd_default_rot #b3').hide();
    $('#bpd_default_rot #b2').show().bind('click.rfh', function(e){
        pub_cart_twist(arm,[0,0,0],[0,$('#default_rot_slider').slider('value'),0]);
    });
    $('#bpd_default_rot #b1').hide();
};
