var head_pub;
var base_pub;
var scales={head:50,base:50,rarm:50,larm:50,rrot:0.25*Math.PI,lrot:0.25*Math.PI};
//var arm_joints={right:[],left:[]}
//var pointing_frame='openni_rgb_optical_frame'
var pointing_frame='head_mount_kinect_rgb_optical_frame'
var head_state;
var torso_state;
var plane = 'xy';

function teleop_init(){
    node.subscribe('/head_traj_controller_state_throttle',
            function(msg){
            window.head_state = msg.actual;
            window.head_joints = msg.joint_names;
            });
    node.subscribe('/r_gripper_controller_state_throttle',
            function(msg){
            $('#rg_slider').show().slider("option", "value", msg.process_value);
            });
    node.subscribe('/l_gripper_controller_state_throttle',
            function(msg){
            $('#lg_slider').show().slider("option", "value", msg.process_value);
            });
    node.subscribe('/torso_state_throttle',
            function(msg){
            window.torso_state = msg.actual.positions[0];
            $('#torso_slider').show().slider("option", "value", msg.actual.positions[0])
            });
    //node.subscribe('/r_arm_controller_state_throttle', function(msg){
    //                                    window.arm_joints.right = msg.actual});
    //node.subscribe('/l_arm_controller_state_throttle', function(msg){
    //                                   window.arm_joints.left = msg.actual});

    //Arm Publishers
    // var sides = ["right","left"];
    // for (var i=0; i < sides.length; i++){
    //    pubs['wt_'+sides[i]+'_arm_pose_commands'] = 'geometry_msgs/Point';
    //    pubs['wt_'+sides[i]+'_arm_angle_commands'] = 'trajectory_msgs/JointTrajectoryPoint';
    //    pubs['wt_lin_move_'+sides[i]] = 'std_msgs/Float32';
    //    pubs['wt_adjust_elbow_'+sides[i]] = 'std_msgs/Float32';
    // };
    advertise('head_nav_goal','geometry_msgs/PoseStamped');
    advertise('head_traj_controller/point_head_action/goal','pr2_controllers_msgs/PointHeadActionGoal');
    advertise('head_traj_controller/joint_trajectory_action/goal', 'pr2_controllers_msgs/JointTrajectoryActionGoal');
    advertise('base_controller/command','geometry_msgs/Twist');
    advertise('l_gripper_controller/command', 'pr2_controllers_msgs/Pr2GripperCommand');
    advertise('r_gripper_controller/command', 'pr2_controllers_msgs/Pr2GripperCommand');
};
$(function(){
        $('#scale_slider').slider({value:50,min:0,max:100,step:1,orientation:'vertical'}); 
        $('#rg_slider').slider({min:-0.005,max:0.09,step:0.001,orientation:'vertical'}); 
        $('#rg_slider').unbind("slidestop").bind("slidestop", function(event,ui){
            pub_gripper('r',$('#rg_slider').slider("value"));
            log('Opening/Closing Right Gripper');
            });	
        $('#lg_slider').slider({min:-0.005,max:0.09,step:0.001,orientation:'vertical'}); 
        $('#lg_slider').unbind("slidestop").bind("slidestop", function(event,ui){
            pub_gripper('l',$('#lg_slider').slider("value"));
            log("Opening/Closing Left Gripper");
            });	
        $('#torso_slider').slider({min:0.0,max:0.3,step:0.01,orientation:'vertical'});
        $('#torso_slider').unbind("slidestop").bind("slidestop",function(event,ui){
            pub_torso($('#torso_slider').slider("value"))
            });	
        $('#rg_grab').click(function(){gripper_grab('r')});
        $('#lg_grab').click(function(){gripper_grab('l')});
        $('#rg_release').click(function(){gripper_release('r')});
        $('#lg_release').click(function(){gripper_release('l')});
        });

function pub_gripper(arm,grpos) {
    node.publish(arm+'_gripper_controller/command',
            'pr2_controllers_msgs/Pr2GripperCommand',
            json({'position': grpos ,'max_effort':-1}));
};

function gripper_grab(arm){
    pub_gripper(arm,-0.005);
};

function gripper_release(arm){
    pub_gripper(arm,0.09);
};

function pub_head_traj(head_traj_goal, dist){ //Send pan/tilt trajectory commands to head
    if (head_traj_goal.goal.trajectory.points[0].positions[0] < -2.70) {
        head_traj_goal.goal.trajectory.points[0].positions[0] = -2.70};
    if (head_traj_goal.goal.trajectory.points[0].positions[0] > 2.70) {
        head_traj_goal.goal.trajectory.points[0].positions[0] = 2.70};
    if (head_traj_goal.goal.trajectory.points[0].positions[1] < -0.5) {
        head_traj_goal.goal.trajectory.points[0].positions[1] = -0.5};
    if (head_traj_goal.goal.trajectory.points[0].positions[1] > 1.4) {
        head_traj_goal.goal.trajectory.points[0].positions[1] = 1.4};
    head_traj_goal.goal.trajectory.joint_names = window.head_joints;
    head_traj_goal.goal.trajectory.points[0].velocities = [0, 0];
    head_traj_goal.goal.trajectory.points[0].time_from_start.secs = Math.max(2*dist, 1);
    head_traj_goal.goal.trajectory.points[0].time_from_start.nsecs = 0.0;
    node.publish('head_traj_controller/joint_trajectory_action/goal', 
            'pr2_controllers_msgs/JointTrajectoryActionGoal',
            json(head_traj_goal));
};

function pub_head_goal(x,y,z,frame) { //Send 3d point to look at using kinect
    node.publish('head_traj_controller/point_head_action/goal',
                 'pr2_controllers_msgs/PointHeadActionGoal',
                 json(
                     {'goal':{'target':{'header':{'frame_id':frame },
                     'point':{'x':x, 'y':y, 'z':z}},
                     'pointing_axis':{'x':0, 'y':0, 'z':1},
                     'pointing_frame':window.pointing_frame,
                     'min_duration':{'secs':0.0,'nsecs':0.0},
                     'max_velocity':0.5}}
                 )
    )
};

function teleop_head() {
    $('#bpd_default_rot, #cart_frame_select, #cart_frame_select_label, #cart_controller').hide();
    window.head_pub = window.clearInterval(head_pub);
    log('Controlling Head');
    $('#scale_slider').unbind("slidestop").bind("slidestop", function(event,ui){scales.head = $('#scale_slider').slider("value")});
    $('#scale_slider').show().slider("option", "value", scales.head);

    //	$("#tp").unbind().show();
    //	$("#tp").click(function(e){
    //		window.head_pub = window.clearInterval(head_pub);
    //		x = (e.pageX - Math.round(this.width/2) - this.x)/200;
    //		y = (e.pageY - Math.round(this.height/2) - this.y)/200;
    //   		head_traj_goal = JTAGoal;
    //		head_traj_goal.goal.trajectory.points[0] = window.head_state;
    //		head_traj_goal.goal.trajectory.points[0].positions[0] -= x;
    //		head_traj_goal.goal.trajectory.points[0].positions[1] += y;
    //                pub_head_traj(head_traj_goal, Math.sqrt(x*x+y*y));
    //	});

    $('#bpd_default').find(':button').unbind('.rfh').text('');
    $('#b9, #b7', '#bpd_default').hide(); 
    $('#b8, #b6, #b5, #b4, #b2', '#bpd_default').bind('click.rfh', function(e){
            window.head_pub = window.clearInterval(head_pub);
            });
    $('#bpd_default #b8').show().bind('click.rfh', function(e){//head up 
            head_traj_goal = JTAGoal;
            head_traj_goal.goal.trajectory.points[0] = window.head_state;
            head_traj_goal.goal.trajectory.points[0].positions[1] -= scales.head/100;
            pub_head_traj(head_traj_goal, scales.head/100);
            });
    $('#bpd_default #b6').show().bind('click.rfh', function(e){ //head right
            head_traj_goal = JTAGoal;
            head_traj_goal.goal.trajectory.points[0] = window.head_state;
            head_traj_goal.goal.trajectory.points[0].positions[0] -= scales.head/100;
            pub_head_traj(head_traj_goal, scales.head/100);
            });
    $('#bpd_default #b5').show().text("_|_").bind('click.rfh', function(e){ //center head to (0,0)
            pub_head_goal(0.8, 0.0, -0.25, '/base_footprint');
            });
    $('#bpd_default #b4').show().bind('click.rfh', function(e){ //head left
            head_traj_goal = JTAGoal;
            head_traj_goal.goal.trajectory.points[0] = window.head_state;
            head_traj_goal.goal.trajectory.points[0].positions[0] += scales.head/100;
            pub_head_traj(head_traj_goal, scales.head/100);
            });
    $('#bpd_default #b3').show().removeClass('arrow_rot_x_pos').text("Track Right Hand").bind('click.rfh', function(e){
            window.head_pub = window.clearInterval(head_pub);
            window.head_pub = window.setInterval("pub_head_goal(0,0,0,'r_gripper_tool_frame');",200);
            });	
    $('#bpd_default #b2').show().bind('click.rfh', function(e){ //head down
            head_traj_goal = JTAGoal;
            head_traj_goal.goal.trajectory.points[0] = window.head_state;
            head_traj_goal.goal.trajectory.points[0].positions[1] += scales.head/100;
            pub_head_traj(head_traj_goal, scales.head/100);
            });
    $('#bpd_default #b1').show().removeClass('arrow_rot_x_neg').text("Track Left Hand").bind('click.rfh', function(e){
            window.head_pub = window.clearInterval(head_pub);
            window.head_pub = window.setInterval("pub_head_goal(0,0,0,'l_gripper_tool_frame');",200);
            });
};

function base_pub_conf(selector,bx,by,bz){
    console.log("Checking State of "+selector);
    if ($(selector).hasClass('ui-state-active')){
        console.log("Found "+selector+" Active");
        bx_scaled = 0.002*scales.base*bx;
        by_scaled = 0.002*scales.base*by;
        bz_scaled = 0.006*scales.base*bz;
        node.publish('base_controller/command', 'geometry_msgs/Twist',
                '{"linear":{"x":'+bx_scaled+',"y":'+by_scaled+',"z":0},'+
                '"angular":{"x":0,"y":0,"z":'+bz_scaled+'}}');
        setTimeout(function(){base_pub_conf(selector,bx,by,bz)}, 100);
    } else {
        console.log('End driving pub for '+selector);
    };
};

function teleop_base() {
    log("Controlling Base");
    $('#bpd_default_rot, #cart_frame_select, #cart_frame_select_label, #cart_controller').hide();
    $('#scale_slider').unbind("slidestop").bind("slidestop", function(event,ui){scales.base = $('#scale_slider').slider("value")});
    $('#scale_slider').show().slider("option", "value", scales.base);
    //$("#tp").unbind().hide();
    $('#bpd_default').find(':button').unbind('.rfh').text('');
    $('#b9, #b7, #b5','#bpd_default').hide()

        $('#bpd_default #b8').show().bind('mousedown.rfh', function(e){
                base_pub_conf("#bpd_default #"+e.target.id, 1,0,0);
                });
    $('#bpd_default #b6').show().bind('mousedown.rfh', function(e){
            base_pub_conf("#bpd_default #"+e.target.id, 0,-1,0);
            });
    $('#bpd_default #b4').show().bind('mousedown.rfh', function(e){
            base_pub_conf("#bpd_default #"+e.target.id, 0,1,0);
            });
    $('#bpd_default #b3').show().addClass('arrow_rot_x_pos').bind('mousedown.rfh', function(e){
            base_pub_conf("#bpd_default #"+e.target.id, 0,0,-1);
            });
    $('#bpd_default #b2').show().bind('mousedown.rfh', function(e){
            base_pub_conf("#bpd_default #"+e.target.id, -1,0,0);
            });
    $('#bpd_default #b1').show().addClass('arrow_rot_x_neg').bind('mousedown.rfh', function(e){
            base_pub_conf("#bpd_default #"+e.target.id, 0,0,1);
            });
};
