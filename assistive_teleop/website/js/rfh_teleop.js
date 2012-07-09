var head_pub;
var base_pub;
var scales={head:50,base:50,rarm:50,larm:50,rrot:0.25*Math.PI,lrot:0.25*Math.PI};
//var arm_joints={right:[],left:[]}
var pointing_frame='openni_rgb_optical_frame'
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

function pub_torso(tz){
    var dir = (tz < window.torso_state) ? 'Lowering' : 'Raising';
    log(dir+" Torso");
	node.publish('torso_controller/position_joint_action/goal',
                 'pr2_controllers_msgs/SingleJointPositionActionGoal',
                 json({'goal':{'position':tz}})
                )
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
  		node.publish('head_traj_controller/joint_trajectory_action/goal', 
                     'pr2_controllers_msgs/JointTrajectoryActionGoal',
                     json(head_traj_goal));
};

function pub_head_goal(x,y,z,frame) { //Send 3d point to look at using kinect
	node.publish('head_traj_controller/point_head_action/goal', 'pr2_controllers_msgs/PointHeadActionGoal', json(
		{  'goal':{'target':{'header':{'frame_id':frame },
				             'point':{'x':x, 'y':y, 'z':z}},
                   'pointing_axis':{'x':0, 'y':0, 'z':1},
                   'pointing_frame':window.pointing_frame,
                   'max_velocity':0.5}}))
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
//function pub_lin_move(){
//    node.publish('wt_lin_move_'+window.arm(),'std_msgs/Float32', json({"data":window.lin_move}));
//    log('Sending '+window.arm().toUpperCase()+' Hand Advance/Retreat command');
//};
//
//function control_arm(x,y,z){
//	goal = window.gm_point;
//    goal.x = x;
//    goal.y = y;
//    goal.z = z;
//    log('Sending command to '+window.arm().toUpperCase()+' arm');
//    node.publish('wt_'+window.arm()+'_arm_pose_commands',
//                 'geometry_msgs/Point', json(goal));
//};

//function teleop_arm() {
//	var x,y,z,b9txt,b7txt;
//	if (plane == 'xy'){
//        x=0;
//        y=1;
//        z=2;
//        b9txt='Up';
//        b7txt='Down'
//    } else if (plane == 'yz') {
//        x=2;
//        y=1;
//        z=0;
//        b9txt='Further Away';
//        b7txt='Closer'
//    };
//
//    log('Controlling '+window.arm().toUpperCase()+' Arm');
//    $('#scale_slider').unbind("slidestop").bind("slidestop", function(event,ui){scales[window.arm()[0]+'arm'] = $('#scale_slider').slider("value")});
//    $('#scale_slider').show().slider("option", "value", scales[window.arm()[0]+'arm']);
//
//   // $("#tp").unbind().show();
//   // $("#tp").click(function(e){
//   //     y = -(e.pageX - Math.round(this.width/2) - this.x)/500;
//   //     x = -(e.pageY - Math.round(this.height/2) - this.y)/500;
//   //     control_arm(x,y,0);	
//   // });
//
//    $('#bpd_default').find(':button').unbind('.rfh');
//    $('#bpd_default #b9').show().text(b9txt).bind('click.rfh', function(e){
//        control_arm(0,0,scales[window.arm()[0]+'arm']/500);
//    });
//    $('#bpd_default #b8').show().text("^").bind('click.rfh', function(e){
//        control_arm(scales[window.arm()[0]+'arm']/500,0,0);
//    });
//    $('#bpd_default #b7').show().text(b7txt).bind('click.rfh', function(e){
//        control_arm(0,0,-scales[window.arm()[0]+'arm']/500);
//    });
//    $('#bpd_default #b6').show().text(">").bind('click.rfh', function(e){
//        control_arm(0,-scales[window.arm()[0]+'arm']/500,0);
//    });
//    $('#bpd_default #b5').hide()
//    $('#bpd_default #b4').show().text("<").bind('click.rfh', function(e){
//        control_arm(0,scales[window.arm()[0]+'arm']/500,0);
//    });
//    $('#bpd_default #b3').show().text("Advance").bind('click.rfh', function(e){
//        window.lin_move=0.1*(scales[window.arm()[0]+'arm']/100);
//        pub_lin_move();
//    });
//    $('#bpd_default #b2').show().text("v").bind('click.rfh', function(e){
//        control_arm(-scales[window.arm()[0]+'arm']/500,0,0);
//    });
//    $('#bpd_default #b1').show().text("Retreat").bind('click.rfh', function(e){
//        window.lin_move=-0.1*(scales[window.arm()[0]+'arm']/100);
//        pub_lin_move();
//    });
//};

//function pub_elbow(dir) {
//    dir =  (window.arm()=='right') ? -dir : dir; //Catch reflection
//    var action = (dir == 1) ? 'Raise ' : 'Lower ';
//    node.publish('wt_adjust_elbow_'+window.arm(),'std_msgs/Float32', json({"data":dir}));
//    log('Sending command to ' + action + window.arm().toUpperCase() + ' elbow')
//};
//
//function pub_arm_joints(angles){
//    node.publish('wt_'+window.arm()+'_arm_angle_commands', 'trajectory_msgs/JointTrajectoryPoint', json(angles))
//};

//function teleop_wrist() {
//    log('Controlling '+window.arm().toUpperCase()+' Hand');
//    $('#scale_slider').unbind("slidestop").bind("slidestop", function(event,ui){scales[window.arm()[0]+'wrist'] = $('#scale_slider').slider("value")});
//    $('#scale_slider').show().slider("option", "value", scales[window.arm()[0]+'wrist']);
//
//   // $("#tp").unbind().show();
//   // $("#tp").click(function(e){
//   //     x = (e.pageX - Math.round(this.width/2) - this.x);
//   //     y = (e.pageY - Math.round(this.height/2) - this.y);
//   //     var joint_goals = arm_joints[window.arm()];
//   //     joint_goals.positions[4] += x*0.0107;
//   //     joint_goals.positions[5] -= y*(Math.PI/200);
//   //     pub_arm_joints(joint_goals)
//   // });
//
//    $('#bpd_default').find(':button').unbind('.rfh');
//    $('#bpd_default #b9').show().text("Hand Roll Right").bind('click.rfh', function(e){
//        joint_goals = arm_joints[window.arm()];
//        joint_goals.positions[6] += scales[window.arm()[0]+'wrist']*(Math.PI/200);
//        pub_arm_joints(joint_goals)
//    });
//    $('#bpd_default #b8').show().text("Wrist Flex Out").bind('click.rfh', function(e){
//        joint_goals = arm_joints[window.arm()];
//        joint_goals.positions[5] += scales[window.arm()[0]+'wrist']*0.0107;
//        pub_arm_joints(joint_goals)
//    });
//    $('#bpd_default #b7').show().text("Hand Roll Left").bind('click.rfh', function(e){
//        joint_goals = arm_joints[window.arm()];
//        joint_goals.positions[6] -= scales[window.arm()[0]+'wrist']*(Math.PI/200);
//        pub_arm_joints(joint_goals)
//    });
//    $('#bpd_default #b6').show().text("Arm Roll Right").bind('click.rfh', function(e){
//        joint_goals = arm_joints[window.arm()];
//        joint_goals.positions[4] += scales[window.arm()[0]+'wrist']*(Math.PI/200);
//        pub_arm_joints(joint_goals)
//    });
//    $('#bpd_default #b5').hide()
//    $('#bpd_default #b4').show().text("Arm Roll Left").bind('click.rfh', function(e){
//        joint_goals = arm_joints[window.arm()];
//        joint_goals.positions[4] -= scales[window.arm()[0]+'wrist']*0.0107;
//        pub_arm_joints(joint_goals)
//    });
//    $('#bpd_default #b3').show().text("Raise Elbow").bind('click.rfh', function(e){
//        pub_elbow(0.01*scales[window.arm()[0]+'wrist'])
//    });
//    $('#bpd_default #b2').show().text("Wrist Flex In").bind('click.rfh', function(e){ 
//        joint_goals = arm_joints[window.arm()];
//        joint_goals.positions[5] -= scales[window.arm()[0]+'wrist']*(Math.PI/200);
//        pub_arm_joints(joint_goals)
//    });
//    $('#bpd_default #b1').show().text("Lower Elbow").bind('click.rfh', function(e){
//        pub_elbow(-0.01*scales[window.arm()[0]+'wrist'])
//    });
//};

