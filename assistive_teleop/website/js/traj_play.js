var traj_actions = ['Shaving Left Cheek', 'Shaving Right Cheek', 'Servoing'];

function traj_play_init(){
    console.log("Begin Traj Play Init");
    var traj_play_act_spec = new ros.actionlib.ActionSpec('pr2_traj_playback/TrajectoryPlayAction');
    window.traj_play_r_client = new ros.actionlib.SimpleActionClient(node,'/trajectory_playback_r', traj_play_act_spec);

    traj_play_r_client.wait_for_server(10, function(e){
          if(!e) {log("Couldn't find right trajectory playback action server.");}
          else {console.log("Found Right Trajectory Playback Action");};
      });

    window.traj_play_l_client = new ros.actionlib.SimpleActionClient(node,'/trajectory_playback_l', traj_play_act_spec);
    traj_play_l_client.wait_for_server(10, function(e){
          if(!e) {log("Couldn't find left trajectory playback action server.");}
          else {console.log("Found Left Trajectory Playback Action");};
      });

    load_traj_activities();
    init_TrajPlayGoal();
    console.log("End Traj Play Init");
};

function init_TrajPlayGoal(){
	if (window.get_msgs_free){
        window.get_msgs_free = false;
        console.log('Locking for TrajPlayGoal');
		node.rosjs.callService('/rosbridge/msgClassFromTypeString',
                          json(["pr2_traj_playback/TrajectoryPlayGoal"]),
                          function(msg){window.TrajPlayGoal=msg;
                                        window.get_msgs_free = true;
                                        console.log('Unlocking: Got TrajPlayGoal');
                          });
	} else {
        console.log("TrajPlayGoal Waiting for msg lock");
        setTimeout(function(){init_TrajPlayGoal();},500);
    }
};

$(function(){
    $("#traj_radio").buttonset().addClass('centered');// :radio, #traj_play_radio label").button()
    $(".traj_play_radio_label").addClass('centered');
    $('#traj_play_act_sel, #traj_play_arm_sel').bind('change',function(){update_trajectories()});
    $('label:first', '#traj_radio').removeClass('ui-corner-left').addClass('ui-corner-top');
    $('label:last', '#traj_radio').removeClass('ui-corner-right').addClass('ui-corner-bottom');
    });

function load_traj_activities(){
    if (window.get_param_free){
        window.get_param_free = false;
        console.log("Traj play activities has locked get_param");
        node.rosjs.callService('/rosbridge/get_param','["face_adls_traj_modes"]',
              function(msg){window.traj_acts = msg;
                    for (var i in msg){
                        $('#traj_play_act_sel').append('<option value="'+msg[i]+'">'+msg[i]+'</option>');};
                    window.get_param_free = true;
                    console.log("Traj play has released get_param");
                    load_traj_params();
                    });
    } else {
          console.log("Traj Play Activities waiting for rosparam service");
          setTimeout(function(){load_traj_activities()},500);
    };
};

function load_traj_params(){
    if (window.get_param_free){
        window.get_param_free = false;
        console.log("Traj play has locked get_param");
        node.rosjs.callService('/rosbridge/get_param','["face_adls_traj_files"]',
                      function(msg){window.face_adls_params = msg;
                                    window.get_param_free = true;
                                    console.log("Traj play has released get_param");
                                    update_trajectories();
                                    });
    } else {
          console.log("Traj Play tab waiting for rosparam service");
          setTimeout(function(){load_traj_params()},500);
    };
};

function update_trajectories(){
    var act = $('#traj_play_act_sel option:selected').val(); 
    var hand = $('#traj_play_arm_sel option:selected').val();
    var opts = window.face_adls_params[act][hand]
    $('#traj_play_select').empty();
    for (var key in opts){
        $('#traj_play_select').append('<option value="'+key+'">'+key+'</option>');
    };
};

function traj_play_send_goal(){
    var goal = window.TrajPlayGoal;
    var act = $('#traj_play_act_sel option:selected').val(); 
    var hand = $('#traj_play_arm_sel option:selected').val();
    var traj = $('#traj_play_select').val();
    var settings = window.face_adls_params[act][hand][traj]
    goal.mode = parseInt($('input:checked','#traj_radio').val());
    goal.reverse = settings[0];
    goal.setup_velocity = settings[1];
    goal.traj_rate_mult = settings[2];
    goal.filepath = settings[3];
    if (hand == 'Right'){
        window.traj_play_r_client.send_goal(goal)
    } else {
        window.traj_play_l_client.send_goal(goal)
    };
    console.log("Sending Trajectory Play Goal");
};

