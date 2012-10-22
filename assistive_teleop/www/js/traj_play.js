var TrajectoryPlayback = function (arm) {
    'use strict';
    var trajPlay = this;
    getMsgDetails('hrl_pr2_traj_playback/TrajectoryPlayGoal');
    trajPlay.actClient = new ActionClient({
            ros: window.ros,
            serverName: 'trajectory_playback_'+arm[0],
            actionName: 'hrl_pr2_traj_playback/TrajectoryPlayAction'});
    trajPlay.pauseServiceClient = new window.ros.Service({
        name:'/trajectory_playback_'+arm[0]+'_pause',
        serviceType: 'std_srvs/Empty'})
    trajPlay.pause = function () {
            trajPlay.pauseServiceClient.callService({}, function () {});
        };
    trajPlay.stopServiceClient = new window.ros.Service({
        name:'/trajectory_playback_'+arm[0]+'_stop',
        serviceType: 'std_srvs/Empty'})
    trajPlay.stop = function () {
            trajPlay.stopServiceClient.callService({}, function () {});
        };

    trajPlay.modesParam = new window.ros.Param({
        name: 'face_adls_traj_modes'});
    trajPlay.trajFilesParam = new window.ros.Param({
        name: 'face_adls_traj_files' });
    trajPlay.sendGoal = function () {
        var goal = composeMsg('hrl_pr2_traj_playback/TrajectoryPlayGoal');
        var act = $('#traj_play_act_sel option:selected').val(); 
        var hand = $('#traj_play_arm_sel option:selected').val();
        var traj = $('#traj_play_select').val();
        var settings = trajPlay.trajFilesParam.value[act][hand][traj]
        goal.mode = parseInt($('input:checked','#traj_radio').val());
        goal.reverse = settings[0];
        goal.setup_velocity = settings[1];
        goal.traj_rate_mult = settings[2];
        goal.filepath = settings[3];
        var ACGoal = new trajPlay.actClient.Goal(goal);
        ACGoal.send(); 
        console.log("Sending Trajectory Play Goal");
    };
};

var initTrajPlay = function () {
    //FIXME/TODO: Fix nested parameter calls.  Separate arms and parameters 
    //on backend so separate nodes can work independently.  Current version 
    //is functional but ugly.
    window.trajPlayList = [new TrajectoryPlayback('left'), new TrajectoryPlayback('right')]
    window.trajPlayList[0].modesParam.get(function (valList) {
        window.trajPlayList[0].modesParam.value = valList;
        window.trajPlayList[1].modesParam.value = valList;
        console.log('Param: '+trajPlayList[0].modesParam.name+' -- Value:'+valList.toString());
        for (var i in valList){
            $('#traj_play_act_sel').append('<option value="'+valList[i]+'">'+valList[i]+'</option>');
            };
        window.trajPlayList[0].trajFilesParam.get(function (val) {
            console.log('Param: '+trajPlayList[0].modesParam.name+' -- Value:'+val.toString());
            window.trajPlayList[0].trajFilesParam.value = val;
            window.trajPlayList[1].trajFilesParam.value = val;
            updateInterface();
            });
        });
    
    var updateInterface = function () {
        var act = $('#traj_play_act_sel option:selected').val(); 
        var hand = $('#traj_play_arm_sel option:selected').val();
        var opts = window.trajPlayList[0].trajFilesParam.value[act][hand]
        $('#traj_play_select').empty();
        for (var key in opts) {
            $('#traj_play_select').append('<option value="'+key+'">'+key+'</option>');
        };
    };

    var sendTrajPlayGoal = function () {
        var hand = $('#traj_play_arm_sel option:selected').val();
        if (hand === 'Left') {
            window.trajPlayList[0].sendGoal();
        } else if (hand === 'Right') {
            window.trajPlayList[1].sendGoal();
        };
    };

    var pauseTrajPlay = function () {
        window.trajPlayList[0].pause();
        window.trajPlayList[1].pause();
        };

    var stopTrajPlay = function () {
        window.trajPlayList[0].stop();
        window.trajPlayList[1].stop();
    };

    $("#traj_radio").buttonset().addClass('centered');
    $(".traj_play_radio_label").addClass('centered');
    $('#traj_play_act_sel, #traj_play_arm_sel').bind('change',function(){updateInterface()});
    $('label:first', '#traj_radio').removeClass('ui-corner-left').addClass('ui-corner-top');
    $('label:last', '#traj_radio').removeClass('ui-corner-right').addClass('ui-corner-bottom');
    $('#traj_play_play').click(sendTrajPlayGoal);
    $('#traj_play_pause').click(pauseTrajPlay);
    $('#traj_play_stop').click(stopTrajPlay);
};
