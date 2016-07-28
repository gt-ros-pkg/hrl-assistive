function fullscreenStart(div_id) {
    var curr_w = $(div_id).css("width");
    var curr_h = $(div_id).css("height");
    var curr_z = $(div_id).css("z-index");
    var curr_p = $(div_id).css("position");
    $(div_id).css({"width":"100%",
                   "height":"100%",
                   "z-index":"10",
                   "position":"fixed"});
    return {'height':"0%"};
}
function fullscreenStop(div_id, previous_css) {
    $(div_id).css(previous_css);
}
function glow(div_id) {
    $(div_id).css({"-webkit-box-shadow": "0px 0px 20px rgba(255,255,255,0.8)",
                   "-moz-box-shadow": "0px 0px 20px rgba(255,255,255,0.8)",
                   "box-shadow": "0px 0px 20px rgba(255,255,255,0.8)"});
}
function unglow(div_id) {
    $(div_id).css({"-webkit-box-shadow": "",
                   "-moz-box-shadow": "",
                   "box-shadow": ""});
}
function enableButton(button_id) {
    $(button_id).css("opacity", "1.0");
    $(button_id).css("pointer-events", "auto"); 
}
function disableButton(button_id) {
    $(button_id).css("opacity", "0.6");
    $(button_id).css("pointer-events", "none"); 
}
var previous_css = {'height':"0%"};
var ManipulationTask = function (ros) {
    'use strict';
    var manTask = this;
    manTask.available = true;
    manTask.ros = ros;
    //Topic used in manTask
    manTask.USER_INPUT_TOPIC = "manipulation_task/user_input";
    manTask.USER_FEEDBACK_TOPIC = "manipulation_task/user_feedback";
    manTask.EMERGENCY_TOPIC = "manipulation_task/emergency";
    manTask.STATUS_TOPIC = "manipulation_task/status";
    manTask.current_step = 1;
    manTask.max_step = 0;
    manTask.feedback_received = false;
    manTask.feedingDistanceSynched = true;
    manTask.handshaked = false;
    //status_topic and publishing
    manTask.statusPub = new manTask.ros.Topic({
        name: manTask.STATUS_TOPIC,
        messageType: 'std_msgs/String'
    });
    manTask.statusPub.advertise();
    manTask.scoop = function () {
        if (manTask.available) {
            var msg = new manTask.ros.Message({
                data: 'Scooping'
            });
            manTask.statusPub.publish(msg);
            manTask.current_step = 0;
            manTask.max_step = 3;
            assistive_teleop.log('Please, follow the step 2 to select the action.');
            manTask.feedback_received = false;
            return true;
        } else {
            return false;
        }
    };

    manTask.feed = function () {
        if (manTask.available) {
            var msg = new manTask.ros.Message({
                data: 'Feeding'
            });
            assistive_teleop.log('Please, follow the step 2 to select the action.');
            manTask.statusPub.publish(msg);
            manTask.current_step = 0;
            manTask.max_step = 5;
            manTask.feedback_received = false;
            return true;
        } else {
            return false;
        }
    };

    manTask.both = function () {
        if (manTask.available) {
            var msg = new manTask.ros.Message({
                data: 'Clean'
            });
            assistive_teleop.log('Please, follow the step 2 to select the action.');
            manTask.statusPub.publish(msg);
            return true;
        } else {
            return false;
        }
    };


    //Publisher used for start, stop, and continue
    manTask.userInputPub = new manTask.ros.Topic({
        name: manTask.USER_INPUT_TOPIC,
        messageType: 'std_msgs/String'
    });
    manTask.userInputPub.advertise();

    manTask.emergencyPub = new manTask.ros.Topic({
        name: manTask.EMERGENCY_TOPIC,
        messageType: 'std_msgs/String',
        queue_size: 2
    });
    manTask.emergencyPub.advertise();

    manTask.userFeedbackPub = new manTask.ros.Topic({
        name: manTask.USER_FEEDBACK_TOPIC,
        messageType: 'std_msgs/String'
    });
    manTask.userFeedbackPub.advertise();
    // Function for start, stop, and continue

    manTask.feedingDistancePub = new manTask.ros.Topic({
        name: 'feeding/manipulation_task/feeding_dist_request',
        messageType: 'std_msgs/Int64'
    });
    manTask.feedingDistancePub.advertise();

    manTask.start = function () {
        if (manTask.available) {
            var msg = new manTask.ros.Message({
                data: 'Start'
            });
            manTask.userInputPub.publish(msg);
            assistive_teleop.log('Starting the manipulation task');
            assistive_teleop.log('Please, follow the step 3 when "Requesting Feedback" message shows up.');
            console.log('Publishing Start msg to manTask system.');
            //$('#tabs').css.("width","100%");
            return true;
        } else {
            return false;
        }
    };

    manTask.stop = function () {
        var msg = new manTask.ros.Message({
          data: 'STOP'
        });
        manTask.emergencyPub.publish(msg);
        assistive_teleop.log('Stopping the manipulation task');
        assistive_teleop.log('Please, press "Continue" to re-start the action. Or re-start from step 1.');
        console.log('Publishing Stop msg to manTask system.');
        manTask.available=false;
    };

    manTask.continue_ = function () {
        if (manTask.available) {
            var msg = new manTask.ros.Message({
                data: 'Continue'
            });
            manTask.userInputPub.publish(msg);
            assistive_teleop.log('Continuing the manipulation task');
            assistive_teleop.log('Please, follow the step 3 when "Requesting Feedback" message shows up.');
            console.log('Publishing Continue msg to manTask system.');
            return true;
        } else {
            return false;
        }
    };
    // Function to report the feedback
    manTask.success = function () {
        var msg = new manTask.ros.Message({
          data: 'SUCCESS'
        });
        manTask.userFeedbackPub.publish(msg);
        assistive_teleop.log('Successful run');
        console.log('Reporting the feedback message.');
        manTask.feedback_received = true;
    };

    manTask.failure = function () {
        var msg = new manTask.ros.Message({
          data: 'FAIL'
        });
        manTask.userFeedbackPub.publish(msg);
        assistive_teleop.log('Failed run');
        console.log('Reporting the feedback message.');
        manTask.feedback_received = true;
    };

    manTask.skip = function () {
        var msg = new manTask.ros.Message({
          data: 'SKIP'
        });
        manTask.userFeedbackPub.publish(msg);
        assistive_teleop.log('No report has been filed');
        console.log('Reporting the feedback message.');
        manTask.feedback_received = true;
    };

    //part added in 7/18
    manTask.availableSub = new manTask.ros.Topic({
        name: 'manipulation_task/available',
        messageType: 'std_msgs/String'});
    manTask.availableSub.subscribe(function (msg) {
        if(msg.data=="true") {
            manTask.available=true;
        } else {
            manTask.available=false;
        }
    });

    manTask.feedingDistanceRequest = function() {
        if (manTask.feedingDistanceSynched) {
            var new_dist = parseInt(document.getElementById("man_task_Feeding_dist").value);
            var msg = new manTask.ros.Message({
                data: new_dist
            });
            manTask.feedingDistancePub.publish(msg);
            manTask.feedingDistanceSynched = false;
            document.getElementById("man_task_Feeding_dist").disabled = true;
        }
    }

    manTask.feedingDistanceSub = new manTask.ros.Topic({
        name: 'feeding/manipulation_task/feeding_dist_state',
        messageType: 'std_msgs/Int64'
    });
    manTask.feedingDistanceSub.subscribe(function (msg) {
        document.getElementById("man_task_Feeding_dist").value = msg.data;
        manTask.feedingDistanceSynched = true;
        document.getElementById("man_task_Feeding_dist").disabled = false;
    });
    //part added.
    /*
    manTask.feedbackSub = new manTask.ros.Topic({
        name: 'manipulation_task/feedbackRequest',
        messageType: 'std_msgs/String'});
    manTask.feedbackSub.subscribe(function (msg) {
        assistive_teleop.log(msg.data);
        if(msg.data=="Requesting Feedback!") {
        //assistive_teleop.log("worked?");
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Clean');
            //enableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            enableButton('#man_task_success');
            enableButton('#man_task_Fail');
            disableButton('#man_task_start');
            enableButton('#man_task_Skip');
        }
        if(msg.data=="No feedback requested") {
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Clean');
            //enableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            enableButton('#man_task_success');
            enableButton('#man_task_Fail');
            disableButton('#man_task_start');
            enableButton('#man_task_Skip');
            /*
            enableButton('#man_task_Scooping');
            enableButton('#man_task_Feeding');
            enableButton('#man_task_Clean');
            disableButton('#man_task_start');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            disableButton('#man_task_stop');
            disableButton('#man_task_Skip');
            enableButton('#ad_scooping_sense_min');
            enableButton('#ad_scooping_sense_max');
            enableButton('#ad_scooping_slider');
            enableButton('#ad_feeding_sense_min');
            enableButton('#ad_feeding_sense_max');
            enableButton('#ad_feeding_slider');
            */
    /*
        }
        fullscreenStop('#fullscreenOverlay', previous_css);
    });
    */

    manTask.guiStatusSub = new manTask.ros.Topic({
        name: 'manipulation_task/gui_status',
        messageType: 'std_msgs/String'});
    manTask.guiStatusSub.subscribe(function(msg) {
        if(msg.data == 'select task') {
            enableButton('#man_task_Scooping');
            enableButton('#man_task_Feeding');
            enableButton('#man_task_Clean');
            disableButton('#man_task_start');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            disableButton('#man_task_stop');
            disableButton('#man_task_Skip');

            fullscreenStop('#fullscreenOverlay', previous_css);
            enableButton('#ad_scooping_sense_min');
            enableButton('#ad_scooping_sense_max');
            enableButton('#ad_scooping_slider');
            enableButton('#ad_feeding_sense_min');
            enableButton('#ad_feeding_sense_max');
            enableButton('#ad_feeding_slider');
            manTask.available=true;
        } else if (msg.data == 'wait start') {
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Clean');
            disableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            enableButton('#man_task_start');
            disableButton('#man_task_Skip');

            fullscreenStop('#fullscreenOverlay', previous_css);
            disableButton('#ad_scooping_sense_min');
            disableButton('#ad_scooping_sense_max');
            disableButton('#ad_scooping_slider');
            disableButton('#ad_feeding_sense_min');
            disableButton('#ad_feeding_sense_max');
            disableButton('#ad_feeding_slider');
        } else if (msg.data == 'in motion') {
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Clean');
            enableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            disableButton('#man_task_start');
            disableButton('#man_task_Skip');

            previous_css = fullscreenStart('#fullscreenOverlay', previous_css);
            disableButton('#ad_scooping_sense_min');
            disableButton('#ad_scooping_sense_max');
            disableButton('#ad_scooping_slider');
            disableButton('#ad_feeding_sense_min');
            disableButton('#ad_feeding_sense_max');
            disableButton('#ad_feeding_slider');            
            manTask.available=true;
        } else if (msg.data == 'stopping') {
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Clean');
            enableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            disableButton('#man_task_start');
            disableButton('#man_task_Skip');

            fullscreenStop('#fullscreenOverlay', previous_css);
            disableButton('#ad_scooping_sense_min');
            disableButton('#ad_scooping_sense_max');
            disableButton('#ad_scooping_slider');
            disableButton('#ad_feeding_sense_min');
            disableButton('#ad_feeding_sense_max');
            disableButton('#ad_feeding_slider');
            manTask.available=false;
        } else if (msg.data == 'stopped') {
            enableButton('#man_task_Scooping');
            enableButton('#man_task_Feeding');
            enableButton('#man_task_Clean');
            disableButton('#man_task_stop');
            enableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            disableButton('#man_task_start');
            disableButton('#man_task_Skip');
            
            fullscreenStop('#fullscreenOverlay', previous_css);
            enableButton('#ad_scooping_sense_min');
            enableButton('#ad_scooping_sense_max');
            enableButton('#ad_scooping_slider');
            enableButton('#ad_feeding_sense_min');
            enableButton('#ad_feeding_sense_max');
            enableButton('#ad_feeding_slider');
            manTask.available=true;
        } else if (msg.data == 'request feedback') {
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Clean');
            disableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            enableButton('#man_task_success');
            enableButton('#man_task_Fail');
            disableButton('#man_task_start');
            enableButton('#man_task_Skip');

            fullscreenStop('#fullscreenOverlay', previous_css);
            enableButton('#ad_scooping_sense_min');
            enableButton('#ad_scooping_sense_max');
            enableButton('#ad_scooping_slider');
            enableButton('#ad_feeding_sense_min');
            enableButton('#ad_feeding_sense_max');
            enableButton('#ad_feeding_slider');
            manTask.available=false;
        }
        manTask.handshaked = true;
    });
    /*
    manTask.adScoopingSliderSub = new manTask.ros.Topic({
        name: 'scooping/manipulation_task/ad_senstivity_request',
        messageType:'std_msgs/Float64'});
    manTask.adScoopingSliderSub.subscribe(function (msg) {
        
    });
    */
    manTask.proceedSub = new manTask.ros.Topic({
        name: 'manipulation_task/proceed',
        messageType: 'std_msgs/String'});
    manTask.proceedSub.subscribe(function (msg) {
        var cmd = "";
        var sub_cmd = "";
        if(msg.data.length > 0) {
            var cmd_loc = 0;
            for (var i = 0; i < msg.data.length; i++) {
                if (msg.data.charAt(i) == ':') {
                    cmd_loc = i;
                    break;
                }
            }
            if (cmd_loc == 0) {
                cmd_loc = msg.data.length;
            }
            cmd = msg.data.substring(0,cmd_loc);
            sub_cmd = msg.data.substring(cmd_loc+1, msg.data.length);
        }
        if(cmd=="Next") {
            document.getElementById('step_table1').innerHTML = document.getElementById('step_table2').innerHTML;
            document.getElementById('step_table2').innerHTML = document.getElementById('step_table3').innerHTML;
            document.getElementById('step_table3').innerHTML = sub_cmd;
            /*
            manTask.current_step = manTask.current_step + 1;
            if (manTask.current_step <= manTask.max_step) {
                unglow('#step_table' + (manTask.current_step - 1));
                glow('#step_table' + manTask.current_step);
            } else if (manTask.current_step == (manTask.max_step + 1)) {
                unglow('#step_table' + (manTask.current_step - 1));
            }
            */
        } else if (cmd == "Start") {
            var comma_loc = 0;
            var arr       = [];
            for (var i = 0; i <= sub_cmd.length; i++) {
                if (sub_cmd.charAt(i) == ',' || i == sub_cmd.length) {
                    arr.push(sub_cmd.substring(comma_loc, i));
                    comma_loc = i + 1;
                }
            }
            document.getElementById('step_table1').innerHTML = " ";
            document.getElementById('step_table2').innerHTML = arr[0];
            document.getElementById('step_table3').innerHTML = arr[1];
            glow('#step_table2');
        } else if (cmd == "Set") {
            var comma_loc = 0;
            var arr       = [];
            for (var i = 0; i <= sub_cmd.length; i++) {
                if (sub_cmd.charAt(i) == ',' || i == sub_cmd.length) {
                    arr.push(sub_cmd.substring(comma_loc, i));
                    comma_loc = i + 1;
                }
            }
            document.getElementById('step_table1').innerHTML = arr[0];
            document.getElementById('step_table2').innerHTML = arr[1];
            document.getElementById('step_table3').innerHTML = arr[2];
            glow('#step_table2');
        } else if (cmd == "Done") {
            unglow('#step_table2');
            fullscreenStop('#fullscreenOverlay', previous_css);
            /*
            if(manTask.feedback_received) {
                enableButton('#man_task_Scooping');
                enableButton('#man_task_Feeding');
                enableButton('#man_task_Clean');
                disableButton('#man_task_start');
                disableButton('#man_task_Continue');
                disableButton('#man_task_success');
                disableButton('#man_task_Fail');
                disableButton('#man_task_stop');
                disableButton('#man_task_Skip');
                enableButton('#ad_scooping_sense_min');
                enableButton('#ad_scooping_sense_max');
                enableButton('#ad_scooping_slider');
                enableButton('#ad_feeding_sense_min');
                enableButton('#ad_feeding_sense_max');
                enableButton('#ad_feeding_slider');
            } else {
                disableButton('#man_task_Scooping');
                disableButton('#man_task_Feeding');
                disableButton('#man_task_Clean');
                disableButton('#man_task_stop');
                disableButton('#man_task_Continue');
                enableButton('#man_task_success');
                enableButton('#man_task_Fail');
                disableButton('#man_task_start');
                enableButton('#man_task_Skip');
            }
            */
        }
    });


    manTask.scoopingResultSub= new manTask.ros.Topic({
        name: 'scooping/manipulation_task/eval_status',
        messageType: 'hrl_msgs/FloatArray'
    });
    manTask.scoopingResultSub.subscribe( function(msg) {
        if (msg.data.length == 2) {
            //document.getElementById('ad_scooping_result_1').innerHTML = msg.data[0];
            //document.getElementById('ad_scooping_result_1').innerHTML = document.getElementById('ad_scooping_result_1').innerHTML + "%";
            document.getElementById('ad_scooping_result_2').innerHTML = msg.data[1];
            document.getElementById('ad_scooping_result_2').innerHTML = parseFloat(document.getElementById('ad_scooping_result_2').innerHTML).toFixed(1) + "%";
        }
    });


    manTask.feedingResultSub= new manTask.ros.Topic({
        name: 'feeding/manipulation_task/eval_status',
        messageType: 'hrl_msgs/FloatArray'
    });
    manTask.feedingResultSub.subscribe( function(msg) {
        //document.getElementById('ad_feeding_result_1').innerHTML = "hello";
        if (msg.data.length == 2) {
            //document.getElementById('ad_feeding_result_1').innerHTML = msg.data[0];
            //document.getElementById('ad_feeding_result_1').innerHTML = document.getElementById('ad_feeding_result_1').innerHTML + "%";
            document.getElementById('ad_feeding_result_2').innerHTML = msg.data[1];
            document.getElementById('ad_feeding_result_2').innerHTML = parseFloat(document.getElementById('ad_feeding_result_2').innerHTML).toFixed(1) + "%";
        }
    });

    //part added on 4/7 to accomodate anomaly signal.
    /*
    manTask.emergencySub = new manTask.ros.Topic({
        name: 'manipulation_task/emergency',
        messageType: 'std_msgs/String'});
    manTask.emergencySub.subscribe(function (msg) {
        if(msg.data!="STOP") {
            manTask.available = false;
            enableButton('#man_task_Scooping');
            enableButton('#man_task_Feeding');
            enableButton('#man_task_Clean');
            enableButton('#man_task_stop');
            enableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            disableButton('#man_task_start');
            disableButton('#man_task_Skip');
            
            fullscreenStop('#fullscreenOverlay', previous_css);
            enableButton('#ad_scooping_sense_min');
            enableButton('#ad_scooping_sense_max');
            enableButton('#ad_scooping_slider');
            enableButton('#ad_feeding_sense_min');
            enableButton('#ad_feeding_sense_max');
            enableButton('#ad_feeding_slider');

        }

    });
    */



};

var initManTaskTab = function() {
    assistive_teleop.manTask = new ManipulationTask(assistive_teleop.ros);
    assistive_teleop.log('initiating manipulation Task');
    $('#man_task_Feeding_dist').change(function(){
        assistive_teleop.manTask.feedingDistanceRequest();
    });
    $('#man_task_Scooping').click(function(){
        if(assistive_teleop.manTask.handshaked) {
            assistive_teleop.manTask.scoop();
        }
        /*
        if (assistive_teleop.manTask.scoop()) {
            /*
            var table = document.getElementById('step_table');
            while(table.rows[0]) table.deleteRow(0);
            var row = table.insertRow(0);
            for (i = 0; i < assistive_teleop.manTask.max_step; i++) {
                var cell = row.insertCell(i);
                cell.id = 'step_table' + (i + 1);
                cell.innerHTML = "Scooping" + (i + 1);
            }
            glow('#step_table' + (1));
            */
        /*
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Clean');
            disableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            enableButton('#man_task_start');
            disableButton('#man_task_Skip');
        }
        */
    });

    $('#man_task_Feeding').click(function(){
        if(assistive_teleop.manTask.handshaked) {
            assistive_teleop.manTask.feed();
        }
        /*
        if (assistive_teleop.manTask.feed()) {
            /*
            var table = document.getElementById('step_table');
            while(table.rows[0]) table.deleteRow(0);
            var row = table.insertRow(0);
            for (i = 0; i < assistive_teleop.manTask.max_step; i++) {
                var cell = row.insertCell(i);
                cell.id = 'step_table' + (i + 1);
                cell.innerHTML = "Feeding" + (i + 1);
            }
            glow('#step_table1');
            */
        /*
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Clean');
            disableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            enableButton('#man_task_start');
            disableButton('#man_task_Skip');
        }
        */
    });

    $('#man_task_Clean').click(function(){
        if(assistive_teleop.manTask.handshaked) {
            assistive_teleop.manTask.both();
        }
        /*
        if(assistive_teleop.manTask.both()) {
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Clean');
            disableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            enableButton('#man_task_start');
            disableButton('#man_task_Skip');
        }
        */
    });
    $('#man_task_start').click(function(){
        if(assistive_teleop.manTask.start()) {
            document.getElementById('step_table1').innerHTML = " ";
            document.getElementById('step_table2').innerHTML = "Waiting for robot";
            unglow('#step_table2');
            document.getElementById('step_table3').innerHTML = " ";
            /*
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Clean');
            enableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            disableButton('#man_task_start');
            disableButton('#man_task_Skip');

            previous_css = fullscreenStart('#fullscreenOverlay');
            disableButton('#ad_scooping_sense_min');
            disableButton('#ad_scooping_sense_max');
            disableButton('#ad_scooping_slider');
            disableButton('#ad_feeding_sense_min');
            disableButton('#ad_feeding_sense_max');
            disableButton('#ad_feeding_slider');
            */
        }

    });
    $('#man_task_stop').click(function(){
        assistive_teleop.manTask.stop();
        /*
        enableButton('#man_task_Scooping');
        enableButton('#man_task_Feeding');
        enableButton('#man_task_Clean');
        enableButton('#man_task_stop');
        enableButton('#man_task_Continue');
        disableButton('#man_task_success');
        disableButton('#man_task_Fail');
        disableButton('#man_task_start');
        disableButton('#man_task_Skip');

        fullscreenStop('#fullscreenOverlay', previous_css);
        enableButton('#ad_scooping_sense_min');
        enableButton('#ad_scooping_sense_max');
        enableButton('#ad_scooping_slider');
        enableButton('#ad_feeding_sense_min');
        enableButton('#ad_feeding_sense_max');
        enableButton('#ad_feeding_slider');
        */
    });
    $('#fullscreenOverlay').click(function(){
        assistive_teleop.manTask.stop();
        /*
        enableButton('#man_task_Scooping');
        enableButton('#man_task_Feeding');
        enableButton('#man_task_Clean');
        enableButton('#man_task_stop');
        enableButton('#man_task_Continue');
        disableButton('#man_task_success');
        disableButton('#man_task_Fail');
        disableButton('#man_task_start');
        disableButton('#man_task_Skip');

        fullscreenStop('#fullscreenOverlay', previous_css);
        enableButton('#ad_scooping_sense_min');
        enableButton('#ad_scooping_sense_max');
        enableButton('#ad_scooping_slider');
        enableButton('#ad_feeding_sense_min');
        enableButton('#ad_feeding_sense_max');
        enableButton('#ad_feeding_slider');
        */
    });
    $('#man_task_Continue').click(function(){
        assistive_teleop.manTask.continue_();
        /*
        if (assistive_teleop.manTask.continue_()) {
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Clean');
            enableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            disableButton('#man_task_start');
            disableButton('#man_task_Skip');

            previous_css = fullscreenStart('#fullscreenOverlay');
            disableButton('#ad_scooping_sense_min');
            disableButton('#ad_scooping_sense_max');
            disableButton('#ad_scooping_slider');
            disableButton('#ad_feeding_sense_min');
            disableButton('#ad_feeding_sense_max');
            disableButton('#ad_feeding_slider');
        }
        */
    });
    $('#man_task_success').click(function(){
        assistive_teleop.manTask.success();
        /*
        enableButton('#man_task_Scooping');
        enableButton('#man_task_Feeding');
        enableButton('#man_task_Clean');
        disableButton('#man_task_stop');
        disableButton('#man_task_Continue');
        disableButton('#man_task_success');
        disableButton('#man_task_Fail');
        disableButton('#man_task_start');
        disableButton('#man_task_Skip');

        enableButton('#ad_scooping_sense_min');
        enableButton('#ad_scooping_sense_max');
        enableButton('#ad_scooping_slider');
        enableButton('#ad_feeding_sense_min');
        enableButton('#ad_feeding_sense_max');
        enableButton('#ad_feeding_slider');
        */
    });
    $('#man_task_Fail').click(function(){
        assistive_teleop.manTask.failure();
        /*
        enableButton('#man_task_Scooping');
        enableButton('#man_task_Feeding');
        enableButton('#man_task_Clean');
        disableButton('#man_task_stop');
        disableButton('#man_task_Continue');
        disableButton('#man_task_success');
        disableButton('#man_task_Fail');
        disableButton('#man_task_start');
        disableButton('#man_task_Skip');

        enableButton('#ad_scooping_sense_min');
        enableButton('#ad_scooping_sense_max');
        enableButton('#ad_scooping_slider');
        enableButton('#ad_feeding_sense_min');
        enableButton('#ad_feeding_sense_max');
        enableButton('#ad_feeding_slider');
        */
    });

    $('#man_task_Skip').click(function(){
        assistive_teleop.manTask.skip();
        /*
        enableButton('#man_task_Scooping');
        enableButton('#man_task_Feeding');
        enableButton('#man_task_Clean');
        disableButton('#man_task_stop');
        disableButton('#man_task_Continue');
        disableButton('#man_task_success');
        disableButton('#man_task_Fail');
        disableButton('#man_task_start');
        disableButton('#man_task_Skip');

        enableButton('#ad_scooping_sense_min');
        enableButton('#ad_scooping_sense_max');
        enableButton('#ad_scooping_slider');
        enableButton('#ad_feeding_sense_min');
        enableButton('#ad_feeding_sense_max');
        enableButton('#ad_feeding_slider');
        */
    });


}
