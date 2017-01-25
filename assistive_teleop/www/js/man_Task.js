function fullscreenStart(div_id) {
    var curr_w = $(div_id).css("width");
    var curr_h = $(div_id).css("height");
    var curr_z = $(div_id).css("z-index");
    var curr_p = $(div_id).css("position");
    $(div_id).css({"width":"100%",
                   "height":"100%",
                   "z-index":"10",
                   "position":"fixed"});
    return {'width':"0%",'height':"0%"};
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
function manTask_yes(curr_id) {
    //var curr_id = event.target.id;
    glow("#"+curr_id);
    unglow("#"+curr_id.substring(0, curr_id.length-3) + "no");
    $("#"+curr_id).css({'color':'black','background-color':'white'});
    $("#"+curr_id.substring(0, curr_id.length-3) + "no").css({'color':'white','background-color':'black'});
    document.getElementById(curr_id).value="true";
    document.getElementById(curr_id.substring(0, curr_id.length-3) + "no").value="false";
}
function manTask_no(curr_id) {
    //var curr_id = event.target.id;
    glow("#"+curr_id);
    unglow("#"+curr_id.substring(0, curr_id.length-2) + "yes");
    $("#"+curr_id).css({'color':'black','background-color':'white'});
    $("#"+curr_id.substring(0, curr_id.length-2) + "yes").css({'color':'white','background-color':'black'});
    document.getElementById(curr_id).value="true";
    document.getElementById(curr_id.substring(0, curr_id.length-2) + "yes").value="false";
}
function enableButton(button_id) {
    $(button_id).css("opacity", "1.0");
    $(button_id).css("pointer-events", "auto"); 
}
function disableButton(button_id) {
    $(button_id).css("opacity", "0.6");
    $(button_id).css("pointer-events", "none"); 
}
var previous_css = {'width':"0%",'height':"0%"};
var ManipulationTask = function (ros) {
    'use strict';
    var manTask = this;
    manTask.available = true;
    manTask.ros = ros;
    //Topic used in manTask
    manTask.USER_INPUT_TOPIC = "manipulation_task/user_input";
    //manTask.USER_FEEDBACK_TOPIC = "manipulation_task/user_feedback";
    manTask.EMERGENCY_TOPIC = "manipulation_task/emergency";
    manTask.STATUS_TOPIC = "manipulation_task/status";
    manTask.current_step = 1;
    manTask.max_step = 0;
    manTask.feedback_received = false;
    manTask.feedingDistanceSynched = true;
    manTask.handshaked = false;
    manTask.current_task = 'Init';
    manTask.anomaly_detected = false;
    manTask.self_stop = false;
    //status_topic and publishing
    manTask.statusPub = new manTask.ros.Topic({
        name: manTask.STATUS_TOPIC,
        messageType: 'std_msgs/String',
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
            manTask.available=false
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
            manTask.available=false
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
            manTask.available=false
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

    /*
    manTask.userFeedbackPub = new manTask.ros.Topic({
        name: manTask.USER_FEEDBACK_TOPIC,
        messageType: 'std_msgs/String'
    });
    manTask.userFeedbackPub.advertise();
    */

    manTask.questionPub  = new manTask.ros.Topic({
        name: "manipulation_task/user_feedback",
        messageType: 'hrl_msgs/StringArray'
    });
    manTask.questionPub.advertise();

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
            manTask.self_stop=false;
            document.getElementById('non_anomaly_disp').style.display='';
            document.getElementById('anomaly_disp').style.display='none';
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
        manTask.self_stop = true;
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
            manTask.self_stop = false;
            document.getElementById('non_anomaly_disp').style.display='';
            document.getElementById('anomaly_disp').style.display='none';
            return true;
        } else {
            return false;
        }
    };
    // Function to report the feedback
    /*
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
    */
    manTask.question_skip = function() {
        var msg = new manTask.ros.Message({
            data: ['SKIP']
        });
        manTask.questionPub.publish(msg);
    };
    manTask.question_send = function() {
        var question_array = [];
        if (manTask.current_task == "Scooping") {
            var curr_table = document.getElementById("scooping_questions");
            for (var i = 0; i < curr_table.rows.length; i++) {

                if (document.getElementById("scooping_questions" + (i+1) +"_yes").value=="true") {
                    question_array.push("TRUE");
                } else {
                    question_array.push("FALSE");
                }
            }
            //document.getElementById("question_skip").innerHTML = "4";
        } 
        if (manTask.current_task == "Feeding") {
            var curr_table = document.getElementById("feeding_questions");
            for (var i = 0; i < curr_table.rows.length; i++) {
                if (document.getElementById("feeding_questions" + (i+1) +"_yes").value=="true") {
                    question_array.push("TRUE");
                } else {
                    question_array.push("FALSE");
                }
            }
            //document.getElementById("question_skip").innerHTML = "4";
        } 
        //document.getElementById("question_skip").innerHTML = "hello2";
        var msg = new manTask.ros.Message({
            data: question_array
        });
        //document.getElementById("question_skip").innerHTML = "hello";
        manTask.questionPub.publish(msg);
    }

    //part added in 7/18
    manTask.statusSub = new manTask.ros.Topic({
        name: manTask.STATUS_TOPIC,
        messageType: 'std_msgs/String',
    });
    manTask.statusSub.subscribe(function (msg) {
        manTask.current_task = msg.data;
        if (msg.data == "Scooping") {
            //document.getElementById("fullscreen_scooping_paragraph").style.visibility='visible';
            //document.getElementById("fullscreen_feeding_paragraph").style.visibility='hidden';
            document.getElementById("scooping_questions").style.display='';//setAttribute("hidden", false);//style.visibility='visible';
            document.getElementById("feeding_questions").style.display='none';//setAttribute("hidden", true);//style.visibility='hidden';
        } else if (msg.data == "Feeding") {
            //document.getElementById("fullscreen_scooping_paragraph").style.visibility='hidden';
            //document.getElementById("fullscreen_feeding_paragraph").style.visibility='visible';
            document.getElementById("scooping_questions").style.display='none';//visibility='hidden';//setAttribute("hidden", true);//
            document.getElementById("feeding_questions").style.display='';//visibility='visible';//setAttribute("hidden",false);//
        }
    });
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
            fullscreenStop('#fullscreenOverlay2', previous_css);
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

            document.getElementById('step_table1').innerHTML = " ";
            document.getElementById('step_table2').innerHTML = "Waiting for robot";
            unglow('#step_table2');
            document.getElementById('step_table3').innerHTML = " ";
            fullscreenStop('#fullscreenOverlay', previous_css);
            fullscreenStop('#fullscreenOverlay2', previous_css);
            manTask.anomaly_detected = false;
            disableButton('#ad_scooping_sense_min');
            disableButton('#ad_scooping_sense_max');
            disableButton('#ad_scooping_slider');
            disableButton('#ad_feeding_sense_min');
            disableButton('#ad_feeding_sense_max');
            disableButton('#ad_feeding_slider');
            manTask.available=true;
            manTask.start()
        } else if (msg.data == 'stand by') {
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Clean');
            enableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            enableButton('#man_task_start');
            disableButton('#man_task_Skip');

            document.getElementById('step_table1').innerHTML = " ";
            document.getElementById('step_table2').innerHTML = "Waiting for robot";
            unglow('#step_table2');
            document.getElementById('step_table3').innerHTML = " ";
            fullscreenStop('#fullscreenOverlay', previous_css);
            fullscreenStop('#fullscreenOverlay2', previous_css);
            manTask.anomaly_detected = false;
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
            fullscreenStop('#fullscreenOverlay2', previous_css);
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
            fullscreenStop('#fullscreenOverlay2', previous_css);
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
            fullscreenStop('#fullscreenOverlay2', previous_css);
            enableButton('#ad_scooping_sense_min');
            enableButton('#ad_scooping_sense_max');
            enableButton('#ad_scooping_slider');
            enableButton('#ad_feeding_sense_min');
            enableButton('#ad_feeding_sense_max');
            enableButton('#ad_feeding_slider');
            manTask.available=true;
            if (manTask.anomaly_detected && !manTask.self_stop) {
                document.getElementById('anomaly_disp').style.display = '';
                document.getElementById('non_anomaly_disp').style.display = 'none';
            }
            manTask.anomaly_detected = false;
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
            previous_css = fullscreenStart('#fullscreenOverlay2');
            /*
            if (manTask.anomaly_detected) {
                document.getElementById("fullscreen_anomaly_paragraph").innerHTML = "Anomaly was detected";
            } else {
                document.getElementById("fullscreen_anomaly_paragraph").innerHTML = "Anomaly was not detected";
            }
            */
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

    //part added 1/20/17 to display anomaly type

    manTask.anomalyResultSub = new manTask.ros.Topic({
        name: 'manipulation_task/anomaly_type',
        messageType: 'std_msgs/String'});
    manTask.anomalyResultSub.subscribe( function(msg) {
        if (msg.data == '') {
            document.getElementById('anomaly_type').innerHTML = 'Classifying';
        } else {
            document.getElementById('anomaly_type').innerHTML = msg.data;
        }
        document.getElementById('anomaly_type').style.display = '';
        document.getElementById('anomaly_disp').style.display = '';
        document.getElementById('non_anomaly_disp').style.display='none';
    });
    //part added on 4/7 to accomodate anomaly signal.
    manTask.emergencySub = new manTask.ros.Topic({
        name: 'manipulation_task/emergency',
        messageType: 'std_msgs/String'});
    manTask.emergencySub.subscribe(function (msg) {
        if(msg.data!="STOP") {
            manTask.anomaly_detected = true;
            if (manTask.self_stop == false) {
                document.getElementById('anomaly_type').innerHTML = 'Classifying';
                document.getElementById('non_anomaly_disp').style.display='none';
                document.getElementById('anomaly_disp').style.display='';
            } else {
                document.getElementById('non_anomaly_disp').style.display='';
                document.getElementById('anomaly_disp').style.display='none';
            }
        }

    });



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
    });

    $('#man_task_Feeding').click(function(){
        if(assistive_teleop.manTask.handshaked) {
            assistive_teleop.manTask.feed();
        }
    });

    $('#man_task_Clean').click(function(){
        if(assistive_teleop.manTask.handshaked) {
            assistive_teleop.manTask.both();
        }
    });
    $('#man_task_start').click(function(){
        assistive_teleop.manTask.start()
    });
    $('#man_task_stop').click(function(){
        assistive_teleop.manTask.stop();
    });
    $('#fullscreenOverlay').click(function(){
        assistive_teleop.manTask.stop();
    });
    $('#man_task_Continue').click(function(){
        assistive_teleop.manTask.continue_();
    });
    /*
    $('#man_task_success').click(function(){
        assistive_teleop.manTask.success();
    });
    $('#man_task_Fail').click(function(){
        assistive_teleop.manTask.failure();
    });

    $('#man_task_Skip').click(function(){
        assistive_teleop.manTask.skip();
    });
    */
    $('#question_skip').click(function() {
        assistive_teleop.manTask.question_skip();
    });
    $('#question_send').click(function() {
        assistive_teleop.manTask.question_send();
    });
    manTask_yes("scooping_questions1_yes");
    manTask_no("scooping_questions2_no");
    manTask_no("scooping_questions3_no");
    manTask_yes("feeding_questions1_yes");
    manTask_no("feeding_questions2_no");
    manTask_no("feeding_questions3_no");
    
}
