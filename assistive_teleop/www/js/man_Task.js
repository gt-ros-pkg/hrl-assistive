function enableButton(button_id) {
    $(button_id).css("opacity", "1.0");
    $(button_id).css("pointer-events", "auto"); 
}
function disableButton(button_id) {
    $(button_id).css("opacity", "0.6");
    $(button_id).css("pointer-events", "none"); 
}
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
            assistive_teleop.log('Please, follow the step 2 to select the action.');
            manTask.statusPub.publish(msg);
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
            return true;
        } else {
            return false;
        }
    };

    manTask.both = function () {
        if (manTask.available) {
            var msg = new manTask.ros.Message({
                data: 'Init'
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
        messageType: 'std_msgs/String'
    });
    manTask.emergencyPub.advertise();

    manTask.userFeedbackPub = new manTask.ros.Topic({
        name: manTask.USER_FEEDBACK_TOPIC,
        messageType: 'std_msgs/String'
    });
    manTask.userFeedbackPub.advertise();
    // Function for start, stop, and continue
    manTask.start = function () {
        if (manTask.available) {
            var msg = new manTask.ros.Message({
                data: 'Start'
            });
            manTask.userInputPub.publish(msg);
            assistive_teleop.log('Starting the manipulation task');
            assistive_teleop.log('Please, follow the step 3 when "Requesting Feedback" message shows up.');
            console.log('Publishing Start msg to manTask system.');
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
    };

    manTask.failure = function () {
        var msg = new manTask.ros.Message({
          data: 'FAIL'
        });
        manTask.userFeedbackPub.publish(msg);
        assistive_teleop.log('Failed run');
        console.log('Reporting the feedback message.');
    };

    manTask.skip = function () {
        var msg = new manTask.ros.Message({
          data: 'SKIP'
        });
        manTask.userFeedbackPub.publish(msg);
        assistive_teleop.log('No report has been filed');
        console.log('Reporting the feedback message.');
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
    //part added.
    manTask.feedbackSub = new manTask.ros.Topic({
        name: 'manipulation_task/feedbackRequest',
        messageType: 'std_msgs/String'});
    manTask.feedbackSub.subscribe(function (msg) {
        assistive_teleop.log(msg.data);
        if(msg.data=="Requesting Feedback!") {
        //assistive_teleop.log("worked?");
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Init');
            disableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            enableButton('#man_task_success');
            enableButton('#man_task_Fail');
            disableButton('#man_task_start');
            enableButton('#man_task_Skip');
        }
        if(msg.data=="No feedback requested") {
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Init');
            disableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            enableButton('#man_task_success');
            enableButton('#man_task_Fail');
            disableButton('#man_task_start');
            enableButton('#man_task_Skip');
            /*
            enableButton('#man_task_Scooping');
            enableButton('#man_task_Feeding');
            enableButton('#man_task_Init');
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
        }

    });

    //part added on 4/7 to accomodate anomaly signal.
    manTask.emergencySub = new manTask.ros.Topic({
        name: 'manipulation_task/emergency',
        messageType: 'std_msgs/String'});
    manTask.emergencySub.subscribe(function (msg) {
        if(msg.data!="STOP") {
            manTask.available = false;
            enableButton('#man_task_Scooping');
            enableButton('#man_task_Feeding');
            enableButton('#man_task_Init');
            disableButton('#man_task_start');
            enableButton('#man_task_Continue');
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

        }

    });




};

var initManTaskTab = function() {
  assistive_teleop.manTask = new ManipulationTask(assistive_teleop.ros);
  assistive_teleop.log('initiating manipulation Task');

    $('#man_task_Scooping').click(function(){
        if (assistive_teleop.manTask.scoop()) {
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Init');
            disableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            enableButton('#man_task_start');
            disableButton('#man_task_Skip');
        }
    });

    $('#man_task_Feeding').click(function(){
        if (assistive_teleop.manTask.feed()) {
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Init');
            disableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            enableButton('#man_task_start');
            disableButton('#man_task_Skip');
        }
    });

    $('#man_task_Init').click(function(){
        if(assistive_teleop.manTask.both()) {
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Init');
            disableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            enableButton('#man_task_start');
            disableButton('#man_task_Skip');
        }
 
    });
    $('#man_task_start').click(function(){
        if(assistive_teleop.manTask.start()) {
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Init');
            enableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            disableButton('#man_task_start');
            disableButton('#man_task_Skip');

            disableButton('#ad_scooping_sense_min');
            disableButton('#ad_scooping_sense_max');
            disableButton('#ad_scooping_slider');
            disableButton('#ad_feeding_sense_min');
            disableButton('#ad_feeding_sense_max');
            disableButton('#ad_feeding_slider');
        }

    });
    $('#man_task_stop').click(function(){
        assistive_teleop.manTask.stop();
        enableButton('#man_task_Scooping');
        enableButton('#man_task_Feeding');
        enableButton('#man_task_Init');
        enableButton('#man_task_stop');
        enableButton('#man_task_Continue');
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
 
    });
    $('#man_task_Continue').click(function(){
        if (assistive_teleop.manTask.continue_()) {
            disableButton('#man_task_Scooping');
            disableButton('#man_task_Feeding');
            disableButton('#man_task_Init');
            enableButton('#man_task_stop');
            disableButton('#man_task_Continue');
            disableButton('#man_task_success');
            disableButton('#man_task_Fail');
            disableButton('#man_task_start');
            disableButton('#man_task_Skip');

            disableButton('#ad_scooping_sense_min');
            disableButton('#ad_scooping_sense_max');
            disableButton('#ad_scooping_slider');
            disableButton('#ad_feeding_sense_min');
            disableButton('#ad_feeding_sense_max');
            disableButton('#ad_feeding_slider');
        }
    });
    $('#man_task_success').click(function(){
        assistive_teleop.manTask.success();
        enableButton('#man_task_Scooping');
        enableButton('#man_task_Feeding');
        enableButton('#man_task_Init');
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

    });
    $('#man_task_Fail').click(function(){
        assistive_teleop.manTask.failure();
        enableButton('#man_task_Scooping');
        enableButton('#man_task_Feeding');
        enableButton('#man_task_Init');
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
    });

    $('#man_task_Skip').click(function(){
        assistive_teleop.manTask.skip();
        enableButton('#man_task_Scooping');
        enableButton('#man_task_Feeding');
        enableButton('#man_task_Init');
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
    });


}
