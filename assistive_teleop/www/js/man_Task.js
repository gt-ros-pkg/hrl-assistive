var ManipulationTask = function (ros) {
    'use strict';
    var manTask = this;
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
        var msg = new manTask.ros.Message({
          data: 'Scooping'
        });
        assistive_teleop.log('Please, follow the step 2 to select the action.');
        manTask.statusPub.publish(msg);
    };

    manTask.feed = function () {
        var msg = new manTask.ros.Message({
          data: 'Feeding'
        });
        assistive_teleop.log('Please, follow the step 2 to select the action.');
        manTask.statusPub.publish(msg);
    };

    manTask.both = function () {
        var msg = new manTask.ros.Message({
          data: 'Init'
        });
        assistive_teleop.log('Please, follow the step 2 to select the action.');
        manTask.statusPub.publish(msg);
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
        var msg = new manTask.ros.Message({
          data: 'Start'
        });
        manTask.userInputPub.publish(msg);
        assistive_teleop.log('Starting the manipulation task');
        assistive_teleop.log('Please, follow the step 3 when "Requesting Feedback" message shows up.');
        console.log('Publishing Start msg to manTask system.');
    };

    manTask.stop = function () {
        var msg = new manTask.ros.Message({
          data: 'STOP'
        });
        manTask.emergencyPub.publish(msg);
        assistive_teleop.log('Stopping the manipulation task');
        assistive_teleop.log('Please, press "Continue" to re-start the action. Or re-start from step 1.');
        console.log('Publishing Stop msg to manTask system.');
    };

    manTask.continue_ = function () {
        var msg = new manTask.ros.Message({
          data: 'Continue'
        });
        manTask.userInputPub.publish(msg);
        assistive_teleop.log('Continuing the manipulation task');
        assistive_teleop.log('Please, follow the step 3 when "Requesting Feedback" message shows up.');
        console.log('Publishing Continue msg to manTask system.');
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


    //part added.
    manTask.feedbackSub = new manTask.ros.Topic({
        name: 'manipulation_task/feedbackRequest',
        messageType: 'std_msgs/String'});
    manTask.feedbackSub.subscribe(function (msg) {
        assistive_teleop.log(msg.data);
        if(msg.data=="Requesting Feedback!") {
        //assistive_teleop.log("worked?");
        $('#man_task_Scooping').css("opacity","0.6");
        $('#man_task_Feeding').css("opacity","0.6");
        $('#man_task_Init').css("opacity","0.6");
        $('#man_task_stop').css("opacity","0.6");
        $('#man_task_Continue').css("opacity","0.6");
        $('#man_task_success').css("opacity","1.0");
        $('#man_task_Fail').css("opacity","1.0");
        $('#man_task_start').css("opacity","0.6");
        $('#man_task_Skip').css("opacity","1.0");
        // "pointer-events" "none" "auto"

        $('#man_task_Scooping').css("pointer-events","none");
        $('#man_task_Feeding').css("pointer-events","none");
        $('#man_task_Init').css("pointer-events","none");
        $('#man_task_stop').css("pointer-events","none");
        $('#man_task_Continue').css("pointer-events","none");
        $('#man_task_start').css("pointer-events","none");

        }

    });

    //part added on 4/7 to accomodate anomaly signal.
    manTask.feedbackSub = new manTask.ros.Topic({
        name: 'manipulation_task/emergency',
        messageType: 'std_msgs/String'});
    manTask.feedbackSub.subscribe(function (msg) {
        if(msg.data!="STOP") {

        $('#man_task_Scooping').css("opacity","1.0");
        $('#man_task_Feeding').css("opacity","1.0");
        $('#man_task_Init').css("opacity","1.0");
        $('#man_task_start').css("opacity","0.6");
        $('#man_task_Continue').css("opacity","1.0");
        $('#man_task_success').css("opacity","0.6");
        $('#man_task_Fail').css("opacity","0.6");
        $('#man_task_stop').css("opacity","0.6"); 
        $('#man_task_Skip').css("opacity","0.6");

        $('#man_task_Continue').css("pointer-events","auto");
        $('#man_task_start').css("pointer-events","auto");
        $('#man_task_stop').css("pointer-events","none");
 
        $('#ad_scooping_sense_min').css("pointer-events","auto");
        $('#ad_scooping_sense_max').css("pointer-events","auto");
        $('#ad_scooping_slider').css("pointer-events","auto");
        $('#ad_scooping_sense_min').css("opacity","1.0");
        $('#ad_scooping_sense_max').css("opacity","1.0");
        $('#ad_scooping_slider').css("opacity","1.0");

        $('#ad_feeding_sense_min').css("pointer-events","auto");
        $('#ad_feeding_sense_max').css("pointer-events","auto");
        $('#ad_feeding_slider').css("pointer-events","auto");
        $('#ad_feeding_sense_min').css("opacity","1.0");
        $('#ad_feeding_sense_max').css("opacity","1.0");
        $('#ad_feeding_slider').css("opacity","1.0");

        }

    });




};

var initManTaskTab = function() {
  assistive_teleop.manTask = new ManipulationTask(assistive_teleop.ros);
  assistive_teleop.log('initiating manipulation Task');

    $('#man_task_Scooping').click(function(){
        assistive_teleop.manTask.scoop();
        $('#man_task_Scooping').css("opacity","0.6");
        $('#man_task_Feeding').css("opacity","0.6");
        $('#man_task_Init').css("opacity","0.6");
        $('#man_task_stop').css("opacity","0.6");
        $('#man_task_Continue').css("opacity","0.6");
        $('#man_task_success').css("opacity","0.6");
        $('#man_task_Fail').css("opacity","0.6");
        $('#man_task_start').css("opacity","1.0");
        $('#man_task_Skip').css("opacity","0.6");

        $('#man_task_Continue').css("pointer-events","auto");
        $('#man_task_start').css("pointer-events","auto");


    });
    $('#man_task_Feeding').click(function(){
        assistive_teleop.manTask.feed();
        $('#man_task_Scooping').css("opacity","0.6");
        $('#man_task_Feeding').css("opacity","0.6");
        $('#man_task_Init').css("opacity","0.6");
        $('#man_task_stop').css("opacity","0.6");
        $('#man_task_Continue').css("opacity","0.6");
        $('#man_task_success').css("opacity","0.6");
        $('#man_task_Fail').css("opacity","0.6");
        $('#man_task_start').css("opacity","1.0");
        $('#man_task_Skip').css("opacity","0.6");

        $('#man_task_Continue').css("pointer-events","auto");
        $('#man_task_start').css("pointer-events","auto"); 

    });
    $('#man_task_Init').click(function(){
        assistive_teleop.manTask.both();
        $('#man_task_Scooping').css("opacity","0.6");
        $('#man_task_Feeding').css("opacity","0.6");
        $('#man_task_Init').css("opacity","0.6");
        $('#man_task_stop').css("opacity","0.6");
        $('#man_task_Continue').css("opacity","0.6");
        $('#man_task_success').css("opacity","0.6");
        $('#man_task_Fail').css("opacity","0.6"); 
        $('#man_task_start').css("opacity","1.0");
        $('#man_task_Skip').css("opacity","0.6");

        $('#man_task_Continue').css("pointer-events","auto");
        $('#man_task_start').css("pointer-events","auto");
 
    });
    $('#man_task_start').click(function(){
        assistive_teleop.manTask.start();
        $('#man_task_Scooping').css("opacity","0.6");
        $('#man_task_Feeding').css("opacity","0.6");
        $('#man_task_Init').css("opacity","0.6");
        $('#man_task_start').css("opacity","0.6");
        $('#man_task_Continue').css("opacity","0.6");
        $('#man_task_success').css("opacity","0.6");
        $('#man_task_Fail').css("opacity","0.6");
        $('#man_task_stop').css("opacity","1.0"); 
        $('#man_task_Skip').css("opacity","0.6");

        $('#man_task_Continue').css("pointer-events","none");
        $('#man_task_start').css("pointer-events","none");
        $('#man_task_stop').css("pointer-events","auto"); 

        $('#ad_scooping_sense_min').css("pointer-events","auto");
        $('#ad_scooping_sense_max').css("pointer-events","auto");
        $('#ad_scooping_slider').css("pointer-events","auto");
        $('#ad_scooping_sense_min').css("opacity","0.6");
        $('#ad_scooping_sense_max').css("opacity","0.6");
        $('#ad_scooping_slider').css("opacity","0.6");

        $('#ad_feeding_sense_min').css("pointer-events","auto");
        $('#ad_feeding_sense_max').css("pointer-events","auto");
        $('#ad_feeding_slider').css("pointer-events","auto");
        $('#ad_feeding_sense_min').css("opacity","0.6");
        $('#ad_feeding_sense_max').css("opacity","0.6");
        $('#ad_feeding_slider').css("opacity","0.6");


    });
    $('#man_task_stop').click(function(){
        assistive_teleop.manTask.stop();
        $('#man_task_Scooping').css("opacity","1.0");
        $('#man_task_Feeding').css("opacity","1.0");
        $('#man_task_Init').css("opacity","1.0");
        $('#man_task_start').css("opacity","0.6");
        $('#man_task_Continue').css("opacity","1.0");
        $('#man_task_success').css("opacity","0.6");
        $('#man_task_Fail').css("opacity","0.6");
        $('#man_task_stop').css("opacity","0.6"); 
        $('#man_task_Skip').css("opacity","0.6");

        $('#man_task_Continue').css("pointer-events","auto");
        $('#man_task_start').css("pointer-events","auto");
        $('#man_task_stop').css("pointer-events","none");

        $('#ad_scooping_sense_min').css("pointer-events","auto");
        $('#ad_scooping_sense_max').css("pointer-events","auto");
        $('#ad_scooping_slider').css("pointer-events","auto");
        $('#ad_scooping_sense_min').css("opacity","1.0");
        $('#ad_scooping_sense_max').css("opacity","1.0");
        $('#ad_scooping_slider').css("opacity","1.0");

        $('#ad_feeding_sense_min').css("pointer-events","auto");
        $('#ad_feeding_sense_max').css("pointer-events","auto");
        $('#ad_feeding_slider').css("pointer-events","auto");
        $('#ad_feeding_sense_min').css("opacity","1.0");
        $('#ad_feeding_sense_max').css("opacity","1.0");
        $('#ad_feeding_slider').css("opacity","1.0");
 
    });
    $('#man_task_Continue').click(function(){
        assistive_teleop.manTask.continue_();
        $('#man_task_Scooping').css("opacity","0.6");
        $('#man_task_Feeding').css("opacity","0.6");
        $('#man_task_Init').css("opacity","0.6");
        $('#man_task_start').css("opacity","0.6");
        $('#man_task_Continue').css("opacity","0.6");
        $('#man_task_success').css("opacity","0.6");
        $('#man_task_Fail').css("opacity","0.6");
        $('#man_task_stop').css("opacity","1.0"); 
        $('#man_task_Skip').css("opacity","0.6");

        $('#man_task_Continue').css("pointer-events","none");
        $('#man_task_start').css("pointer-events","none");
        $('#man_task_stop').css("pointer-events","auto");

        $('#ad_scooping_sense_min').css("pointer-events","auto");
        $('#ad_scooping_sense_max').css("pointer-events","auto");
        $('#ad_scooping_slider').css("pointer-events","auto");
        $('#ad_scooping_sense_min').css("opacity","0.6");
        $('#ad_scooping_sense_max').css("opacity","0.6");
        $('#ad_scooping_slider').css("opacity","0.6");

        $('#ad_feeding_sense_min').css("pointer-events","auto");
        $('#ad_feeding_sense_max').css("pointer-events","auto");
        $('#ad_feeding_slider').css("pointer-events","auto");
        $('#ad_feeding_sense_min').css("opacity","0.6");
        $('#ad_feeding_sense_max').css("opacity","0.6");
        $('#ad_feeding_slider').css("opacity","0.6");
 
    });
    $('#man_task_success').click(function(){
        assistive_teleop.manTask.success();
        $('#man_task_Scooping').css("opacity","1.0");
        $('#man_task_Feeding').css("opacity","1.0");
        $('#man_task_Init').css("opacity","1.0");
        $('#man_task_start').css("opacity","0.6");
        $('#man_task_Continue').css("opacity","0.6");
        $('#man_task_success').css("opacity","0.6");
        $('#man_task_Fail').css("opacity","0.6");
        $('#man_task_stop').css("opacity","0.6"); 
        $('#man_task_Skip').css("opacity","0.6");

    //re-enable click
        $('#man_task_Scooping').css("pointer-events","auto");
        $('#man_task_Feeding').css("pointer-events","auto");
        $('#man_task_Init').css("pointer-events","auto");
        $('#man_task_stop').css("pointer-events","auto");
        $('#man_task_Continue').css("pointer-events","auto");
        $('#man_task_start').css("pointer-events","auto");

        $('#ad_scooping_sense_min').css("pointer-events","auto");
        $('#ad_scooping_sense_max').css("pointer-events","auto");
        $('#ad_scooping_slider').css("pointer-events","auto");
        $('#ad_scooping_sense_min').css("opacity","1.0");
        $('#ad_scooping_sense_max').css("opacity","1.0");
        $('#ad_scooping_slider').css("opacity","1.0");

        $('#ad_feeding_sense_min').css("pointer-events","auto");
        $('#ad_feeding_sense_max').css("pointer-events","auto");
        $('#ad_feeding_slider').css("pointer-events","auto");
        $('#ad_feeding_sense_min').css("opacity","1.0");
        $('#ad_feeding_sense_max').css("opacity","1.0");
        $('#ad_feeding_slider').css("opacity","1.0");

    });
    $('#man_task_Fail').click(function(){
        assistive_teleop.manTask.failure();
        $('#man_task_Scooping').css("opacity","1.0");
        $('#man_task_Feeding').css("opacity","1.0");
        $('#man_task_Init').css("opacity","1.0");
        $('#man_task_start').css("opacity","0.6");
        $('#man_task_Continue').css("opacity","0.6");
        $('#man_task_success').css("opacity","0.6");
        $('#man_task_Fail').css("opacity","0.6");
        $('#man_task_stop').css("opacity","0.6");  
        $('#man_task_Skip').css("opacity","0.6");

    //re-enable click
        $('#man_task_Scooping').css("pointer-events","auto");
        $('#man_task_Feeding').css("pointer-events","auto");
        $('#man_task_Init').css("pointer-events","auto");
        $('#man_task_stop').css("pointer-events","auto");
        $('#man_task_Continue').css("pointer-events","auto");
        $('#man_task_start').css("pointer-events","auto");

        $('#ad_scooping_sense_min').css("pointer-events","auto");
        $('#ad_scooping_sense_max').css("pointer-events","auto");
        $('#ad_scooping_slider').css("pointer-events","auto");
        $('#ad_scooping_sense_min').css("opacity","1.0");
        $('#ad_scooping_sense_max').css("opacity","1.0");
        $('#ad_scooping_slider').css("opacity","1.0");

        $('#ad_feeding_sense_min').css("pointer-events","auto");
        $('#ad_feeding_sense_max').css("pointer-events","auto");
        $('#ad_feeding_slider').css("pointer-events","auto");
        $('#ad_feeding_sense_min').css("opacity","1.0");
        $('#ad_feeding_sense_max').css("opacity","1.0");
        $('#ad_feeding_slider').css("opacity","1.0");

    });

    $('#man_task_Skip').click(function(){
        assistive_teleop.manTask.skip();
        $('#man_task_Scooping').css("opacity","1.0");
        $('#man_task_Feeding').css("opacity","1.0");
        $('#man_task_Init').css("opacity","1.0");
        $('#man_task_start').css("opacity","0.6");
        $('#man_task_Continue').css("opacity","0.6");
        $('#man_task_success').css("opacity","0.6");
        $('#man_task_Fail').css("opacity","0.6");
        $('#man_task_stop').css("opacity","0.6");  
        $('#man_task_Skip').css("opacity","0.6");
    //re-enable click
        $('#man_task_Scooping').css("pointer-events","auto");
        $('#man_task_Feeding').css("pointer-events","auto");
        $('#man_task_Init').css("pointer-events","auto");
        $('#man_task_stop').css("pointer-events","auto");
        $('#man_task_Continue').css("pointer-events","auto");
        $('#man_task_start').css("pointer-events","auto");

        $('#ad_scooping_sense_min').css("pointer-events","auto");
        $('#ad_scooping_sense_max').css("pointer-events","auto");
        $('#ad_scooping_slider').css("pointer-events","auto");
        $('#ad_scooping_sense_min').css("opacity","1.0");
        $('#ad_scooping_sense_max').css("opacity","1.0");
        $('#ad_scooping_slider').css("opacity","1.0");

        $('#ad_feeding_sense_min').css("pointer-events","auto");
        $('#ad_feeding_sense_max').css("pointer-events","auto");
        $('#ad_feeding_slider').css("pointer-events","auto");
        $('#ad_feeding_sense_min').css("opacity","1.0");
        $('#ad_feeding_sense_max').css("opacity","1.0");
        $('#ad_feeding_slider').css("opacity","1.0");

    });


}
