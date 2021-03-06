var ArServo = function (ros) {
  'use strict';
  var arServo = this;
  arServo.SERVO_FEEDBACK_TOPIC = 'pr2_ar_servo/state_feedback';
  arServo.SERVO_APPROACH_TOPIC = 'pr2_ar_servo/tag_confirm';
  arServo.SERVO_PREEMPT_TOPIC = 'pr2_ar_servo/preempt';
  arServo.TEST_GOAL_TOPIC = 'ar_servo_goal_data';
  arServo.TEST_TAG_TOPIC = 'r_pr2_ar_pose_marker';
  arServo.SERVO_CONFIRM_CAMERA = 'AR Tag';

  arServo.ros = ros;
  arServo.state = 0;

  arServo.ros.getMsgDetails('hrl_pr2_ar_servo/ARServoGoalData');
  arServo.testPub = new arServo.ros.Topic({
    name: arServo.TEST_GOAL_TOPIC,
    messageType: 'hrl_pr2_ar_servo/ARServoGoalData'
  });
  arServo.testPub.advertise();
  arServo.sendTestGoal = function () {
    var msg = new arServo.ros.composeMsg('hrl_pr2_ar_servo/ARServoGoalData');
    console.log(msg);
    msg.marker_topic = arServo.TEST_TAG_TOPIC; 
    msg.tag_goal_pose.header.frame_id = "/base_link",
    msg.tag_goal_pose.pose.position.x = 1;
    msg.tag_goal_pose.pose.position.y = 0;
    msg.tag_goal_pose.pose.position.z = 0.6;
    msg.tag_goal_pose.pose.orientation.x = -0.5;// 0.70710678,
    msg.tag_goal_pose.pose.orientation.y = 0.5;//.70710678,
    msg.tag_goal_pose.pose.orientation.z = 0.5;//0.70710678,
    msg.tag_goal_pose.pose.orientation.w = -0.5//70710678,
    arServo.testPub.publish(msg);
  }

  arServo.approachPub = new arServo.ros.Topic({
    name: arServo.SERVO_APPROACH_TOPIC,
    messageType: 'std_msgs/Bool'
  });
  arServo.approachPub.advertise();
  arServo.approach = function () {
    var msg = new arServo.ros.Message({
      data: true
    });
    arServo.approachPub.publish(msg);
    console.log('Publishing ar_servo begin approach msg');
  };

  arServo.preemptApproachPub = new arServo.ros.Topic({
    name: arServo.SERVO_PREEMPT_TOPIC,
    messageType: 'std_msgs/Bool'
  });
  arServo.preemptApproachPub.advertise();
  arServo.preemptApproach = function () {
    var msg = new arServo.ros.Message({
      data: true
    });
    arServo.preemptApproachPub.publish(msg);
    console.log('Publishing ar_servo preempt approach msg');
  };

  arServo.stateCB = function (msg) {
    arServo.state = msg.data;
  };
  arServo.stateCBList = [arServo.stateCB];

  arServo.stateSub = new arServo.ros.Topic({
    name: arServo.SERVO_FEEDBACK_TOPIC,
    messageType: 'std_msgs/Int8'
  });
  arServo.stateSub.subscribe(function (msg) {
    var i = 0;
    for (i = 0; i < arServo.stateCBList.length; i += 1) {
      arServo.stateCBList[i](msg);
    }
  });
};

function initARServoTab(tabDivId) {
  'use strict';
  assistive_teleop.arServo = new ArServo(assistive_teleop.ros);
  var divRef = '#'+tabDivId

  $("#tabs").on("tabsbeforeactivate", function (event, ui) {
    if (ui.newPanel.selector === divRef) {
      assistive_teleop.mjpeg.setCamera(assistive_teleop.arServo.SERVO_CONFIRM_CAMERA);
    };
  });

  $(divRef).css({"position":"relative"});
  $(divRef).append('<table><tr>' +
                           '<td id="' + tabDivId + 'R0C0"></td>' +
                           '<td id="' + tabDivId + 'R0C1"></td>' +
                           '<td id="' + tabDivId + 'R0C2"></td>' +
                           '</tr></table>');
  $(divRef+'R0C0').append('<button id="' + tabDivId + '_approach">' +
                                    'Approach </button>')
    .click(function () {
      assistive_teleop.arServo.approach();
    });

  $(divRef+'R0C1').append('<button id="' + tabDivId + '_preempt">Stop </button>')
    .click(function () {
      assistive_teleop.arServo.preemptApproach();
    });

  $(divRef+'R0C2').append('<button id="'+tabDivId+'_test">Test</button>').click(function () {
      assistive_teleop.arServo.sendTestGoal();
      });
  $(divRef+' :button').button().css({
    'height': "75px",
    'width': "200px",
    'font-size': '150%',
    'text-align':"center"
  });
  $(divRef+'_approach'+',#'+tabDivId+'_preempt').show().button('option', 'disabled', true);

  $(divRef).append('<table id="' + tabDivId +
                   '_T0"><tr><td id="' + tabDivId +
                   '_R0C0"></td><td id="' + tabDivId +
                   '_R0C1"></td></tr></table>');
  $(divRef+'_T0').append('<tr><td id="' + tabDivId + '_R1C0"></td></tr>')

  // Info dialog box -- Pops up with instructions for using the body registration tab
  var INFOTEXT = "The Servoing Tab allows you to position the robot relative to an Augmented Reality (AR) Tag. </br>" +
                 "To servo to a location:</br></br>"+
                 "1. The robot must have a goal position.  This is provided by giving the robot a task.</br>"+
                 "2. Use the 'Default Controls' tab to bring the desired tag into view of the servoing camera.</br>" +
                 "3. When the tag is consistently highlighted in the camera, the interface will tell you to begin servoing. " +
                     "Press 'Approach'</br>"+
                 "4. The robot will approach the goal location based on the location of the AR Tag.</br>" +
                 "5. If something interrupts the servoing (collision with arm, lost view of tag, etc.)," +
                     " correct the problem and continue servoing by pressing 'Approach' again. </br>" +
                 "6. If you wish to stop servoing, press 'Stop.'  You may resume by pressing 'Approach' again. </br>" +
                 "7. When the goal has been reached, the interface will inform you that 'Servoing has completed successfully,' "+
                     "and the view will return to the head camera."

  $(divRef).append('<div id="'+tabDivId+'_infoDialog">' + INFOTEXT + '</div>');
  $(divRef+'_infoDialog').dialog({autoOpen:false,
                            buttons: [{text:"Ok", click:function(){$(this).dialog("close");}}],
                            modal:true,
                            title:"Servoing Info",
                            width:"70%"
                            });

  //Info button - brings up info dialog
  $(divRef).append('<button id="'+tabDivId+'_info"> Help </button>');
  $(divRef+'_info').button();
  $(divRef+'_info').click(function () { $(divRef+'_infoDialog').dialog("open"); } );
  $(divRef+'_info').css({"position":"absolute",
                          "top":"10px",
                          "right":"10px"});
  
  var arServoFeedbackCb = function (msg) {
    var text = "Unknown result from servoing feedback";
    switch (msg.data) {
    case 1:
      text = "Searching for AR Tag.";
      var idx = $("#tabs a[href='#tabServoing']").parent().index();
      $("#tabs").tabs("option", "active", idx);
      break;
    case 2:
      text = "AR Tag Found - Begin Approach.";
      $('#'+tabDivId+'_approach').button('option', 'disabled', false);
      $('#'+tabDivId+'_preempt').button('option', 'disabled', true);
      assistive_teleop.mjpeg.setCamera(assistive_teleop.arServo.SERVO_CONFIRM_CAMERA);
      break;
    case 3:
      text = "Unable to Locate AR Tag. ADJUST VIEW AND RETRY.";
      $('#'+tabDivId+'_approach').button('option', 'disabled', true);
      $('#'+tabDivId+'_preempt').button('option', 'disabled', true);
      assistive_teleop.mjpeg.setCamera(assistive_teleop.arServo.SERVO_CONFIRM_CAMERA);
      break;
    case 4:
      text = "Servoing";
      $('#'+tabDivId+'_approach').button('option', 'disabled', true);
      $('#'+tabDivId+'_preempt').button('option', 'disabled', false);
      break;
    case 5:
      text = "Servoing Completed Successfully.";
      $('#'+tabDivId+'_approach'+', #'+tabDivId+'_preempt').button('option', 'disabled', true);
      assistive_teleop.mjpeg.setCamera('Head');
      break;
    case 6:
      text = "Detected Collision with arms. Servoing Stopped.";
      $('#'+tabDivId+'_approach').button('option', 'disabled', false);
      $('#'+tabDivId+'_preempt').button('option', 'disabled', true);
      break;
    case 7:
      text = "Detected Collision with base laser. Servoing Stopped.";
      $('#'+tabDivId+'_approach').button('option', 'disabled', false);
      $('#'+tabDivId+'_preempt').button('option', 'disabled', true);
      break;
    case 8:
      text = "View of AR Tag lost. Servoing Stopped.";
      $('#'+tabDivId+'_approach'+', #'+tabDivId+'_preempt').button('option', 'disabled', true);
      assistive_teleop.mjpeg.setCamera(assistive_teleop.arServo.SERVO_CONFIRM_CAMERA);
      break;
    case 9:
      text = "Servoing Stopped by User.";
      $('#'+tabDivId+'_approach').show().button('option', 'disabled', false);
      $('#'+tabDivId+'_preempt').show().button('option', 'disabled', true);
      assistive_teleop.mjpeg.setCamera(assistive_teleop.arServo.SERVO_CONFIRM_CAMERA);
      break;
    }
    log(text);
  };
  assistive_teleop.arServo.stateCBList.push(arServoFeedbackCb);
}
