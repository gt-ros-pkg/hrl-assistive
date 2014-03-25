var ArServo = function (ros) {
  'use strict';
  var arServo = this;
  arServo.SERVO_FEEDBACK_TOPIC = 'pr2_ar_servo/state_feedback';
  arServo.SERVO_APPROACH_TOPIC = 'pr2_ar_servo/tag_confirm';
  arServo.SERVO_PREEMPT_TOPIC = 'pr2_ar_servo/preempt';
  arServo.SERVO_CONFIRM_IMG_TOPIC = 'ar_servo/confirmation_rotated';


  arServo.ros = ros;
  arServo.state = 0;

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
  window.arServo = new ArServo(window.ros);
  var divRef = '#'+tabDivId

  $("#tabs").on("tabsbeforeactivate", function (event, ui) {
    if (ui.newPanel.selector === divRef) {
      window.mjpeg.setCamera(window.arServo.SERVO_CONFIRM_IMG_TOPIC);
    };
  });

  $(divRef).css({"position":"relative"});
  $(divRef).append('<table><tr>' +
                           '<td id="' + tabDivId + 'R0C0"></td>' +
                           '<td id="' + tabDivId + 'R0C1"></td>' +
                           '</tr></table>');
  $(divRef+'R0C0').append('<button id="' + tabDivId + '_approach">' +
                                    'Approach </button>')
    .click(function () {
      window.arServo.approach();
    });

  $(divRef+'R0C1').append('<button id="' + tabDivId + '_preempt">' +
                                    'Stop </button>')
    .click(function () {
      window.arServo.preemptApproach();
    });
  $(divRef+' :button').button().css({
    'height': "75px",
    'width': "200px",
    'font-size': '150%',
    'text-align':"center"
  });
  $(divRef+'_approach'+',#'+tabDivId+'_preempt').show().fadeTo(0, 0.5);

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
      $('#'+tabDivId+'_approach').show().fadeTo(0, 1)
      $('#'+tabDivId+'_preempt').show().fadeTo(0, 0.5);
      window.mjpeg.setCamera(window.arServo.SERVO_CONFIRM_IMG_TOPIC);
      break;
    case 3:
      text = "Unable to Locate AR Tag. ADJUST VIEW AND RETRY.";
      $('#'+tabDivId+'_approach').show().fadeTo(0, 0.5)
      $('#'+tabDivId+'_preempt').show().fadeTo(0, 0.5);
      window.mjpeg.setCamera(window.arServo.SERVO_CONFIRM_IMG_TOPIC);
      break;
    case 4:
      text = "Servoing";
      $('#'+tabDivId+'_approach').show().fadeTo(0, 0.5)
      $('#'+tabDivId+'_preempt').show().fadeTo(0, 1);
      break;
    case 5:
      text = "Servoing Completed Successfully.";
      $('#'+tabDivId+'_approach'+', #'+tabDivId+'_preempt').show().fadeTo(0, 0.5);
      window.mjpeg.setCamera('/head_mount_kinect/rgb/image_color');
      break;
    case 6:
      text = "Detected Collision with arms. Servoing Stopped.";
      $('#'+tabDivId+'_approach').show().fadeTo(0, 1)
      $('#'+tabDivId+'_preempt').show().fadeTo(0, 0.5);
      break;
    case 7:
      text = "Detected Collision with base laser. Servoing Stopped.";
      $('#'+tabDivId+'_approach').show().fadeTo(0, 1)
      $('#'+tabDivId+'_preempt').show().fadeTo(0, 0.5);
      break;
    case 8:
      text = "View of AR Tag lost. Servoing Stopped.";
      $('#'+tabDivId+'_approach'+', #'+tabDivId+'_preempt').show().fadeTo(0, 0.5);
      window.mjpeg.setCamera(window.arServo.SERVO_CONFIRM_IMG_TOPIC);
      break;
    case 9:
      text = "Servoing Stopped by User.";
      $('#'+tabDivId+'_approach').show().fadeTo(0, 1)
      $('#'+tabDivId+'_preempt').show().fadeTo(0, 0.5);
      window.mjpeg.setCamera(window.arServo.SERVO_CONFIRM_IMG_TOPIC);
      break;
    }
    log(text);
  };
  window.arServo.stateCBList.push(arServoFeedbackCb);
}
