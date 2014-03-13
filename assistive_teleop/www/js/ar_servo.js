var ArServo = function (ros) {
  'use strict';
  var arServo = this;
  arServo.ros = ros;
  arServo.state = 0;

  arServo.approachPub = new arServo.ros.Topic({
    name: 'pr2_ar_servo/tag_confirm',
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
    name: 'pr2_ar_servo/preempt',
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
    name: 'pr2_ar_servo/state_feedback',
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
  $('#' + tabDivId).append('<table><tr>' +
                           '<td id="' + tabDivId + 'R0C0"></td>' +
                           '<td id="' + tabDivId + 'R0C1"></td>' +
                           '</tr></table>');
  $('#' + tabDivId + 'R0C0').append('<button id="' + tabDivId + '_approach">' +
                                    'Approach </button>')
    .click(function () {
      window.arServo.approach();
    });

  $('#' + tabDivId + 'R0C1').append('<button id="' + tabDivId + '_preempt">' +
                                    'Stop </button>')
    .click(function () {
      window.arServo.preemptApproach();
    });
  $('#' + tabDivId + ' :button').button().css({
    'height': "75px",
    'width': "200px",
    'font-size': '150%',
    'text-align':"center"
  });
  $('#'+tabDivId+'_approach'+',#'+tabDivId+'_preempt').show().fadeTo(0, 0.5);

  var arServoFeedbackCb = function (msg) {
    var text = "Unknown result from servoing feedback";
    switch (msg.data) {
    case 1:
      text = "Searching for AR Tag.";
      break;
    case 2:
      text = "AR Tag Found - Begin Approach.";
      $('#'+tabDivId+'_approach'+', #'+tabDivId+'_preempt').show().fadeTo(0, 1);
      window.mjpeg.setCamera('ar_servo/confirmation_rotated');
      break;
    case 3:
      text = "Unable to Locate AR Tag. ADJUST VIEW AND RETRY.";
      window.mjpeg.setCamera('ar_servo/confirmation_rotated');
      break;
    case 4:
      text = "Servoing";
      break;
    case 5:
      text = "Servoing Completed Successfully.";
      $('#'+tabDivId+'_approach'+', #'+tabDivId+'_preempt').show().fadeTo(0, 0.5);
      window.mjpeg.setCamera('/head_mount_kinect/rgb/image_color');
      break;
    case 6:
      text = "Detected Collision with Arms while Servoing.  " + "ADJUST AND RE-DETECT TAG.";
      $('#'+tabDivId+'_approach'+', #'+tabDivId+'_preempt').show().fadeTo(0, 0.5);
      break;
    case 7:
      text = "Detected Collision in Base Laser while Servoing.  " + "ADJUST AND RE-DETECT TAG.";
      $('#'+tabDivId+'_approach'+', #'+tabDivId+'_preempt').show().fadeTo(0, 0.5);
      break;
    case 8:
      text = "View of AR Tag Was Lost.  ADJUST (IF NECESSARY) AND RE-DETECT.";
      $('#'+tabDivId+'_approach'+', #'+tabDivId+'_preempt').show().fadeTo(0, 0.5);
      window.mjpeg.setCamera('ar_servo/confirmation_rotated');
      break;
    case 9:
      text = "Servoing Stopped by User. RE-DETECT TAG";
      $('#'+tabDivId+'_approach'+', #'+tabDivId+'_preempt').show().fadeTo(0, 0.5);
      window.mjpeg.setCamera('ar_servo/confirmation_rotated');
      break;
    }
    log(text);
  };
  window.arServo.stateCBList.push(arServoFeedbackCb);
}
