var ArServo = function () {
  'use strict';
  var arServo = this;
  arServo.state = 0;
  arServo.findTagPub = new window.ros.Topic({
    name: 'pr2_ar_servo/find_tag',
    messageType: 'std_msgs/Bool'
  });
  arServo.findTagPub.advertise();
  arServo.detectTag = function () {
    var msg = new window.ros.Message({
      data: true
    });
    arServo.findTagPub.publish(msg);
    console.log('Publishing ar_servo detect tag msg');
  };

  arServo.approachPub = new window.ros.Topic({
    name: 'pr2_ar_servo/tag_confirm',
    messageType: 'std_msgs/Bool'
  });
  arServo.approachPub.advertise();
  arServo.approach = function () {
    var msg = new window.ros.Message({
      data: true
    });
    arServo.approachPub.publish(msg);
    console.log('Publishing ar_servo begin approach msg');
  };

  arServo.preemptApproachPub = new window.ros.Topic({
    name: 'pr2_ar_servo/preempt',
    messageType: 'std_msgs/Bool'
  });
  arServo.preemptApproachPub.advertise();
  arServo.preemptApproach = function () {
    var msg = new window.ros.Message({
      data: true
    });
    arServo.preemptApproachPub.publish(msg);
    console.log('Publishing ar_servo preempt approach msg');
  };

  arServo.stateCB = function (msg) {
    arServo.state = msg.data;
  };
  arServo.stateCBList = [arServo.stateCB];

  arServo.stateSub = new window.ros.Topic({
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
  window.arServo = new ArServo();
  $('#' + tabDivId).append('<table><tr>' +
                           '<td id="' + tabDivId + 'R0C0"></td>' +
                           '<td id="' + tabDivId + 'R0C1"></td>' +
                           '<td id="' + tabDivId + 'R0C2"></td>' +
                           '</tr></table>');
  $('#' + tabDivId + 'R0C0').append('<button id="' + tabDivId + 'DetectTag">' +
                                    ' Detect Tag </button>')
    .click(function () {
      window.arServo.detectTag();
    });

  $('#' + tabDivId + 'R0C1').append('<button id="' + tabDivId + 'Approach">' +
                                    ' Approach </button>')
    .click(function () {
      window.arServo.approach();
    });
  $('#' + tabDivId + 'R0C2').append('<button id="' + tabDivId + 'Preempt">' +
                                    ' Stop </button>')
    .click(function () {
      window.arServo.preemptApproach();
    });
  $('#' + tabDivId + ' :button').button().css({
    'height': "75px",
    'width': "150px",
    'font-size': '150%'
  });

  var arServoFeedbackCb = function (msg) {
    var text = "Unknown result from servoing feedback";
    switch (msg.data) {
    case 1:
      text = "Searching for AR Tag.";
      break;
    case 2:
      text = "AR Tag Found. CONFIRM LOCATION AND BEGIN APPROACH.";
      $('#servo_approach, #servo_stop').show().fadeTo(0, 1);
      $('#servo_detect_tag').fadeTo(0, 0.5);
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
      $('#servo_approach, #servo_stop').fadeTo(0, 0.5);
      $('#servo_detect_tag').fadeTo(0, 1);
      window.mjpeg.setCamera('head_mount_kinect/rgb/image_color');
      break;
    case 6:
      text = "Detected Collision with Arms while Servoing.  " + "ADJUST AND RE-DETECT TAG.";
      $('#servo_approach, #servo_stop').fadeTo(0, 0.5);
      $('#servo_detect_tag').fadeTo(0, 1);
      break;
    case 7:
      text = "Detected Collision in Base Laser while Servoing.  " + "ADJUST AND RE-DETECT TAG.";
      $('#servo_approach, #servo_stop').fadeTo(0, 0.5);
      $('#servo_detect_tag').fadeTo(0, 1);
      break;
    case 8:
      text = "View of AR Tag Was Lost.  ADJUST (IF NECESSARY) AND RE-DETECT.";
      $('#servo_approach, #servo_stop').fadeTo(0, 0.5);
      $('#servo_detect_tag').fadeTo(0, 1);
      window.mjpeg.setCamera('ar_servo/confirmation_rotated');
      break;
    case 9:
      text = "Servoing Stopped by User. RE-DETECT TAG";
      $('#servo_approach, #servo_stop').fadeTo(0, 0.5);
      $('#servo_detect_tag').fadeTo(0, 1);
      window.mjpeg.setCamera('ar_servo/confirmation_rotated');
      break;
    }
    log(text);
  };
  window.arServo.stateCBList.push(arServoFeedbackCb);
}
