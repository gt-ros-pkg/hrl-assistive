var RYDS = function (ros) {
  'use strict';
  var ryds = this;
  ryds.ros = ros;
  ryds.USER_INPUT_TOPIC = "user_input";
  ryds.EMERGENCY_TOPIC = "emergency";

  ryds.userInputPub = new ryds.ros.Topic({
    name: ryds.USER_INPUT_TOPIC,
    messageType: 'std_msgs/String'
  });
  ryds.userInputPub.advertise();

  ryds.emergencyPub = new ryds.ros.Topic({
    name: ryds.EMERGENCY_TOPIC,
    messageType: 'std_msgs/String'
  });
  ryds.emergencyPub.advertise();

  ryds.start = function () {
    var msg = new ryds.ros.Message({
      data: 'Start'
    });
    ryds.userInputPub.publish(msg);
    console.log('Publishing Start msg to RYDS system.');
  };

  ryds.stop = function () {
    var msg = new ryds.ros.Message({
      data: 'STOP'
    });
    ryds.emergencyPub.publish(msg);
    console.log('Publishing Stop msg to RYDS system.');
  };

  ryds.continue_ = function () {
    var msg = new ryds.ros.Message({
      data: 'Continue'
    });
    ryds.userInputPub.publish(msg);
    console.log('Publishing Continue msg to RYDS system.');
  };

}


var initRYDSTab = function (tabDivId) {
  'use strict';
  assistive_teleop.ryds = new RYDS(assistive_teleop.ros);
  var divRef = '#'+tabDivId;

  $(divRef).append('<table><tr>' +
                     '<td id="' + tabDivId + 'R0C0"></td>' +
                     '<td id="' + tabDivId + 'R0C1"></td>' +
                     '<td id="' + tabDivId + 'R0C2"></td>' +
                     '</tr></table>');
  $(divRef+'R0C0').append('<button id="' + tabDivId + '_start">' +
                                    'Start </button>')
    .click(function () {
      assistive_teleop.ryds.start();
    });

  $(divRef+'R0C1').append('<button id="' + tabDivId + '_stop">' +
                                    'Stop </button>')
    .click(function () {
      assistive_teleop.ryds.stop();
    });

  $(divRef+'R0C2').append('<button id="' + tabDivId + '_continue">' +
                                    'Continue </button>')
    .click(function () {
      assistive_teleop.ryds.continue_();
    });

  $(divRef+' :button').button().css({
    'height': "75px",
    'width': "200px",
    'font-size': '150%',
    'text-align':"center"
  });
}
