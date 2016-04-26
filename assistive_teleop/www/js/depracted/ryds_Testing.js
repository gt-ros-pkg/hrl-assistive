var RYDS = function (ros) {
    'use strict';
    var ryds = this;
    ryds.ros = ros;
    //RYDS Topics
    ryds.USER_INPUT_TOPIC = "user_input";
    ryds.EMERGENCY_TOPIC = "emergency";

    //Feedback Topics
    ryds.FEEDBACK_TOPIC = "feedback";

    //Bowl Topics
    ryds.REG_CONFIRM_CAMERA = 'Head Registration'
    ryds.CONFIRM_TOPIC="RYDS_Confirm";


    //RYDS Publishers
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
        assistive_teleop.log('Starting yogurt feeding');
        console.log('Publishing Start msg to RYDS system.');
    };

    ryds.stop = function () {
        var msg = new ryds.ros.Message({
          data: 'STOP'
        });
        ryds.emergencyPub.publish(msg);
        assistive_teleop.log('Stopping yogurt feeding');
        console.log('Publishing Stop msg to RYDS system.');
    };

    ryds.continue_ = function () {
        var msg = new ryds.ros.Message({
          data: 'Continue'
        });
        ryds.userInputPub.publish(msg);
        assistive_teleop.log('Continuing yogurt feeding');
        console.log('Publishing Continue msg to RYDS system.');
    };


    //Feedback publisher and functions.
    ryds.feedbackPub = new ryds.ros.Topic({
        name: ryds.FEEDBACK_TOPIC,
        messageType: 'std_msgs/String'
    });
    ryds.feedbackPub.advertise();

    ryds.success = function () {
        var msg = new ryds.ros.Message({
          data: 'Success'
        });
        ryds.feedbackPub.publish(msg);
        assistive_teleop.log('Reporting result:Success');
        console.log('Publishing Success msg to RYDS system.');
    };

    ryds.fail = function () {
        var msg = new ryds.ros.Message({
          data: 'Fail'
        });
        ryds.feedbackPub.publish(msg);
        assistive_teleop.log('Reporting result:Fail');
        console.log('Publishing Fail msg to RYDS system.');
    };

}


var initRYDSTab = function (tabDivId) {
    'use strict';
    assistive_teleop.ryds = new RYDS(assistive_teleop.ros);
    var divRef = '#'+tabDivId;

    $("#tabs").on("tabsbeforeactivate", function (event, ui) {
      if (ui.newPanel.selector === divRef) {
        assistive_teleop.mjpeg.setCamera(assistive_teleop.ryds.REG_CONFIRM_CAMERA);
      }
    });

    //Set up table for buttons
    $(divRef).css({"position":"relative"});
    $(divRef).append('<table id="' + tabDivId +
                     '_T0"><tr><td id="' + tabDivId + '_R0C0"></td><td id="' + tabDivId + '_R0C1"></td><td id="' + tabDivId + '_R0C2"></td></tr></table>');
    
    $(divRef+'_T0').append('<tr><td id="' + tabDivId + '_R1C0"></td><td id="'+ tabDivId + '_R1C1"></td><td id="' + tabDivId + '_R1C2"></td></tr>')

    //Info Dialogue Box
     var INFOTEXT = "The RYDS Tab allows you to use the robot to perform a feeding task.</br>" +
                   "To identify the bowl being used:</br></br>"+
                   "1. Have the robot look at the bowl using the head camera.</br>"+
                   "2. Select the 'Register Bowl' button.</br>" +
                   "3. Click on the bowl in the camera view.</br>"+
                   "4. Observe the overlaid points, showing where the robot finds the bowl. </br>" + 
                   "5. If the model does not line up with the bowl, try getting a better view of the bowl and repeat 2-4.</br>" + 
                   "6. If the model does line up with the bowl, click 'Confirm Registration' to confirm that it is correct.</br>" +
                   "7. After you have registered the bowl, click on the 'Body Registration' tab. </br>" +
                   "8. Register your head using the instructions provided. (See the help button on the 'Body Registration' tab) </br>" +
                   "9. After you have registered your head. Click on the 'RYDS' tab again. </br>"+
                   "10. Click the 'Start' button to begin the feeding task. </br>"+
                   "11. If you wish to continue the feeding task, click the 'Continue' button. </br>" +
                   "12. If you wish to stop the feeding task at any time, click the 'Stop' button. </br>"   

    $(divRef).append('<div id="'+tabDivId+'_infoDialog">' + INFOTEXT + '</div>');
    $(divRef+'_infoDialog').dialog({autoOpen:false,
                              buttons: [{text:"Ok", click:function(){$(this).dialog("close");}}],
                              modal:true,
                              title:"RYDS Info",
                              width:"70%"
                              });

    //Info button - brings up info dialog
    $(divRef).append('<button id="'+tabDivId+'_info"> Help </button>');
    $(divRef+'_info').button();
    $(divRef+'_info').click(function () { $(divRef+'_infoDialog').dialog("open"); } );
    $(divRef+'_info').css({"position":"absolute",
                            "top":"10px",
                            "right":"10px"});


    //Bowl Registration Button
    $(divRef+'_R0C0').append('<button class="centered" id="feedback_Success"> Success </button>');
    $("#feedback_Success").button();
    $("#feedback_Success").attr("title", "Click to reprot successful trial.");
    $(divRef+'_R0C0').click(assistive_teleop.ryds.success);

    //Confirm Bowl Registration Button
    //Bowl Confirmation Button
    $(divRef+'_R0C1').append('<button class="centered" id="feedback_fail"> Fail </button>');
    $("#feedback_fail").button({disabled: true }); //true });
    $("#feedback_fail").attr("title", "Click to reprot unsuccessful trial.");
    $(divRef+'_R0C1').click(assistive_teleop.ryds.fail);
    
    //Start Button
    $(divRef+'_R1C0').append('<button class="centered" id="RYDS_Start"> Start </button>')
    $("#RYDS_Start").button()
    $("#RYDS_Start").attr("title", "Click to begin yogurt feeding");
    $(divRef+'_R1C0').click(assistive_teleop.ryds.start);
      
    //Stop Button
    $(divRef+'_R1C1').append('<button class="centered" id="RYDS_Stop"> Stop </button>')
    $("#RYDS_Stop").button()
    $("#RYDS_Stop").attr("title", "Click to stop yogurt feeding");
    $(divRef+'_R1C1').click(assistive_teleop.ryds.stop);
    
    
    //Continue Button
    $(divRef+'_R1C2').append('<button class="centered" id="RYDS_Continue"> Continue </button>')
    $("#RYDS_Continue").button()
    $("#RYDS_Continue").attr("title", "Click to continue yogurt feeding");
    $(divRef+'_R1C2').click(assistive_teleop.ryds.continue_);    


    
    $(divRef+' :button').button().css({
      'height': "75px",
      'width': "200px",
      'font-size': '150%',
      'text-align':"center"

});
}
