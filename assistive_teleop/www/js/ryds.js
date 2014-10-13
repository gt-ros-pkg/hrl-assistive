var RYDS = function (ros) {
    'use strict';
    var ryds = this;
    ryds.ros = ros;
    //RYDS Topics
    ryds.USER_INPUT_TOPIC = "user_input";
    ryds.EMERGENCY_TOPIC = "emergency";

    //Bowl Topics
    ryds.REG_CONFIRM_CAMERA = 'Head Registration'
    ryds.BOWL_REGISTRATION_TOPIC = "RYDS_Action";
    ryds.BOWL_LOCATION_TOPIC = "RYDS_CupLocation";
    ryds.BOWL_CONFIRMATION_TOPIC = "RYDS_BowlConfirmation";
    ryds.CONFIRM_TOPIC="RYDS_Confirm";

    //Bowl Registration Service     
    ryds.bowlRegServiceClient = new ryds.ros.Service({
        name:'/finding_bowl_service',
        serviceType:'cup_finder/CupFinder'///Pixel23d'
    });

    ryds.RegisterBowl = function (u, v) {
        assistive_teleop.log("Made it this far");
        ryds.bowlRegServiceClient.callService({u:u,v:v}, function (resp) {
            console.log('Initialize Bowl Registration Service Returned Success: '+resp.success);
            console.log(resp);
            $('#img_act_select').val('looking');
            if (resp.success){
                $("#confirm_reg").button({disabled: false });
            }
        });       
    };

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

    //Bowl Registration Publishers and Subscribers
    //Add subscriber for Continue button
    ryds.bowlRegistrationPub = new ryds.ros.Topic({
        name: ryds.BOWL_REGISTRATION_TOPIC,
        messageType : 'std_msgs/String'
    });
    ryds.bowlRegistrationPub.advertise();

    ryds.bowlConfirmPub = new ryds.ros.Topic({
        name: ryds.CONFIRM_TOPIC,
        messageType : 'std_msgs/String'
    });
    ryds.bowlConfirmPub.advertise();

    ryds.BowlLocationSub = new ryds.ros.Topic({
        name: ryds.BOWL_LOCATION_TOPIC,
        messageType : 'geometry_msgs/PoseStamped'
    });   
    
    //Callback for Bowl System
    ryds.BowlLocationSub.subscribe(function(msg) {
       ryds.finalPose = msg
       $("#confirm_bowl_reg").button({ disabled: false });
       assistive_teleop.log("Press 'Confirm Registration' to complete registration process");
       //ryds.BowlLocationSub.unsubscribe();
    });

    ryds.BowlConfirmationPub = new ryds.ros.Topic({
        name: ryds.BOWL_CONFIRMATION_TOPIC,
        messageType : 'geometry_msgs/PoseStamped'
    });
    ryds.BowlConfirmationPub.advertise() 

    ryds.bowlRegInit = function () {
        var msg = new ryds.ros.Message({
            data : "RYDS_FindingCup"
        });
        $('#img_act_select').val('BowlReg');
        ryds.bowlRegistrationPub.publish(msg);
        assistive_teleop.log("Click on the bowl to begin bowl registration");
        console.log('Publishing start message to Bowl registration system');
    }
    
    ryds.confirmBowlRegistration = function () {
        var msg = new ryds.ros.Message({
            data : "RYDS_BowlRegConfirm"
        });
        $('#img_act_select').val('looking');
        ryds.bowlRegistrationPub.publish(msg);
        ryds.bowlConfirmPub.publish(msg);
        assistive_teleop.log("Bowl registration confirmed");
        var i = 0;
        while (i<10) {
            ryds.BowlConfirmationPub.publish(ryds.finalPose);
            i=i+1;
        }
    }

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
                   "9. After you have registered your head. Click on the 'RYDS' tab again. </br>"
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
    $(divRef+'_R0C0').append('<button class="centered" id="reg_bowl"> Register Bowl </button>');
    $("#reg_bowl").button();
    $("#reg_bowl").attr("title", "Click to begin bowl registration.");
    $(divRef+'_R0C0').click(assistive_teleop.ryds.bowlRegInit);

    //Confirm Bowl Registration Button
    //Bowl Confirmation Button
    $(divRef+'_R0C1').append('<button class="centered" id="confirm_bowl_reg"> Confirm Registration</button>');
    $("#confirm_bowl_reg").button({disabled: true }); //true });
    $("#confirm_bowl_reg").attr("title", "Click to confirm that bowl registration is correct.");
    $(divRef+'_R0C1').click(assistive_teleop.ryds.confirmBowlRegistration);
    
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
