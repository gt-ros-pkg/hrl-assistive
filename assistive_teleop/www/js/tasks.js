var TaskInterface = function (ros) {
    'use strict';
    var taskUI = this;
    taskUI.ros = ros;

    taskUI.sendTaskPub = new taskUI.ros.Topic({
      name: 'action_location_goal',
      messageType: 'std_msgs/String'
    });
    taskUI.sendTaskPub.advertise();

    taskUI.sendTask = function () {
      var loc = $("#task_location_select option:selected").text();
      var act = $("#task_action_select option:selected").text();
      var msg = new taskUI.ros.Message({
        data: loc
      });
      taskUI.sendTaskPub.publish(msg);
      var txt = "Sending Task: " + act + " the " + loc;
      window.log(txt);
    };
}

var initTaskInterface = function (tabDivId) {
    window.taskUI = new TaskInterface(window.ros);
    var divRef = "#"+tabDivId;
    $(divRef).css({"position":"relative"});
    $(divRef).append('<table id="' + tabDivId + '_T0">' +
                       '<tr>' + 
                         '<th id="' + tabDivId + '_R0C0">Action</th>' +
                         '<th id="' + tabDivId + '_R0C1">Location</th>' +
                       '</tr>' +
                       '<tr>' +
                         '<td id="' + tabDivId + '_R1C0"></td>' +
                         '<td id="' + tabDivId + '_R1C1"></td>' +
                       '</tr>' +
                       '<tr>' +
                         '<td id="' + tabDivId + '_R2C0" colspan="2"></td>' +
                       '</tr>' +
                     '</table>');
    $(divRef+'_R1C0').append('<select id="task_action_select">' + 
                               '<option>Touch</option>' +
                             '</select>');
    $(divRef+'_R1C1').append('<select id="task_location_select">' + 
                               '<option>Knee</option>' +
                               '<option>Arm</option>' +
                               '<option>Shoulder</option>' +
                               '<option>Face</option>' +
                             '</select>');
    $(divRef+'_R2C0').append('<button id="send_task"> Send Task </button>');
    $("#send_task").button()
    $(divRef+'_R2C0').click(window.taskUI.sendTask);

    // Info dialog box -- Pops up with instructions for using the body registration tab
    var INFOTEXT = "The Tasks Tab allows you to command the robot to do various tasks.</br>" +
                   "To send a task:</br></br>"+
                   "1. Select the action you would like the robot to perform.</br>"+
                   "2. Select the location where you would like the robot to perform this action.</br>" +
                   "3. Click 'Send Task.'</br></br>" +
                   "The interface will prompt you if further assistance is required."

    $(divRef).append('<div id="'+tabDivId+'_infoDialog">' + INFOTEXT + '</div>');
    $(divRef+'_infoDialog').dialog({autoOpen:false,
                              buttons: [{text:"Ok", click:function(){$(this).dialog("close");}}],
                              modal:true,
                              title:"Task Command Info",
                              width:"70%"
                              });

    //Info button - brings up info dialog
    $(divRef).append('<button id="'+tabDivId+'_info"> Help </button>');
    $(divRef+'_info').button();
    $(divRef+'_info').click(function () { $(divRef+'_infoDialog').dialog("open"); } );
    $(divRef+'_info').css({"position":"absolute",
                            "top":"10px",
                            "right":"10px"});
}

