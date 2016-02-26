var TaskInterface = function (ros) {
    'use strict';
    var taskUI = this;
    taskUI.ros = ros;

    taskUI.startTaskPub = new taskUI.ros.Topic({
      name: 'action_location_goal',
      messageType: 'std_msgs/String'
    });
    taskUI.startTaskPub.advertise();

    taskUI.startTask = function () {
      var loc = $("#task_location_select option:selected").text();
      var act = $("#task_action_select option:selected").text();
      var msg = new taskUI.ros.Message({
        data: loc
      });
      taskUI.startTaskPub.publish(msg);
      var txt = "Starting Task, including moving base to perform the task: " + act + " the " + loc;
      assistive_teleop.log(txt);
    };

    taskUI.moveArmPub = new taskUI.ros.Topic({
      name: 'move_arm_to_goal',
      messageType: 'std_msgs/String'
    });
    taskUI.moveArmPub.advertise();

    taskUI.moveArm = function () {
      var loc = $("#task_location_select option:selected").text();
      var act = $("#task_action_select option:selected").text();
      var msg = new taskUI.ros.Message({
        data: loc
      });
      taskUI.moveArmPub.publish(msg);
      var txt = "Moving Arm to perform the task: " + act + " the " + loc;
      assistive_teleop.log(txt);
    };
}

var initTaskInterface = function (tabDivId) {
    assistive_teleop.taskUI = new TaskInterface(assistive_teleop.ros);
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
                         '<td id="' + tabDivId + '_R2C2" colspan="2"></td>' +
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
    $(divRef+'_R2C0').append('<button id="start_task"> Start Task </button>');
    $("#start_task").button()
    $(divRef+'_R2C0').click(assistive_teleop.taskUI.startTask);
    $(divRef+'_R2C2').append('<button id="move_arm"> Move Arm </button>');
    $("#move_arm").button()
    $(divRef+'_R2C2').click(assistive_teleop.taskUI.moveArm);

    // Info dialog box -- Pops up with instructions for using the body registration tab
    var INFOTEXT = "The Tasks Tab allows you to command the robot to do various tasks.</br>" +
                   "To send a task:</br></br>"+
                   "1. Select the action you would like the robot to perform.</br>"+
                   "2. Select the location where you would like the robot to perform this action.</br>" +
                   "3. Click 'Start Task. The PR2 will move its base and the bed will adjust.'</br></br>" +
                   "4. Click 'Move Arm. The PR2 will move its arm to the task area.'</br></br>" +
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

