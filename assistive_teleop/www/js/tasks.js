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
      var act = $("#task_action_select option:selected").text();
      var side = $("#task_side_select option:selected").text();
      var area = $("#task_area_select option:selected").text();
      var msg = new taskUI.ros.Message({
        data: act + ' ' + side + ' ' + area
      });
      taskUI.startTaskPub.publish(msg);
      var txt = "Starting Task, including moving base to perform the task: " + act + " the " + side + " " + area;
      assistive_teleop.log(txt);
    };

    taskUI.moveBasePub = new taskUI.ros.Topic({
      name: 'move_base_to_goal',
      messageType: 'std_msgs/String'
    });
    taskUI.moveBasePub.advertise();

    taskUI.moveBase = function () {
      var msg = new taskUI.ros.Message({
        data: 'move_base'
      });
      taskUI.moveBasePub.publish(msg);
      var txt = "Moving Base to good location to perform the task";
      assistive_teleop.log(txt);
    };

    taskUI.moveArmPub = new taskUI.ros.Topic({
      name: 'move_arm_to_goal',
      messageType: 'std_msgs/String'
    });
    taskUI.moveArmPub.advertise();

    taskUI.moveArm = function () {
      var act = $("#task_action_select option:selected").text();
      var side = $("#task_side_select option:selected").text();
      var area = $("#task_area_select option:selected").text();
      var msg = new taskUI.ros.Message({
        data: act + ' ' + side + ' ' + area
      });
      taskUI.moveArmPub.publish(msg);
      var txt = "Moving Arm to perform the task: " + act + " the " + side + " " + area;
      assistive_teleop.log(txt);
    };

    taskUI.resetArmPub = new taskUI.ros.Topic({
      name: 'reset_arm_ui',
      messageType: 'std_msgs/String'
    });
    taskUI.resetArmPub.advertise();

    taskUI.resetArm = function () {
      var msg = new taskUI.ros.Message({
        data: 'reach_initialization'
      });
      taskUI.resetArmPub.publish(msg);
      var txt = "Resetting arm configurations";
      assistive_teleop.log(txt);
    };

    taskUI.startTrackARPub = new taskUI.ros.Topic({
      name: 'track_ar_ui',
      messageType: 'std_msgs/Bool'
    });
    taskUI.startTrackARPub.advertise();

    taskUI.startTrackAR = function () {
      var msg = new taskUI.ros.Message({
        data: true
      });
      taskUI.startTrackARPub.publish(msg);
      var txt = "Start tracking the Bed AR tag";
      assistive_teleop.log(txt);
    };

    taskUI.stopTrackARPub = new taskUI.ros.Topic({
      name: 'track_ar_ui',
      messageType: 'std_msgs/Bool'
    });
    taskUI.stopTrackARPub.advertise();

    taskUI.stopTrackAR = function () {
      var msg = new taskUI.ros.Message({
        data: false
      });
      taskUI.stopTrackARPub.publish(msg);
      var txt = "Stop tracking the Bed AR tag";
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
                         '<th id="' + tabDivId + '_R0C1">Side</th>' +
                         '<th id="' + tabDivId + '_R0C2">Area</th>' +
                       '</tr>' +
                       '<tr>' +
                         '<td id="' + tabDivId + '_R1C0"></td>' +
                         '<td id="' + tabDivId + '_R1C1"></td>' +
                         '<td id="' + tabDivId + '_R1C2" colspan="2"></td>' +
                         '<td id="' + tabDivId + '_R1C4" colspan="2"></td>' +
                         '<td id="' + tabDivId + '_R1C6" colspan="2"></td>' +
                       '</tr>' +
                       '<tr>' +
                         '<td id="' + tabDivId + '_R2C0" colspan="2"></td>' +
                         '<td id="' + tabDivId + '_R2C2" colspan="2"></td>' +
                         '<td id="' + tabDivId + '_R2C4" colspan="2"></td>' +
                         '<td id="' + tabDivId + '_R2C6" colspan="2"></td>' +
                       '</tr>' +
                     '</table>');
    $(divRef+'_R1C0').append('<select id="task_action_select">' +
                             '<option>scratching</option>' +
                             '<option>wiping</option>' +
                             '</select>');
    $(divRef+'_R1C1').append('<select id="task_side_select">' +
                             '<option>left</option>' +
                             '<option>right</option>' +
                             '</select>');
    $(divRef+'_R1C2').append('<select id="task_area_select">' +
                               '<option>knee</option>' +
                               '<option>upper_arm</option>' +
                               '<option>face</option>' +
                             '</select>');
    $(divRef+'_R2C0').append('<button id="start_task"> Start Task </button>');
    $("#start_task").button()
    $(divRef+'_R2C0').click(assistive_teleop.taskUI.startTask);
    $(divRef+'_R2C2').append('<button id="move_base"> Move Base </button>');
    $("#move_base").button()
    $(divRef+'_R2C2').click(assistive_teleop.taskUI.moveBase);
    $(divRef+'_R2C4').append('<button id="move_arm"> Move Arm </button>');
    $("#move_arm").button()
    $(divRef+'_R2C4').click(assistive_teleop.taskUI.moveArm);
    $(divRef+'_R1C4').append('<button id="reset_arm"> Reset Arm </button>');
    $("#reset_arm").button()
    $(divRef+'_R1C4').click(assistive_teleop.taskUI.resetArm);
    $(divRef+'_R1C6').append('<button id="start_track_ar"> Start Track AR </button>');
    $("#start_track_ar").button()
    $(divRef+'_R1C6').click(assistive_teleop.taskUI.startTrackAR);
    $(divRef+'_R2C6').append('<button id="stop_track_ar"> Stop Track AR </button>');
    $("#stop_track_ar").button()
    $(divRef+'_R2C6').click(assistive_teleop.taskUI.stopTrackAR);

    // Info dialog box -- Pops up with instructions for using the body registration tab
    var INFOTEXT = "The Tasks Tab allows you to command the robot to do various tasks.</br>" +
                   "To send a task:</br></br>"+
                   "1. Select the action you would like the robot to perform.</br>"+
                   "2. Select the location where you would like the robot to perform this action.</br>" +
                   "3. Click 'Start Task'. The PR2 will move its base and the bed will adjust.</br>" +
                   "4. Click 'Move Arm'. The PR2 will move its arm to the task area.</br>" +
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

