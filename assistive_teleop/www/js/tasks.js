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

}

