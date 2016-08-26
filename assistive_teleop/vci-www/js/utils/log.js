var UserLog = function (options) {
  'use strict'
  var userLog = this;
  var options = options || {};
  userLog.divId = options.divId;
  var timeout_secs = options.timeout || 10;
  userLog.timeout = timeout_secs * 1000; // Give timeout in milliseconds
  userLog.history = {};
  userLog.timers = [];
  userLog.newestTime = new Date().getTime();

  userLog.log = function (message) {
    var time = new Date().getTime();
    var msg = message.toString();
    console.log("Log to user: " + msg);
    userLog.history[time] = msg;
    userLog.timers.push(setTimeout(userLog.update, userLog.timeout + 100));
    userLog.newestTime = time;
    userLog.update();
  };

  userLog.update = function () {
    var cutoff_time = new Date().getTime() - userLog.timeout;
    var html = ""
    var sortedKeys = Object.keys(userLog.history).sort().reverse();
    var numKeys = sortedKeys.length;
    for (var i=0; i < numKeys; i++) {
      var timeIn = sortedKeys[i];
      if (timeIn >= cutoff_time) {
        if (timeIn == userLog.newestTime) {
          html = userLog.history[timeIn] + html;
        } else {
          html = userLog.history[timeIn]  + html;
        }
      }
    }
    $(userLog.divId).html(html);
  };
}

var initUserLog = function (divId) {
    RFH.userlog = new UserLog({'divId':divId, 'timeout':10} );

    //Alias for simpler calls
    RFH.log = RFH.userlog.log;

    //Add wrapper for compatibiltiy with existing code.
    window.log = function(message) { 
      RFH.userlog.log(message);
    }
}
