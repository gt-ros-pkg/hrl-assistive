/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Worcester Polytechnic Institute, Willow Garage
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Worcester Polytechnic Institute, Willow
 *     Garage, nor the names of its contributors may be used to 
 *     endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Russell Toris, Brandon Alexander
 *  Version: September 17, 2012
 *
 *  AMDfied by Jihoon
 *  Version : September 27, 2012
 *
 *********************************************************************/

(function (root, factory) {
    if(typeof define === 'function' && define.amd) {
        define(['eventemitter2'],factory);
    }
    else {
        root.ActionClient = factory(root.EventEmitter2);
    }
}(this, function(EventEmitter2) {
var ActionClient = function(options) {
  var actionClient = this;
  options = options || {};
  actionClient.ros         = options.ros;
  actionClient.serverName  = options.serverName;
  actionClient.actionName  = options.actionName;
  actionClient.timeout     = options.timeout;
  actionClient.goals       = {};

  actionClient.goalTopic = new actionClient.ros.Topic({
    name        : actionClient.serverName + '/goal'
  , messageType : actionClient.actionName + 'Goal'
  });
  actionClient.goalTopic.advertise();

  actionClient.cancelTopic = new actionClient.ros.Topic({
    name        : actionClient.serverName + '/cancel'
  , messageType : 'actionlib_msgs/GoalID'
  });
  actionClient.cancelTopic.advertise();

  var receivedStatus = false;
  var statusListener = new actionClient.ros.Topic({
    name        : actionClient.serverName + '/status'
  , messageType : 'actionlib_msgs/GoalStatusArray'
  });
  statusListener.subscribe(function (statusMessage) {
    receivedStatus = true;

    statusMessage.status_list.forEach(function(status) {
      var goal = actionClient.goals[status.goal_id.id];
      if (goal) {
        goal.emit('status', status);
      }
    });
  });

  // If timeout specified, emit a 'timeout' event if the ActionServer does not
  // respond before the timeout.
  if (actionClient.timeout) {
    setTimeout(function() {
      if (!receivedStatus) {
        actionClient.emit('timeout');
      }
    }, actionClient.timeout);
  }

  // Subscribe to the feedback, and result topics
  var feedbackListener = new actionClient.ros.Topic({
    name        : actionClient.serverName + '/feedback'
  , messageType : actionClient.actionName + 'Feedback'
  });
  feedbackListener.subscribe(function (feedbackMessage) {
    var goal = actionClient.goals[feedbackMessage.status.goal_id.id];

    if (goal) {
      goal.emit('status', feedbackMessage.status);
      goal.emit('feedback', feedbackMessage.feedback);
    }
  });

  var resultListener = new actionClient.ros.Topic({
    name        : actionClient.serverName + '/result'
  , messageType : actionClient.actionName + 'Result'
  });
  resultListener.subscribe(function (resultMessage) {
    var goal = actionClient.goals[resultMessage.status.goal_id.id];

    if (goal) {
      goal.emit('status', resultMessage.status);
      goal.emit('result', resultMessage.result);
    }
  });

  actionClient.cancel = function() {
    var cancelMessage = new ros.Message({});
    actionClient.cancelTopic.publish(cancelMessage);
  };

  actionClient.Goal = function(goalMsg) {
    var goal = this;

    goal.isFinished = false;
    goal.status;
    goal.result;
    goal.feedback;

    var date = new Date();
    goal.goalId = 'goal_' + Math.random() + "_" + date.getTime();
    goal.goalMessage = new actionClient.ros.Message({
      goal_id : {
        stamp: {
          secs  : 0
        , nsecs : 0
        }
      , id: goal.goalId
      }
    , goal: goalMsg
    });

    goal.on('status', function(status) {
      goal.status = status;
    });

    goal.on('result', function(result) {
      goal.isFinished = true;
      goal.result = result;
    });

    goal.on('feedback', function(feedback) {
      goal.feedback = feedback;
    });

    actionClient.goals[goal.goalId] = this;

    goal.send = function(timeout) {
      actionClient.goalTopic.publish(goal.goalMessage);
      if (timeout) {
         setTimeout(function() {
           if (!goal.isFinished) {
             goal.emit('timeout');
           }
         }, timeout);
      }
    };

    goal.cancel = function() {
      var cancelMessage = new ros.Message({
        id: goal.goalId
      });
      actionClient.cancelTopic.publish(cancelMessage);
    };
  };
  actionClient.Goal.prototype.__proto__ = EventEmitter2.prototype;

};
ActionClient.prototype.__proto__ = EventEmitter2.prototype;
return ActionClient;
}
));
