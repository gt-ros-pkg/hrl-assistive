/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Robert Bosch LLC.
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
 *   * Neither the name of the Robert Bosch nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
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
 * Authors: Benjamin Pitzer, Robert Bosch LLC
 * 
 *********************************************************************/

/**
 * Full interface to an ActionServer
 *
 * ActionClient provides a complete client side implementation of the ActionInterface protocol.
 * It provides callbacks for every client side transition, giving the user full observation into
 * the client side state machine.
 * 
 * @class
 * @augments Class
 */
ros.actionlib.ActionClient  = Class.extend(
  /** @lends ros.actionlib.ActionClient# */
  {
  /** 
   *  Constructs an ActionClient and opens connections to an ActionServer.
   *  
   *  @param node The node handle that is used for communication with ROS
   *  @param ns The namespace in which to access the action.  For
   *  example, the "goal" topic should occur under ns/goal
   *  @param action_spec The *Action message type.  The ActionClient
   *  will grab the other message types from this type. A method of the  subclass. 
   */  
  init: function(node, ns, action_spec) {
    this.node = node;
    this.ns = ns;
    this.action_spec = action_spec;
    this.last_status_msg = null;
    
    this.pub_goal = node.advertise(ns + '/goal', action_spec.action_goal_type);
    this.pub_cancel = node.advertise(ns + '/cancel', 'GoalID');

    this.manager = new ros.actionlib.GoalManager(node, action_spec);
    
    var that = this;
    this.manager.register_send_goal_fn(function(msg){that.pub_goal.publish(msg);});
    this.manager.register_cancel_fn(function(msg){that.pub_cancel.publish(msg);});

    node.subscribe(ns + '/status', function(e){that.status_cb(e);});
    node.subscribe(ns + '/result', function(e){that.result_cb(e);});
    node.subscribe(ns + '/feedback', function(e){that.feedback_cb(e);});
  },
  
  status_cb: function (msg) {
    this.last_status_msg = msg
    this.manager.update_statuses(msg)
  },
  
  result_cb: function (msg) {
    this.manager.update_results(msg);
  },
  
  feedback_cb: function (msg) {
    this.manager.update_feedbacks(msg);
  },
  
  /**
   * Waits for the ActionServer to connect to this client.
   *
   * Often, it can take a second for the action server & client to negotiate
   * a connection, thus, risking the first few goals to be dropped. This call lets
   * the user wait until the network connection to the server is negotiated
   */ 
  wait_for_server: function (timeout, callback) {
    var timeout_time = (new Date()).getTime()+ timeout*1000;
    // check every 100 ms if the status messages has arrived
    var that = this;
    var interval = setInterval( function () {
      if(!that.node.ok()) {
        clearInterval(interval);
        callback(false);
      }
      if(that.last_status_msg) {
        clearInterval(interval);
        callback(true);
      }
      if((new Date()).getTime() >= timeout_time) {
        clearInterval(interval);
        callback(false);
      }
    }, 100);
  },
  
  /**
   * Sends a goal to the action server.
   *
   * @param goal An instance of the *Goal message.
   *
   * @param transition_cb Callback that gets called on every client
   * state transition for the sent goal.  It should take in a
   * ClientGoalHandle as an argument.
   *
   * @param feedback_cb Callback that gets called every time
   * feedback is received for the sent goal.  It takes two
   * parameters: a ClientGoalHandle and an instance of the *Feedback
   * message.
   *
   * @returns {ros.actionlib.ClientGoalHandle} ClientGoalHandle for the sent goal.
   */
  send_goal: function (goal, transition_cb, feedback_cb) {
    return this.manager.init_goal(goal, transition_cb, feedback_cb)
  },
      
  /**    
   * Cancels all goals currently running on the action server.
   *
   * Preempts all goals running on the action server at the point
   * that the cancel message is serviced by the action server.
   */
  cancel_all_goals: function() {
      var cancel_msg = new ros.actionlib.GoalID();
      this.pub_cancel.publish(cancel_msg);
  },
      
});