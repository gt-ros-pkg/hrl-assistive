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
 * Global counter for goal ids
 */
ros.actionlib.g_goal_id = 0;

/**
 * Helper class to manage action goals
 * 
 * @class
 * @augments Class
 */
ros.actionlib.GoalManager  = Class.extend(
  /** @lends ros.actionlib.GoalManager# */
  {
  /** 
   *  Constructs a GoalManager based on the action's ActionSpec.
   *  
   *  @param node The node handle that is used for communication with ROS
   *  @param {ros.actionlib.ActionSpec} action_spec the action's message type.
   */  
  init: function(node, action_spec) {
    this.node = node;
    this.statuses = [];
    this.action_spec = action_spec;
    
    this.send_goal_fn = null;
    this.cancel_fn = null;
  },
 
  /** 
   *  Internal function to generate a new goal ID by using the current time and the global goal id counter
   *  
   *  @returns {ros.actionlib.GoalID} a generated goal id
   */  
  _generate_id: function() {
    var id = ros.actionlib.g_goal_id;
    ros.actionlib.g_goal_id++;
    var now = (new ros.Time()).now();
    return new ros.actionlib.GoalID(now,id+"-"+now.toSec());
  },

  /** 
   *  Register a function to send goals to the action server
   *  
   *  @param fn function will send the goal message to the action server
   */  
  register_send_goal_fn: function(fn) {
    this.send_goal_fn = fn;
  },

  /** 
   *  Register a function to send cancel messages to the action server
   *  
   * @param fn function will send the cancel message to the action server
   */    
  register_cancel_fn: function(fn) {
    this.cancel_fn = fn;
  },
  
  /**
   * Sends off a goal and starts tracking its status.
   *
   * @returns {ros.actionlib.ClientGoalHandle} for the sent goal.
   */
  init_goal: function(goal, transition_cb, feedback_cb) {
    var action_goal = new ros.actionlib.ActionGoal(new ros.roslib.Header(), this._generate_id(), goal);
    action_goal.header.stamp = (new ros.Time()).now();
    var csm = new ros.actionlib.CommStateMachine(action_goal, transition_cb, feedback_cb, this.send_goal_fn, this.cancel_fn);
    this.statuses.push(csm);
    this.send_goal_fn(action_goal);
    return new ros.actionlib.ClientGoalHandle(csm);
  },
    
  /**
   * Updates the statuses of all goals from the information in status_array.
   *
   * @param {ros.actionlib.GoalStatusArray} status_array status array
   */
  update_statuses: function(status_array) {
    for(i in this.statuses) {
      var status = this.statuses[i];
      status.update_status(status_array);
    }
  },

  /**
   *  Main function to update the results by processing the action result
   * 
   *  @param action_result the action result
   */
  update_results: function(action_result) {
    for(i in this.statuses) {
      var status = this.statuses[i];
      status.update_result(action_result);
    }
  },

  /**
   *  Function to process the action feedback message
   * 
   *  @param action_feedback an action feedback message
   */
  update_feedbacks: function(action_feedback) {
    for(i in this.statuses) {
      var status = this.statuses[i];
      status.update_feedback(action_feedback);
    }
  },

});