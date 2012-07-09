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
 * Flag for CommStateMachine transitions
 * @constant 
 */
ros.actionlib.NO_TRANSITION = -1;

/**
 * Flag for CommStateMachine transitions
 * @constant 
 */
ros.actionlib.INVALID_TRANSITION = -2;

/**
 * Client side state machine to track the action client's state
 * 
 * @class
 * @augments Class
 */
ros.actionlib.CommStateMachine  = Class.extend(
  /** @lends ros.actionlib.CommStateMachine# */
  {
  /** 
   *  Constructs a CommStateMachine and initializes its state with WAITING_FOR_GOAL_ACK.
   *  
   *  @param {ros.actionlib.ActionGoal} action_goal the action goal
   *  @param transition_cb callback function that is being called at state transitions
   *  @param feedback_cb callback function that is being called at the arrival of a feedback message
   *  @param send_goal_fn function will send the goal message to the action server
   *  @param send_cancel_fn function will send the cancel message to the action server
   */    
  init: function(action_goal, transition_cb, feedback_cb, send_goal_fn, send_cancel_fn) {
    this.action_goal = action_goal;
    this.transition_cb = transition_cb;
    this.feedback_cb = feedback_cb;
    this.send_goal_fn = send_goal_fn;
    this.send_cancel_fn = send_cancel_fn;

    this.state = ros.actionlib.CommState.WAITING_FOR_GOAL_ACK;
    this.latest_goal_status = ros.actionlib.GoalStatus.PENDING;
    this.latest_result = null;
    this.set_transitions();
   },
   
   /** 
    *  Helper method to construct the transition matrix.
    */  
   set_transitions: function() {
     this.transitions = new Array();
     for(var i=0;i<8;i++) {
       this.transitions.push(new Array());
       for(var j=0;j<9;j++) {
         this.transitions[i].push(new Array());
       }
       
       switch(i) {
       case ros.actionlib.CommState.WAITING_FOR_GOAL_ACK:
         this.transitions[i][ros.actionlib.GoalStatus.PENDING] = [ros.actionlib.CommState.PENDING];
         this.transitions[i][ros.actionlib.GoalStatus.ACTIVE] = [ros.actionlib.CommState.ACTIVE];
         this.transitions[i][ros.actionlib.GoalStatus.REJECTED] = [ros.actionlib.CommState.PENDING, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLING] = [ros.actionlib.CommState.PENDING, ros.actionlib.CommState.RECALLING];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLED] = [ros.actionlib.CommState.PENDING, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTED] = [ros.actionlib.CommState.ACTIVE, ros.actionlib.CommState.PREEMPTING, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.SUCCEEDED] = [ros.actionlib.CommState.ACTIVE, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.ABORTED] = [ros.actionlib.CommState.ACTIVE, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTING] = [ros.actionlib.CommState.ACTIVE, ros.actionlib.CommState.PREEMPTING];
         break;
       case ros.actionlib.CommState.PENDING:
         this.transitions[i][ros.actionlib.GoalStatus.PENDING] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.ACTIVE] = [ros.actionlib.CommState.ACTIVE];
         this.transitions[i][ros.actionlib.GoalStatus.REJECTED] = [ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLING] = [ros.actionlib.CommState.RECALLING];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLED] = [ros.actionlib.CommState.RECALLING, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTED] = [ros.actionlib.CommState.ACTIVE, ros.actionlib.CommState.PREEMPTING, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.SUCCEEDED] = [ros.actionlib.CommState.ACTIVE, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.ABORTED] = [ros.actionlib.CommState.ACTIVE, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTING] = [ros.actionlib.CommState.ACTIVE, ros.actionlib.CommState.PREEMPTING];
         break;
       case ros.actionlib.CommState.ACTIVE:
         this.transitions[i][ros.actionlib.GoalStatus.PENDING] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.ACTIVE] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.REJECTED] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLING] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLED] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTED] = [ros.actionlib.CommState.PREEMPTING, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.SUCCEEDED] = [ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.ABORTED] = [ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTING] = [ros.actionlib.CommState.PREEMPTING];
         break;
       case ros.actionlib.CommState.WAITING_FOR_RESULT:
         this.transitions[i][ros.actionlib.GoalStatus.PENDING] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.ACTIVE] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.REJECTED] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLING] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLED] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTED] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.SUCCEEDED] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.ABORTED] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTING] = [ros.actionlib.INVALID_TRANSITION];
         break;
       case ros.actionlib.CommState.WAITING_FOR_CANCEL_ACK:
         this.transitions[i][ros.actionlib.GoalStatus.PENDING] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.ACTIVE] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.REJECTED] = [ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLING] = [ros.actionlib.CommState.RECALLING];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLED] = [ros.actionlib.CommState.RECALLING, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTED] = [ros.actionlib.CommState.PREEMPTING, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.SUCCEEDED] = [ros.actionlib.CommState.PREEMPTING, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.ABORTED] = [ros.actionlib.CommState.PREEMPTING, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTING] = [ros.actionlib.CommState.PREEMPTING];
         break;
       case ros.actionlib.CommState.RECALLING:
         this.transitions[i][ros.actionlib.GoalStatus.PENDING] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.ACTIVE] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.REJECTED] = [ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLING] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLED] = [ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTED] = [ros.actionlib.CommState.PREEMPTING, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.SUCCEEDED] = [ros.actionlib.CommState.PREEMPTING, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.ABORTED] = [ros.actionlib.CommState.PREEMPTING, ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTING] = [ros.actionlib.CommState.PREEMPTING];
         break;
       case ros.actionlib.CommState.PREEMPTING:
         this.transitions[i][ros.actionlib.GoalStatus.PENDING] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.ACTIVE] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.REJECTED] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLING] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLED] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTED] = [ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.SUCCEEDED] = [ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.ABORTED] = [ros.actionlib.CommState.WAITING_FOR_RESULT];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTING] = [ros.actionlib.NO_TRANSITION];
         break;
       case ros.actionlib.CommState.DONE:
         this.transitions[i][ros.actionlib.GoalStatus.PENDING] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.ACTIVE] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.REJECTED] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLING] = [ros.actionlib.INVALID_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.RECALLED] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTED] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.SUCCEEDED] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.ABORTED] = [ros.actionlib.NO_TRANSITION];
         this.transitions[i][ros.actionlib.GoalStatus.PREEMPTING] = [ros.actionlib.INVALID_TRANSITION];
         break;
       default:
         ros_error("Unknown CommState "+i);
       }
     }
   },
   
   /** 
    *  returns true if the other CommStateMachine is identical to this one
    *  
    *  @param {ros.actionlib.CommStateMachine} other other CommStateMachine
    *  @returns {Boolean} true if the other CommStateMachine is identical to this one
    */  
   isEqualTo: function(other) {
     return this.action_goal.goal_id.id == other.action_goal.goal_id.id;
   },
   
   
   /** 
    *  Manually set the state
    *  
    *  @param {ros.actionlib.CommState} state the new state for the CommStateMachine
    */  
   set_state: function(state) {
     ros_debug("Transitioning CommState from "+this.state+" to "+state);
     this.state = state;
   },

   /** 
    *  Finds the status of a goal by its id
    *  
    *  @param {ros.actionlib.GoalStatusArray} status_array the status array
    *  @param {Integer} id the goal id
    */ 
   _find_status_by_goal_id: function(status_array, id) {
     for(s in status_array.status_list) {
       var status = status_array.status_list[s];
       if(status.goal_id.id == id) {
         return status;
       }
     }
     return null;
   },
     
   /**
    *  Main function to update the state machine by processing the goal status array
    * 
    *  @param {ros.actionlib.GoalStatusArray} status_array the status array
    */
   update_status: function(status_array) {
     if(this.state == ros.actionlib.CommState.DONE) {
       return;
     }

     var status = this._find_status_by_goal_id(status_array, this.action_goal.goal_id.id);

     // You mean you haven't heard of me?
     if(!status) {
       if(!(this.state in [ros.actionlib.CommState.WAITING_FOR_GOAL_ACK,
                           ros.actionlib.CommState.WAITING_FOR_RESULT,
                           ros.actionlib.CommState.DONE])) {
           this._mark_as_lost();
       }
       return;
     }

     this.latest_goal_status = status;

     // Determines the next state from the lookup table
     if(this.state >= this.transitions.length) {
       ros_error("CommStateMachine is in a funny state: " + this.state);
       return;
     }
     if(status.status >= this.transitions[this.state].length) {
       ros_error("Got an unknown status from the ActionServer: " + status.status);
       return;
     }
     next_states = this.transitions[this.state][status.status];

     // Knowing the next state, what should we do?
     if(next_states[0] == ros.actionlib.NO_TRANSITION) {
     }
     else if(next_states[0] == ros.actionlib.INVALID_TRANSITION) {
       ros_error("Invalid goal status transition from "+this.state+" to "+status.status);
     }
     else {
       for(s in next_states) {
         var state = next_states[s];
         this.transition_to(state);
       }
     }
   },
   
   /**
    *  Make state machine transition to a new state
    * 
    *  @param {ros.actionlib.CommState} state the new state for the CommStateMachine
    */   
   transition_to: function (state) {
     ros_debug("Transitioning to "+state+" (from "+this.state+", goal: "+this.action_goal.goal_id.id+")");
     this.state = state;
     if(this.transition_cb) {
       this.transition_cb(new ros.actionlib.ClientGoalHandle(this));
     }
   },

   /**
    *  Mark state machine as lost
    */
   _mark_as_lost: function() {
     this.latest_goal_status.status = ros.actionlib.GoalStatus.LOST;
     this.transition_to(ros.actionlib.CommState.DONE);
   },

   /**
    *  Main function to update the results by processing the action result
    * 
    *  @param action_result the action result
    */
   update_result: function(action_result) {
     // Might not be for us
     if(this.action_goal.goal_id.id != action_result.status.goal_id.id)
       return;

     this.latest_goal_status = action_result.status;
     this.latest_result = action_result;

     if(this.state in [ros.actionlib.CommState.WAITING_FOR_GOAL_ACK,
                       ros.actionlib.CommState.WAITING_FOR_CANCEL_ACK,
                       ros.actionlib.CommState.PENDING,
                       ros.actionlib.CommState.ACTIVE,
                       ros.actionlib.CommState.WAITING_FOR_RESULT,
                       ros.actionlib.CommState.RECALLING,
                       ros.actionlib.CommState.PREEMPTING]) {
         // Stuffs the goal status in the result into a GoalStatusArray
         var status_array = new ros.actionlib.GoalStatusArray();
         status_array.status_list.push(action_result.status);
         this.update_status(status_array);
         this.transition_to(ros.actionlib.CommState.DONE);
     }
     else if(this.state == ros.actionlib.CommState.DONE) {
       ros_error("Got a result when we were already in the DONE state");
     }
     else {
       ros_error("In a funny state: "+this.state);
     }
   },

  /**
   *  Function to process the action feedback message
   * 
   *  @param action_feedback an action feedback message
   */
  update_feedback: function(action_feedback) {
     // Might not be for us
     if(this.action_goal.goal_id.id != action_feedback.status.goal_id.id) {
       return;
     }

     if(this.feedback_cb) {
       this.feedback_cb(new ros.actionlib.ClientGoalHandle(this), action_feedback.feedback);
     }
  },
});