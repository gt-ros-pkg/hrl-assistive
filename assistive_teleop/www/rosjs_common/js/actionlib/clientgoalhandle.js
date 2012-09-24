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
 * Client side handle to monitor goal progress.
 *
 * A ClientGoalHandle is a reference counted object that is used to manipulate and monitor the progress
 * of an already dispatched goal. Once all the goal handles go out of scope (or are reset), an
 * ActionClient stops maintaining state for that goal.
 * 
 * @class
 * @augments Class
 */
ros.actionlib.ClientGoalHandle = Class.extend(
  /** @lends ros.actionlib.ClientGoalHandle# */  
  {
  /**
   * Create an empty goal handle.
   *
   * Constructs a goal handle that doesn't track any goal. Calling any method on an empty goal
   * handle other than operator= will trigger an assertion.
   * 
   * @param {ros.actionlib.CommStateMachine} comm_state_machine Communication state machine
   */  
  init: function(comm_state_machine) {
    this.comm_state_machine = comm_state_machine;
  },
  
  /**
   * True if the two ClientGoalHandle's are tracking the same goal
   * 
   * @param {ros.actionlib.ClientGoalHandle} o Other goal handle
   */
  isEqualTo: function(o) {
    if(!o) {
      return false;
    }
    return this.comm_state_machine == o.comm_state_machine;
  },

  /**
   * True if the two ClientGoalHandle's are tracking different goals
   * 
   * @param {ros.actionlib.ClientGoalHandle} o Other goal handle
   */
  isNotEqualTo: function(o) {
    if(!o) {
      return true;
    }
    return !(this.comm_state_machine == o.comm_state_machine);
  },

   /**
    * Sends a cancel message for this specific goal to the ActionServer.
    *
    * Also transitions the client state to WAITING_FOR_CANCEL_ACK
    */
   cancel: function() {
     var cancel_msg = ros.actionlib.GoalID(new ros.Time(), this.comm_state_machine.action_goal.goal_id.id);
     this.comm_state_machine.send_cancel_fn(cancel_msg.toMessage());
     this.comm_state_machine.transition_to(ros.actionlib.CommState.WAITING_FOR_CANCEL_ACK);
   },

  /**
   * Get the state of this goal's communication state machine from interaction with the server
   *
   * Possible States are: WAITING_FOR_GOAL_ACK, PENDING, ACTIVE, WAITING_FOR_RESULT,
   *                      WAITING_FOR_CANCEL_ACK, RECALLING, PREEMPTING, DONE
   *
   * @returns The current goal's communication state with the server
   */
   get_comm_state: function () {
     if(!this.comm_state_machine) {
       ros_error("Trying to get_comm_state on an inactive ClientGoalHandle.");
       return ros.actionlib.CommState.LOST;
     }
     return this.comm_state_machine.state;
   },

  /**
   * Returns the current status of the goal.
   *
   * Possible states are listed in the enumeration in the
   * actionlib_msgs/GoalStatus message.
   *
   * @returns The current status of the goal.
   */
   get_goal_status: function() {
     if (!this.comm_state_machine) {
       ros_error("Trying to get_goal_status on an inactive ClientGoalHandle.");
       return ros.actionlib.GoalStatus.PENDING;
     }
     return this.comm_state_machine.latest_goal_status.status;
   },
   
  /** 
   * Returns the current status text of the goal.
   *
   * The text is sent by the action server.
   *
   * @returns The current status text of the goal.
   */
   get_goal_status_text: function() {
     if(!this.comm_state_machine) {
       ros_error("Trying to get_goal_status_text on an inactive ClientGoalHandle.");
       return "ERROR: Trying to get_goal_status_text on an inactive ClientGoalHandle.";
     }
     return this.comm_state_machine.latest_goal_status.text;
   },
   
   /** 
    * Gets the result produced by the action server for this goal.
    *
    * @returns None if no result was receieved.  Otherwise the goal's result as a *Result message.
    */
   get_result: function() {
     if (!this.comm_state_machine.latest_result) {
       ros_error("Trying to get_result on an inactive ClientGoalHandle.");
       return null;
     }
     return this.comm_state_machine.latest_result.result;
   },
   
   /**
    * Gets the terminal state information for this goal.
    *
    * Possible States Are: RECALLED, REJECTED, PREEMPTED, ABORTED, SUCCEEDED, LOST
    * This call only makes sense if CommState==DONE. This will send ROS_WARNs if we're not in DONE
    *
    * @returns The terminal state as an integer from the GoalStatus message.
    */
    get_terminal_state: function() {
       if(!this.comm_state_machine) {
         ros_error("Trying to get_terminal_state on an inactive ClientGoalHandle.");
         return ros.actionlib.GoalStatus.LOST
       }
  
       if(this.comm_state_machine.state != ros.actionlib.CommState.DONE) {
         ros_warning("Asking for the terminal state when we're in ["+this.comm_state_machine.state+"]");
       }
       
       var goal_status = this.comm_state_machine.latest_goal_status.status;
       if(goal_status in [ros.actionlib.GoalStatus.PREEMPTED, ros.actionlib.GoalStatus.SUCCEEDED,
                          ros.actionlib.GoalStatus.ABORTED, ros.actionlib.GoalStatus.REJECTED,
                          ros.actionlib.GoalStatus.RECALLED, ros.actionlib.GoalStatus.LOST]) {
         return goal_status;
       }
  
       ros_error("Asking for a terminal state, but the goal status is "+goal_status);
       return ros.actionlib.GoalStatus.GoalStatus.LOST;
  },
      
});