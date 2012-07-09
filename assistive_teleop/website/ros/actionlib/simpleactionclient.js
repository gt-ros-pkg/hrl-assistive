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
 * A Simple client implementation of the ActionInterface which supports only one goal at a time. 
 *
 * The SimpleActionClient wraps the existing ActionClient, and exposes a limited set of easy-to-use hooks
 * for the user. Note that the concept of GoalHandles has been completely hidden from the user, and that
 * they must query the SimplyActionClient directly in order to monitor a goal.
 * 
 * @class
 * @augments Class
 */
ros.actionlib.SimpleActionClient  = Class.extend(
  /** @lends ros.actionlib.SimpleActionClient# */
  {
  /**
   * Constructs a SingleGoalActionClient and sets up the necessary ros topics for the ActionInterface
   * 
   *  @param node The node handle that is used for communication with ROS
   *  @param ns The namespace in which to access the action.  For
   *  example, the "goal" topic should occur under ns/goal
   *  @param action_spec The *Action message type.  The ActionClient
   *  will grab the other message types from this type. A method of the  subclass. 
   */
  init: function(node, ns, action_spec) {
    this.node = node;
    this.action_client = new ros.actionlib.ActionClient(node, ns, action_spec)
    this.simple_state = ros.actionlib.SimpleGoalState.DONE;
    this.gh = null;
  },
  
  /**
   * Waits for the ActionServer to connect to this client.
   *
   * Often, it can take a second for the action server & client to negotiate
   * a connection, thus, risking the first few goals to be dropped. This call lets
   * the user wait until the network connection to the server is negotiated
   */ 
  wait_for_server: function (timeout, callback) {
    this.action_client.wait_for_server(timeout, callback);
  },
  
  /**
  * Sends a goal to the ActionServer, and also registers callbacks. 
  *
  * If a previous goal is already active when this is called. We simply forget
  * about that goal and start tracking the new goal. No cancel requests are made.
  *
  * @param goal the action goal
  * 
  * @param done_cb Callback that gets called on transitions to
  * Done.  The callback should take two parameters: the terminal
  * state (as an integer from actionlib_msgs/GoalStatus) and the
  * result.
  *
  * @param active_cb No-parameter callback that gets called on transitions to Active.
  *
  * @param feedback_cb Callback that gets called whenever feedback
  * for this goal is received.  Takes one parameter: the feedback.
  */ 
  send_goal: function(goal, done_cb, active_cb, feedback_cb) {
    
    // destroys the old goal handle
    this.stop_tracking_goal();

    this.done_cb = done_cb;
    this.active_cb = active_cb;
    this.feedback_cb = feedback_cb;
    this.simple_state = ros.actionlib.SimpleGoalState.PENDING;
    var that = this;
    this.gh = this.action_client.send_goal(goal, 
        function(gh){that._handle_transition(gh);}, 
        function(gh,feedback){that._handle_feedback(gh,feedback);});
  },
    
  /**
   * Sends a goal to the ActionServer, waits for the goal to complete, and preempts goal is necessary.
   *
   * If a previous goal is already active when this is called. We simply forget
   * about that goal and start tracking the new goal. No cancel requests are made.
   *
   * If the goal does not complete within the execute_timeout, the goal gets preempted
   *
   * If preemption of the goal does not complete withing the preempt_timeout, this
   * method simply returns
   *
   * @param execute_timeout The time to wait for the goal to complete
   *
   * @param preempt_timeout The time to wait for preemption to complete
   *
   * @returns The goal's state.
   */
//  send_goal_and_wait: function(goal, execute_timeout, preempt_timeout, callback) {
//    this.send_goal(goal);
//    
//    var that = this;
//    //Waits for the server to finish performing the action.
//    this.wait_for_result(execute_timeout, function(result) {
//      if(!result) {
//        ros_error("Didn't receive results from action server.");
//        this.cancel_goal();
//        this.wait_for_result(execute_timeout, function(e) {
//          
//        }
//        callback(this.get_state());
//        return;
//      }
//      // Prints out the result of executing the action
//      var result = client.get_result(); // A FibonacciResult
//      ros_debug("Result:"+", "+result.sequence);
//    });
//  
//    if not self.wait_for_result(execute_timeout):
//        // preempt action
//        rospy.logdebug("Canceling goal")
//        self.cancel_goal()
//        if self.wait_for_result(preempt_timeout):
//          ros_debug("Preempt finished within specified preempt_timeout [%.2f]", preempt_timeout.to_sec());
//        else:
//            rospy.logdebug("Preempt didn't finish specified preempt_timeout [%.2f]", preempt_timeout.to_sec());
//    return self.get_state()
//  },
  
  /**
   * Waits until this goal transitions to done and calls a callback function
   * 
   * @param timeout Max time to block before returning. A zero timeout is interpreted as an infinite timeout.
   * @param callback a callback function that is called when the result is available
   * @returns True if the goal finished. False if the goal didn't finish within the allocated timeout
   */
  wait_for_result: function(timeout, callback) {
    if(!this.gh) {
      ros_error("Called wait_for_goal_to_finish when no goal exists");
      return false;
    }

    var timeout_time = (new Date()).getTime()+ timeout*1000;
  
    // check every 100 ms if the status messages has arrived
    var that = this;
    var interval = setInterval( function () {
      if(!that.node.ok()) {
        clearInterval(interval);
        callback(false);
      }
      
      var time_left = timeout_time - (new Date()).getTime();
      if(timeout > 0 && time_left <= 0) {
        clearInterval(interval);
        callback(that.simple_state == ros.actionlib.SimpleGoalState.DONE);
      }
       
      if(that.simple_state == ros.actionlib.SimpleGoalState.DONE) {
        clearInterval(interval);
        callback(true);
      }
    }, 100);
    
    return true;
  },
        
  /**
   * Gets the Result of the current goal a callback function that is called when the result is available
   */
  get_result: function() {
    if(!this.gh) {
      ros_error("Called get_result when no goal is running");
      return null;
    }
    return this.gh.get_result();
  },

  /** 
   * Get the state information for this goal.
   *
   * Possible States Are: PENDING, ACTIVE, RECALLED, REJECTED,
   * PREEMPTED, ABORTED, SUCCEEDED, LOST.
   *
   * @returns The goal's state. Returns LOST if this
   * SimpleActionClient isn't tracking a goal.
   */
  get_state: function() {
    if(!this.gh) {
      ros_error("Called get_state when no goal is running");
      return ros.actionlib.GoalStatus.LOST;
    }
    var status = this.gh.get_goal_status();
  
    if(status == ros.actionlib.GoalStatus.RECALLING) {
      status = ros.actionlib.GoalStatus.PENDING;
    }
    else if(status == ros.actionlib.GoalStatus.PREEMPTING) {
      status = ros.actionlib.GoalStatus.ACTIVE;
    }
  
    return status;
  },

   
  /**
   * Returns the current status text of the goal.
   *
   * The text is sent by the action server. It is designed to
   * help debugging issues on the server side.
   *
   * @returns The current status text of the goal.
   */
   get_goal_status_text: function() {
     if(!this.gh) {
       ros_error("Called get_goal_status_text when no goal is running");
       return "ERROR: Called get_goal_status_text when no goal is running";
     }
     return this.gh.get_goal_status_text()
   },

   /**
    * Cancels all goals currently running on the action server.
    *
    * This preempts all goals running on the action server at the point that
    * this message is serviced by the ActionServer.
    */
   cancel_all_goals: function () {
     this.action_client.cancel_all_goals();
   },
   
   /**
    * Cancels the goal that we are currently pursuing.
    */
   cancel_goal: function() {
     if(!this.gh) {
       this.gh.cancel();
     }
   },

   /**
    * Stops tracking the state of the current goal. Unregisters this goal's callbacks.
    *
    * This is useful if we want to make sure we stop calling our callbacks before sending a new goal.
    * Note that this does not cancel the goal, it simply stops looking for status info about this goal.
    */
   stop_tracking_goal: function() {
     this.gh = null;
   },

   _handle_transition: function(gh) {
     var comm_state = gh.get_comm_state();
     var error_msg = "Received comm state " + comm_state + " when in simple state " + this.simple_state;
    
     if(comm_state == ros.actionlib.CommState.ACTIVE) {
       if(this.simple_state == ros.actionlib.SimpleGoalState.PENDING) {
         this._set_simple_state(ros.actionlib.SimpleGoalState.ACTIVE);
         if(this.active_cb) {
           this.active_cb();
         }
       }
       else if(this.simple_state == ros.actionlib.SimpleGoalState.DONE) {
         rospy.logerr(error_msg);
       } 
     }
     else if(comm_state == ros.actionlib.CommState.RECALLING) {
       if(this.simple_state != ros.actionlib.SimpleGoalState.PENDING) {
         rospy.logerr(error_msg);
       }
     }
     else if(comm_state == ros.actionlib.CommState.PREEMPTING) {
       if(this.simple_state == ros.actionlib.SimpleGoalState.PENDING) {
         this._set_simple_state(ros.actionlib.SimpleGoalState.ACTIVE);
         if(this.active_cb) {
           this.active_cb();
         }
       }
       else if(this.simple_state == ros.actionlib.SimpleGoalState.DONE) {
         ros_error(error_msg)
       }
     }
     else if(comm_state == ros.actionlib.CommState.DONE) {
       if(this.simple_state in [ros.actionlib.SimpleGoalState.PENDING, ros.actionlib.SimpleGoalState.ACTIVE]) {
         this._set_simple_state(ros.actionlib.SimpleGoalState.DONE);
         if(this.done_cb) {
           this.done_cb(gh.get_goal_status(), gh.get_result());
         }
       }
       else if(this.simple_state == ros.actionlib.SimpleGoalState.DONE) {
         ros_error("SimpleActionClient received DONE twice");
       }
     }
   },
   
   
   _handle_feedback: function(gh, feedback) {
     if(!this.gh) {
       ros_error("Got a feedback callback when we're not tracking a goal. (id: " + gh.comm_state_machine.action_goal.goal_id.id + ")");
     }
     if(!gh.isEqualTo(this.gh)) {
       ros_error("Got a feedback callback on a goal handle that we're not tracking. "+this.gh.comm_state_machine.action_goal.goal_id.id+" vs "+gh.comm_state_machine.action_goal.goal_id.id);
       return;
     }
     if(this.feedback_cb) {
       this.feedback_cb(feedback);
     }
   },
     
   _set_simple_state: function (state) {
     this.simple_state = state;
   },

   
});



