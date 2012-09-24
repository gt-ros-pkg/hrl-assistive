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
 * @namespace Holds classes and functions for the actionlib interface.
 */
ros.actionlib = ros.actionlib || {};

/**
 * @namespace Namespace for goal status constants
 */
ros.actionlib.GoalStatus = ros.actionlib.GoalStatus || {};

/**
 * The goal has yet to be processed by the action server
 * @constant 
 */
ros.actionlib.GoalStatus.PENDING         = 0; 

/**
 * The goal is currently being processed by the action server
 * @constant 
 */
ros.actionlib.GoalStatus.ACTIVE          = 1;

/**
 * The goal received a cancel request after it started executing
 * and has since completed its execution (Terminal State) 
 * @constant 
 */
ros.actionlib.GoalStatus.PREEMPTED       = 2; 

/**
 * The goal was achieved successfully by the action server (Terminal State)
 * @constant 
 */
ros.actionlib.GoalStatus.SUCCEEDED       = 3;

/**
 * The goal was aborted during execution by the action server due
 * to some failure (Terminal State)
 * @constant 
 */
ros.actionlib.GoalStatus.ABORTED         = 4;

/**
 * The goal was rejected by the action server without being processed,
 * because the goal was unattainable or invalid (Terminal State)
 * @constant 
 */
ros.actionlib.GoalStatus.REJECTED        = 5;

/**
 * The goal received a cancel request after it started executing
 * and has not yet completed execution
 * @constant 
 */
ros.actionlib.GoalStatus.PREEMPTING      = 6;

/**
 * The goal received a cancel request before it started executing,
 * but the action server has not yet confirmed that the goal is canceled
 * @constant 
 */
ros.actionlib.GoalStatus.RECALLING       = 7;

/**
 * The goal received a cancel request before it started executing
 * and was successfully cancelled (Terminal State)
 * @constant 
 */
ros.actionlib.GoalStatus.RECALLED        = 8;

/**
 * An action client can determine that a goal is LOST. This should not be
 * sent over the wire by an action server
 * @constant 
 */
ros.actionlib.GoalStatus.LOST            = 9;

/**
 * @namespace Namespace for the communication state
 */
ros.actionlib.CommState = ros.actionlib.CommState || {};
/** @constant */
ros.actionlib.CommState.WAITING_FOR_GOAL_ACK = 0;
/** @constant */
ros.actionlib.CommState.PENDING = 1;
/** @constant */
ros.actionlib.CommState.ACTIVE = 2;
/** @constant */
ros.actionlib.CommState.WAITING_FOR_RESULT = 3;
/** @constant */
ros.actionlib.CommState.WAITING_FOR_CANCEL_ACK = 4;
/** @constant */
ros.actionlib.CommState.RECALLING = 5;
/** @constant */
ros.actionlib.CommState.PREEMPTING = 6;
/** @constant */
ros.actionlib.CommState.DONE = 7;
/** @constant */
ros.actionlib.CommState.LOST = 8;

/**
 * @namespace Namespace for the goal state
 */
ros.actionlib.SimpleGoalState = ros.actionlib.SimpleGoalState || {};
/** @constant */
ros.actionlib.SimpleGoalState.PENDING = 0;
/** @constant */
ros.actionlib.SimpleGoalState.ACTIVE = 1;
/** @constant */
ros.actionlib.SimpleGoalState.DONE = 2;

// include all urdf files at once
ros.include('actionlib/goalid');
ros.include('actionlib/goalstatus');
ros.include('actionlib/goalstatusarray');
ros.include('actionlib/actiongoal');
ros.include('actionlib/clientgoalhandle');
ros.include('actionlib/actionspec');
ros.include('actionlib/commstatemachine');
ros.include('actionlib/goalmanager');
ros.include('actionlib/actionclient');
ros.include('actionlib/simpleactionclient');
