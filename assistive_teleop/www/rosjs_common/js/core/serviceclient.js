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
*********************************************************************/
/**
 * A class to create and handle a ServiceClient.
 * 
 * @class
 * @augments Class
 */
ros.ServiceClient = Class.extend(
/** @lends ros.ServiceClient# */	
{

/**
 * Constructs a ServiceClient and initializes parameters
 * 
 *  @param node NodeHandler of the node connected to the websocket of the system with the desired service 
 *  @param service_name String containing the name of the service the client will call.
 *    
 */
 init: function(node, service_name) 
 {
   this.node = node;
   this.service_name = service_name;
 },

 /**
  * Calls the service
  * 
  *  @param service_message json message containing the service request 
  *  @param callback Function to handle the service response
  *    
  */
 call: function(service_message, callback) 
 {
   this.node.rosjs.callService(this.service_name, service_message, callback);
 },

 /**
  * Waits for the service to be available  Currently a stub function for to be added later 
  * 
  *  @param timeout How long to wait 
  *  @param callback Function to be performed once service is available 
  *    
  */
 wait_for_service:function(timeout, callback)
 {
     /*unfinished stub function*/

     var timeout_time=(new Date()).getTime()+timeout*1000;

     //check every 100ms to see if the status message has arrived
     var that=this;
     callback(true);
     
     
 },

});
