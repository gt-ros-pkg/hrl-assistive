/********************************************************************
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

ros.NodeHandle  = Class.extend({
  init: function(rosjs_url)
  {
    // members
    this.url = rosjs_url;
    this.rosjs = new ros.Connection(rosjs_url);
    this.is_connected = true;
  },

  advertise: function (topic, topic_type) {
    return new ros.Publisher(this, topic, topic_type);
  },
  
  publish: function (topic, typeStr, json) {
    this.rosjs.publish(topic, typeStr, json);
  },
  
  subscribe: function (topic, callback)
  {

   // this.rosjs.addHandler(topic,function(msg) {callback(msg);});
      this.rosjs.callService('/rosjs/subscribe' ,JSON.stringify([topic,0]),function(e) {});
   // this.rosjs.callService('/rosjs/subscribe' ,JSON.stringify([topic,0]),function(e) {});
    this.rosjs.addHandler(topic,function(msg) {callback(msg);});
  },
  
  unsubscribe: function (topic, callback)
  {
    this.rosjs.callService('/rosjs/subscribe' ,JSON.stringify([topic,0]),function(e) {});
    //this.rosjs.callService('/rosjs/subscribe' ,JSON.stringify([topic,0]),function(e) {});
    this.rosjs.addHandler(topic,function(msg) {callback(msg);});
  },
  
  serviceClient: function (service_name)
  {
    return new ros.ServiceClient(this, service_name);
  },
  
  setOnClose: function (callback)
  {
    this.rosjs.setOnClose(callback);
    //this.is_connected = false;
  },
  
  setOnError: function (callback)
  {
    this.rosjs.setOnError(callback);
  },
  
  setOnOpen: function (callback)
  {
    this.rosjs.setOnOpen(callback);
  },
 
  setOnMessage: function (callback)
  {
    this.rosjs.setOnMessage(callback);
  },
  
  ok: function () {
    return this.is_connected;
  },
  
  getParam: function (param, callback) {
    this.rosjs.callService('/rosjs/getParameter' ,ros.json([param,0]), callback);
  },
  
  // Receive one message from topic.
  waitForMessage: function (topic, timeout, callback) {
    this.rosjs.callService('/rosjs/waitForMessage' ,ros.json([topic,timeout,0]), callback);
  },
   
    
});
