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
 * A class to create and handle a ROS node.  This class provides functions to handle the interface with rosbridge.
 * 
 * @class
 * @augments Class
 */
ros.NodeHandle  = Class.extend(
/** @lends ros.NodeHandle# */	
{
  /**
   * Constructs a NodeHandler and connects to rosbridge
   * 
   *  @param rosjs_url The address of the websocket to connect to on the robot system.
   *    
   */

  init: function(rosjs_url)
  {
    // members
    this.url = rosjs_url;
    this.rosjs = new ros.Connection(rosjs_url);
    this.is_connected = true;
  },

  /**
   * Creates a new Publisher
   * 
   *  @param topic String indicating the name the Publisher will publish messages to.
   *  @param topic_type String signifying the type of the messages to be published
   *  
   */  
  advertise: function (topic, topic_type) {
    return new ros.Publisher(this, topic, topic_type);
  },
  
  /**
   * Publishes a message over the 
   * 
   *  @param topic String indicating the name of the topic the message will be published to. 
   *  @param typeStr String indicating the type of the message to be published
   *  @param json The message to be published in json format
   *  	
   * 
   */
  publish: function (topic, typeStr, json) {
    this.rosjs.publish(topic, typeStr, json);
  },
  
  /**
   * Subscribes to a topic 
   * 
   * May be called in the following ways
   * subscribe(topic, callback)
   * subscribe(topic, type, callback, delay)  if delay is not specified a default value will be given
   *
   * topic - String indicating the name of the topic a node should subscribe to
   * callback - Function for handling received messages
   * type - String indicating type of topic
   * delay - Optional parameter specifying how much time in miliseconds you want to pass between receiving message
   *
   *         If messages occur more often than that, old messages will be dropped and only the "latest" message will be sent.  
   *         delay=0 means you want to receive messages whenever they are published but don't mind if messages are dropped if 
   *         the websocket bandwidth cannot handle this rate
   *         delay =-1 means that you never want to drop a message, even if this means building a queue
   *         delay is set to 0 if the parameter is not specified
   *  	
   * 
   */
// overloaded (topic, callback) or (topic, type, callback, delay) 
  subscribe: function(){ 
    if(arguments.length==2)
    {
	topic=arguments[0];
	callback=arguments[1];
	delay=0;
	type=false;

    }
    else if(arguments.length==3){
	topic=arguments[0];
	type=arguments[1];
	callback=arguments[2];
	delay=0;
    }
    else if(arguments.length==4){
	topic=arguments[0];
	type=arguments[1];
	callback=arguments[2];
	delay=arguments[3];
    }
    
    console.log(type);
    if(!type){
	this.rosjs.callService('/rosjs/subscribe' ,JSON.stringify([topic,delay]),function(e) {});
    }
    else{
	this.rosjs.callService('/rosjs/subscribe' ,JSON.stringify([topic,delay, type]),function(e) {});
    }
    //this.rosjs.callService('/rosjs/subscribe' ,JSON.stringify([topic,0]),function(e) {});

    console.log('add handler');  
    this.rosjs.addHandler(topic,callback);
    //this.rosjs.addHandler(topic,function(msg) {callback(msg);});
  },
  
  /**
   * Stops receiving messages on the specified topic 
   * 
   * 
   */
  unsubscribe: function (topic)
  {
    this.rosjs.callService('/rosjs/unsubscribe' ,JSON.stringify([topic]),function(e) {});
    //this.rosjs.callService('/rosjs/subscribe' ,JSON.stringify([topic,0]),function(e) {});
  },
  
  /**
   * Creates a serviceclient 
   * 
   * 
   */
  serviceClient: function (service_name)
  {
    return new ros.ServiceClient(this, service_name);
  },
  
  /**
   * Executes callback if websocket connection is not connected/disconnected
   * 
   * 
   */
  setOnClose: function (callback)
  {
    this.rosjs.setOnClose(callback);
    //this.is_connected = false;
  },
  
  /**
   * Executes callback if websocket connection is has an error
   * 
   * 
   */
  setOnError: function (callback)
  {
    this.rosjs.setOnError(callback);
  },
  
  /**
   * Executes callback if websocket connection is connected
   * 
   * 
   */
  setOnOpen: function (callback)
  {
    this.rosjs.setOnOpen(callback);
  },
 
  /**
   * Executes callback when message is received through websocket connection 
   * 
   * 
   */
  setOnMessage: function (callback)
  {
    this.rosjs.setOnMessage(callback);
  },
  
  /**
   * Returns True if node is connected to rosbridge via websocket connection, else false
   * 
   * 
   */
  ok: function () {
    return this.is_connected;
  },
  
  /**
   * Gets the value of a ROS param from the Parameter Server. 
   * 
   *  @param String of the param whose value you wish to get
   *  @callback Function to handle the received information
   * 
   */
  getParam: function (param, callback) {
    this.rosjs.callService('/rosjs/getParameter' ,ros.json([param,0]), callback);
  },
  
  /**
   * Waits to receive one message from a topic 
   * 
   *  @topic String of the topic name
   *  @timeout How long you are willing to late
   *  @callback Function to handle message
   * 
   */
  waitForMessage: function (topic, timeout, callback) {
    this.rosjs.callService('/rosjs/waitForMessage' ,ros.json([topic,timeout,0]), callback);
  },

  /**
   * Requests a list of all currently available topics.
   * 
   *  @callback Function to handle returned list
   * 
   */
  getTopics: function (callback) {
    this.rosjs.callService('/rosjs/topics', JSON.stringify([]), callback);
  },
  
  /**
   * Requests a list of all currently available services
   * 
   *  @callback Function to handle returned list
   * 
   */
  getServices: function (callback) {
    this.rosjs.callService('/rosjs/services', JSON.stringify([]), callback);
  },   

});
