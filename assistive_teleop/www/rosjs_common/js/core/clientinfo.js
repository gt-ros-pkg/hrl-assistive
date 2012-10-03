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

ros.NodeHandle = function(rosjs_url)
{
  // members
}

ros.NodeHandle.prototype.subscribe = function (topic, callback)
{
  if(!this.connected) {
    
    this.rosjs = new ROS(this.url);
    
    var that = this;
    this.rosjs.setOnClose(function(e) {
      ros_debug("Disconnected or Can't Connect.");
      that.disconnected = true;
    });
    this.rosjs.setOnError(function(e) {
      ros_debug("Unknown error!");
      that.error = true;
    });
    this.rosjs.setOnOpen(function(e) {   
      ros_debug("Connected to " + that.url + ".");
      that.connected = true;
      // subscribe
      that.rosjs.callService('/rosjs/subscribe' ,JSON.stringify([topic,0]),function(e) {});
      that.rosjs.callService('/rosjs/subscribe' ,JSON.stringify([topic,0]),function(e) {});
      that.rosjs.addHandler(topic,function(msg) {callback(msg);});
    });
  }
  else 
  {
    // subscribe
    this.rosjs.callService('/rosjs/subscribe' ,JSON.stringify([topic,0]),function(e) {});
    this.rosjs.callService('/rosjs/subscribe' ,JSON.stringify([topic,0]),function(e) {});
    this.rosjs.addHandler(topic,function(msg) {callback(msg);});
  }
}
