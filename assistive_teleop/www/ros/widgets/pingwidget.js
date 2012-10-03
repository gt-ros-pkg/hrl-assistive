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

PingWidget = ros.widgets.Widget.extend({
  init: function(domobj) {
    this._super(domobj);
    this.ping_intervall = 1000; // send a ping every second
    this.curPing = new ros.Time();
    this.ping_topic = "/ping";
    this.ping_message_type = "ping/Ping";
    this.hash = 0; 
    var that = this;
    setTimeout(function() {
      that.ping(); 
    }, this.ping_intervall);
  },

  check: function() {
  },
  
  receive: function(topic, msg) {
    if(this.hash == msg.hash) {
      this.curPing = (new ros.Time).now();
      var sent = new ros.Time();
      sent.updateFromMessage(msg.stamp);
      this.curPing.subtract(sent);
      var str = "Ping: " + Math.round(this.curPing.toMilliseconds()) + " ms";
      this.domobj.innerHTML = str;
      
      var that = this;
      setTimeout(function() {
        that.ping(); 
      }, this.ping_intervall);
    }
  },
  
  ping: function(topic, msg) {
    var now = (new ros.Time()).now();
    this.hash = Math.floor(Math.random()*10000);
    this.manager.node.publish(this.ping_topic, this.ping_message_type , ros.json({'stamp':now, 'hash': this.hash }));
  },
  
});  