/*******************************************************************************
 * 
 * Software License Agreement (BSD License)
 * 
 * Copyright (c) 2010, Robert Bosch LLC. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met: *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer. * Redistributions in binary
 * form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials provided
 * with the distribution. * Neither the name of the Robert Bosch nor the names
 * of its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 ******************************************************************************/

MotorsMonitorWidget = ros.widgets.Widget.extend({
  init: function(domobj) {
    this._super(domobj);
    this.jquery.append('<div class="tool"></div>');
    this.jquery.menu();
    this.valid = false;
  },
  
  check: function() {
    if (!this.valid) {
      this.jquery.find(".tool").css({backgroundPosition: '0px 0px'});
      this.jquery.find(".menu_pane").html("<b>Motors</b><br />no valid data")
    }
  },

  receive: function(topic, msg) {
    // response is for "motors halted"
    this.valid = msg.motors_halted_valid;
    if (msg.motors_halted.data) {
      this.jquery.find(".tool").css({backgroundPosition: '0px 30px'});
      this.jquery.find(".menu_pane").html("<b>Motors</b><br />The robot motors have been halted." +
        "When the motors are halted, the robot can not run.<br /><br />" + 
        "Go to the" +
        "<a href='...'>Power Controls</a>" +
        "page to reset the motors.");
    } else {
      this.jquery.find(".tool").css({backgroundPosition: '0px 90px'});
      this.jquery.find(".menu_pane").html("<b>Motors</b><br />The robot motors are running.");
    }
  }
});