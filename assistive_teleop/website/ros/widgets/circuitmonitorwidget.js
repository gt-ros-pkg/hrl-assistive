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

CircuitMonitorWidget = ros.widgets.Widget.extend({
  init: function(domobj) {
    this._super(domobj);
    var tool = 
      ['<div class="tool">',
        '<div id="runstop"></div>',
      '</div>'];
    this.jquery.append(tool.join(''));
    this.jquery.menu();
    this.valid = false;
    
    this.state_map = {
        0: "<span class='powerboard_runstop_false'>No Power</span>",
        1: "<span class='powerboard_runstop_standby'>Standby</span>",
        2: "<span class='powerboard_runstop_false'>Pumping</span>",
        3: "<span class='powerboard_runstop_true'>On</span>",
        4: "<span class='powerboard_runstop_false'>Disabled</span>"
      };
  },

  check: function() {
    if (!this.valid) {
      this.jquery.find("#runstop").css({backgroundPosition: '0 0'});
      this.jquery.find(".menu_pane").html("<b>Power Controls</b><br />no valid data");
    }
  },

  receive: function(topic, msg) {
    this.valid = msg.power_board_state_valid;
    // 1. set circuit breakers
    var states = msg.power_board_state.circuit_state;
    var circuit_state = "green";
    str = "<b>Circuit Breakers</b><br />";
    for (var i = 0; i < 3; ++i) {
      var circuit_id = "#circuit_" + i;
      var x_pos = '' + 90 - i*30;
      var y_pos = '0';

      if (states[i] == 3) {
      } else {
    if (states[i] == 1) {
        if (circuit_state != "red") {
      circuit_state = "yellow";
        }
    } else {
        circuit_state = "red";
    }
      }
    }

    str += "Left:&nbsp;" + this.state_map[states[0]] + ' ';
    str += "Base:&nbsp;" + this.state_map[states[1]] + ' ';
    str += "Right:&nbsp;" + this.state_map[states[2]];

    if (circuit_state != "green") {
      str += "<br /><br />One or more of the circuit breakers is not on. ";
      str += "Go to the" +
        "<a href='webui/powerboard.py'>Power Controls</a>" +
        "page to reset the circuits.";
    }

    // 2. set run stop
    str += "<br /><br /><b>Run-stop</b><br />";
    str += "Robot Run-stop: " + (msg.power_board_state.run_stop == "1" ? "<span class='powerboard_runstop_true'>Running</span>" : "<span class='powerboard_runstop_false'>Stopped</span>") + "<br />";
    //str += "Wireless Brake: " + (msg.power_board_state.wireless_stop == "1" ? "<span class='powerboard_runstop_true'>OFF</span>" : "<span class='powerboard_runstop_false'>ON</span>");
    if (msg.power_board_state.run_stop == "1" && msg.power_board_state.wireless_stop == "1" && circuit_state != "red") {
      if (circuit_state == "green") {
          this.jquery.find("#runstop").css({backgroundPosition: '0px 90px'});
          //jQuery("#all_nav").css("border", "none");
      } else if (circuit_state == "yellow") {
          this.jquery.find("#runstop").css({backgroundPosition: '0px 60px'});
          //jQuery("#all_nav").css("border", "1px solid #ff4");
      }
    } else {
      this.jquery.find("#runstop").css({backgroundPosition: '0px 30px'});
      str += "<br />When the run-stop is stopped, the robot can not run. ";
      str += "Go to the" +
        "<a href='webui/powerboard.py'>Power Controls</a>" +
        "page to reset the run-stop.";
      //jQuery("#all_nav").css("border", "1px solid #f44");
    }
    this.jquery.find(".menu_pane").html(str);
  }
});