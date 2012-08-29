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

BatteryMonitorWidget = ros.widgets.Widget.extend({
  init: function(domobj) {
    this._super(domobj);
    this.topics = ['/dashboard_agg']
    this.img = document.createElement('img');
    jQuery(this.img).attr({'src' : 'images/toolbar/battery_gray.png', 'width': 41, 'height': 16});
    this.canvas = document.createElement('canvas');
    jQuery(this.canvas).attr( {'width': this.img.width, 'height': this.img.height});
    this.domobj.appendChild(this.canvas);
    this.span = document.createElement('span');
    jQuery(this.span).attr({ style : "font-size:11px;position:relative;top:-3" });
    this.domobj.appendChild(this.span);
    //this.img.onload = this.draw.bind(this);
    this.jquery.menu();
    this.recharge_active = false;
    this.dialog_active = false;
    this.valid = false;
  },

  check: function() {
    if (!this.valid) {
      this.jquery.find("img, canvas, span").css({"-moz-opacity": 0.4});   
      this.span.innerHTML = " n/a";
      this.jquery.find(".menu_pane").html("<b>Battery</b><br />no valid data")      
    }
  },

  draw: function() {
    if (this.img.complete) {
      var ctx = this.canvas.getContext('2d');  
      try {
        ctx.drawImage(this.img,0,0);  
      } catch(e) {
        this.span.innerHTML = ' ' + this.percent + '%';
        return;
      }
      var width = 32 * this.percent / 100;
      ctx.beginPath();  
      ctx.moveTo(38-width, 1);  
      ctx.lineTo(38, 1);  
      ctx.lineTo(40, 4);  
      ctx.lineTo(40,11);  
      ctx.lineTo(38,15);  
      ctx.lineTo(38-width,15);  
      ctx.lineTo(38-width,1);  
      if (this.percent > 30)
        ctx.fillStyle = "rgb(0, 200, 0)";
      else
        ctx.fillStyle = "rgb(200, 0, 0)";
      ctx.fill();  
    }
    this.span.innerHTML = ' ' + this.percent + '%';
  },

  dialog_check: function(self) {
    if(!self.dialog_active && self.valid && self.percent < 30.0 && !self.recharge_active && !(jQuery.cookie("dismissed_warning" + HDF.CGI.Login)) && !self.ac_present){
      self.dialog_active = true;
      dialogMessage("Battery Status", "The robot's battery is at " + self.percent + "% charge. It is recommended that you plug the robot in at anything below 30%.",
      {
        buttons: {
          "Run Recharge Application": function() {
          //jQuery.cookie("dismissed_warning" + HDF.CGI.Login, true, {path: HDF.CGI.BaseURI});
    self.dialog_active = false;
          document.location.href = HDF.CGI.BaseURI + "webui/appinfo.py?taskid=pr2_recharge_application/pr2_recharge_application";
          },
          "Live in Danger": function() {
         // jQuery.cookie("dismissed_warning" + HDF.CGI.Login, true, {path: HDF.CGI.BaseURI});
          self.dialog_active = false;
          jQuery(this).dialog("close");
          }
        }
      });
    }
  },

  receive: function(topic, msg) {
    if(topic == "/dashboard_agg"){
      this.valid = msg.power_state_valid;
      if(this.valid && msg.power_state.relative_capacity != null) {
        this.percent = parseInt(msg.power_state.relative_capacity);
        this.ac_present = parseInt(msg.power_state.AC_present);

        this.draw();
        this.writeMenuTable(msg.power_state);
        this.jquery.find("img, canvas, span").css({"-moz-opacity": 1});   
      }

//      var self = this;
//      gPump.service_call2("list_tasks", {}, 
//      function(task_list) {
//        var recharge_active = false;
//        for(var i=0; i < task_list.tasks.length; ++i){
//          if(task_list.tasks[i] == "PR2 Recharge"){
//      recharge_active = true;
//            jQuery.cookie("dismissed_warning" + HDF.CGI.Login, null, {path: HDF.CGI.BaseURI});
//          }
//        }
//        self.recharge_active = recharge_active;
//  self.dialog_check(self);
//      });
    }
  },
});
