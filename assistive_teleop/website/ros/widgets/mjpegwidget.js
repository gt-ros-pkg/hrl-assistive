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

ros.widgets.MJPEGWidget = Class.extend({
    
  init:function(widget_div_id, width, height, mjpeg_uri){
  	ros_debug(widget_div_id);
  
  	this.iframe_id="MJPEG_IFRAME";
  	this.widget_div_id=widget_div_id;
  	this.iframe_div_id="iframe_div";
  	this.iframe_width=width+10;
  	this.iframe_height=height+10;
  	
  	//this.divHTML="";
  	
  	//create the html
  	this.createdivHTML();
  	//set up the canvas id image stuff
  
  	this.iframe=document.getElementById(this.iframe_id);
  	this.iframe.src=mjpeg_uri;
  },

  createdivHTML: function(){
    //	iframe_text="<iframe id=\""+this.iframe_ID+"\" width=\""+ this.iframe_width + "\"  height=\""+this.iframe_height + "\"></iframe>";
    //	img_text="<img id=\""+this.iframe_id+"\" width=\""+ this.iframe_width + "\"  height=\""+this.iframe_height + "\"></img>";
  	img_text="<iframe id=\""+this.iframe_id+"\" width=\""+ this.iframe_width + "\"  height=\""+this.iframe_height + "\" autoplay></iframe>";
  	//	console.log(iframe_text);
  	text="<div id=\""+this.iframe_div_id+"\">"+img_text + " </div>";
  	
  	$('#'+this.widget_div_id).html(text);
  	//	console.log($('#'+this.widget_div_id).html());
  },
    
   /* createWebHtml: function(address){
	webstring_header=" <!DOCTYPE html>  <html>  <head> <meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\"> </head> } <body>";
	webstring_footer="</body></html>";
	img_string="<img id=\"MJPEG_IFRAME\" width=\"640\" height=\"480\" src=\""+
	    address + "\"></img>";

//http://shelob:8080/?topic=/remote_lab_cam0/image_raw"></img>
	
    },*/
    
    setsrc: function(mjpeg_uri){
	this.iframe.src=mjpeg_uri;
    },
    setcanvas:function(canvasID, mjpeg_uri){
	this.iframe_id;
	this.mjpeg_uri=mjpeg_uri;
	this.iframe = document.getElementById(canvasID);
	
	this.iframe.src = mjpeg_uri;
    },
    getnewimage:function(mjpeg_uri){
	
//	this.iframe.src="img/remote_lab_header_small.jpg";
//	$('#'+this.iframe_id).remove();

//	this.createdivHTML();
		
//	this.setcanvas(this.iframe_id, mjpeg_uri);
//	console.log($('#'+this.widget_div_id).html());
 
	this.iframe=document.getElementById(this.iframe_id);
	this.iframe.src=mjpeg_uri;

   },
});
