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

ros.widgets.pr2_pickandplace_widget_controller=Class.extend({
    init:function(node, vm){
	console.log('in function');
	this.node=node;
	
	this.vm=vm;
//	this.create_initial_divHTML:function();
	
	this.divbuttonID='buttondiv';
	
	this.detecttableID="detecttable_click";
	this.detecttable_status=true;
	this.detecttable_on_text=" <p><button type=\"button\", id=\"detecttable_click\" style=\"width: 200px;\"> Detect Table</button></p>";
	this.detecttable_disabled_text=" <p><button type=\"button\", id=\"detecttable_click\" style=\"width: 200px;\" disabled> Detect Table</button></p>";

	this.handContainsObject=false;
	
	this.detectobjectID="detectobject_click";
	this.detectobject_status=false;
	this.detectobject_on_text=" <p><button type=\"button\", id=\"detectobj_click\" style=\"width: 200px;\"> Detect Objects</button></p>";
	this.detectobject_disabled_text=" <p><button type=\"button\", id=\"detectobj_click\" style=\"width: 200px;\" disabled> Detect Objects</button></p>";
	
	
	this.movearmID="movearm_click";
	this.movearm_status=true;
	this.movearm_on_text="<p><button type=\"button\", id=\"movearm_click\"  style=\"width: 200px;\"> Move Arm To Side</button></p>";
	this.movearm_disabled_text="<p><button type=\"button\", id=\"movearm_click\"  style=\"width: 200px;\" disabled> Move Arm To Side</button></p>";
	
	this.pick_objectID="pick_object";
	this.pick_object_status=false;
	this.pick_object_on_text="<p><button type=\"button\", id=\"pick_object\" style=\"width: 200px;\"> Pick Object from Table  </button></p>";
	this.pick_object_disabled_text="<p><button type=\"button\", id=\"pick_object\" style=\"width: 200px;\" disabled> Pick Object from Table  </button></p>";
	
	this.place_objectID="place_object";
	this.place_object_status=false;
	this.place_object_on_text="<p><button type=\"button\", id=\"place_object\" style=\"width: 200px;\"> Place Object on Table  </button></p>";
	this.place_object_disabled_text="<p><button type=\"button\", id=\"place_object\" style=\"width: 200px;\" disabled> Place Object on Table  </button></p>";
	
	this.detach_objectID="detach_object";
	this.detach_object_status=true;
	this.detach_object_on_text="<p><button type=\"button\", id=\"detach_object\" style=\"width: 200px;\"> Detach </button></p>";
	this.detach_object_disabled_text="<p><button type=\"button\", id=\"detach_object\" style=\"width: 200px;\" disabled> Detach </button></p>";

	this.refreshID="refresh_click";
	this.refresh_status=true;
	this.refresh_on_text="<p><button type=\"button\", id=\"refresh_click\" style=\"width: 200px;\"> Refresh</button></p> ";
	this.refresh_disabled_text="<p><button type=\"button\", id=\"refresh_click\" style=\"width: 200px;\" disabled> Refresh</button></p> ";
//	this.create_divHTML();
	
	
    },
    

    
    create_buttonHTML:function(){
	div_text="<div id=\""+this.divbuttonID + "\">";

	if(this.detecttable_status)
	    div_text=div_text+this.detecttable_on_text;
	else
	    div_text=div_text+this.detecttable_disabled_text;

	if(this.detectobject_status)
	    div_text=div_text+this.detectobject_on_text;
	else
	    div_text=div_text+this.detectobject_disabled_text;

	if(this.movearm_status)
	    div_text=div_text+this.movearm_on_text;
	else
	    div_text=div_text+this.movearm_disabled_text;

	if(this.pick_object_status)
	    div_text=div_text+this.pick_object_on_text;
	else
	    div_text=div_text+this.pick_object_disabled_text;
	
	if(this.place_object_status)
	    div_text=div_text+this.place_object_on_text;
	else
	    div_text=div_text+this.place_object_disabled_text;

	if(this.detach_object_status)
	    div_text=div_text+this.detach_object_on_text;
	else
	    div_text=div_text+this.detach_object_disabled_text;
/*	if(this.refresh_status)
	    div_text=div_text+this.refresh_on_text;
	else
	    div_text=div_text+this.refresh_disabled_text;
	div_text=div_text+"</div>";
*/
	return div_text;
//	$('#'+this.divID).html(div_text);

    },



  create_busybuttonHTML:function(){
      div_text="<div id=\""+this.divbuttonID + "\">";
      
      div_text=div_text+this.detecttable_disabled_text;
      div_text=div_text+this.detectobject_disabled_text;
      div_text=div_text+this.movearm_disabled_text;
      div_text=div_text+this.pick_object_disabled_text;
      div_text=div_text+this.place_object_disabled_text;
      div_text=div_text+this.detach_object_disabled_text;
      

	return div_text;
//	$('#'+this.divID).html(div_text);

    },
    
    detectTableResult:function(){
	this.detectobject_status=true;
    },
    
    detectObjectResult:function(){
	if(this.handContainsObject)
	    this.place_object_status=true;
	else
	    this.pick_object_status=true;
	
    },
    
    pickUpObjectResult:function(){
	this.handContainsObject=true;
	this.pick_object_status=false;
	this.detecttable_status=false;
	this.detectobject_status=false;
    },

    moveArmToSideResult:function(){
	//if(this.handContainsObject)
	//    this.place_object_status=true;
	this.detecttable_status=true;
	this.detectobject_status=true;
    },

    placeObjectResult:function(){
	this.handContainsObject=false;
	this.place_object_status=false;
	this.detecttable_status=false;
	this.detectobject_status=false;
    },
    
    detachResult:function(){
	this.handContainsObject=false;
	this.place_object_status=false;
	this.detecttable_status=false;
	this.detectobject_status=false;
	this.pick_object_status=false;
    },

});