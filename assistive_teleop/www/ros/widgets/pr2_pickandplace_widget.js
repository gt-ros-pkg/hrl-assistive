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

ros.widgets.pr2_pickandplace_widget=Class.extend({
    init:function(node, divID, vm){
	this.node=node;
	this.divID=divID;
	this.vm=vm;
	this.pickandplace_manager= new ros.pickandplace.PickAndPlaceManager(node,vm);
//	this.create_initial_divHTML:function();
	
	this.divbuttonID='buttondiv';	

	this.currarm='l';  //start with left
	this.leftArmController=new ros.widgets.pr2_pickandplace_widget_controller(node, vm);
	this.rightArmController=new ros.widgets.pr2_pickandplace_widget_controller(node, vm);
	
	
	this.create_divHTML(0);

	
    },

    
    create_divHTML:function(busy){
	console.log("busy "+busy);

	if(this.currarm=='l')
	{
	    div_text="<input type=\"radio\" name=\"selected_arm\" value=\"l\" checked/> Left Arm </label>   <input type=\"radio\" name=\"selected_arm\" value=\"r\"/> Right Arm </label> ";
	    if(busy==1)
		div_text=div_text+this.leftArmController.create_busybuttonHTML();
	    else
		div_text=div_text+this.leftArmController.create_buttonHTML();
	}
	else
	    {
		div_text="<input type=\"radio\" name=\"selected_arm\" value=\"l\" /> Left Arm </label>   <input type=\"radio\" name=\"selected_arm\" value=\"r\" checked/> Right Arm </label> ";
		if(busy==1)
		   div_text=div_text+this.rightArmController.create_busybuttonHTML();
		else
		    div_text=div_text+this.rightArmController.create_buttonHTML();
	    }
	
	$('#'+this.divID).html(div_text);

	this.setUpCallBacks();

    },


    
    

    setUpCallBacks:function(){
	var that=this;

	jQuery('#detecttable_click').click(function(e){
	    console.log('Detecting table');
	    that.create_divHTML(1);
	    that.detectTable();
	});
	
	jQuery('#detectobj_click').click(function(e){
	    console.log('Detecting objects');
	    that.create_divHTML(1);
	    that.detectObjects();
	});
	

	jQuery('#movearm_click').click(function(e){
	    console.log('moving arm');
	    that.create_divHTML(1);
	    that.moveArmToSide();
	    
	});

	jQuery('#pick_object').click(function(e){
	    console.log('Picking up Object');
	    that.create_divHTML(1);
	    that.pickUpObject();
	    
	});

	jQuery('#place_object').click(function(e){
	    console.log('Placing Object');
	    that.create_divHTML(1);
	    that.placeObject();
	    
	});

	jQuery('#detach_object').click(function(e){
	    console.log('Detaching objects');
	    that.create_divHTML(1);
	    that.detach();
	   // that.create_divHTML();
	});

	jQuery('#refresh_click').click(function(e){
	    console.log('Detaching objects');
	    that.create_divHTML(1);
	    that.detach();
	});

	jQuery("input[name='selected_arm']").change(function(e){
	    console.log('radio button change');
	    arm=jQuery("input[name='selected_arm']:checked").val();
	    
	    that.currarm=arm;
	    that.create_divHTML(0);
      });
	

    },
    
    clearReceivedObjects: function(){
	this.pickandplace_manager.clearReceivedObjects();
    },

    addReceivedObject: function(object){
	this.pickandplace_manager.addReceivedObject(object);
    },

    receiveDetectedObjects:function(objs){
	ros_debug("receiveDetectedObjects");
	console.log("in recieveDetectedObjects");
	
	// clear object list
	// clear pick and place manager objects
	this.clearReceivedObjects();
	//this.pickandplace_manager.clearReceivedObjects();
	console.log(objs);
	var objects=objs.objects;
	//ros_debug(objects)
	for ( var o in objects) {
	    ros_debug(o)
	    var object = objects[o];
	    var obj_ind=object.objectid;
	
	    // add object to pick and place manager
	//    this.pickandplace_manager.addReceivedObject(object);
	    this.addReceivedObject(object);
	}
    },
    
    detectTable: function(){
	var that=this;
	this.pickandplace_manager.detectTable(function(e){
		//if(e.length>0){
		    that.leftArmController.detectTableResult();
		    that.rightArmController.detectTableResult();
		    console.log('sigh');
		    //}
	    that.create_divHTML(0);
	});
	
    },

    detectObjects: function(){
	var that=this;
	console.log('objects');
	this.pickandplace_manager.detectObjects("d", function(e){
	     if(e.objects.length>0){
	    that.receiveDetectedObjects(e);
	    that.leftArmController.detectObjectResult();
	    that.rightArmController.detectObjectResult();
	    console.log('sigh');
	    }
	    that.create_divHTML(0);
	});
	

    },
    pickUpObject: function(){
	var that=this;
	console.log('about to pickup');
	this.pickandplace_manager.pickObjectFromSelection(jQuery("input[name='selected_arm']:checked").val(), function(e){
		console.log( e);
		 if(e.success==true){
	    console.log('in callback');
	    if(that.currarm=='l')
		that.leftArmController.pickUpObjectResult();
	    else
		that.rightArmController.pickUpObjectResult();
	    }
	    that.create_divHTML(0);
	});
	
    },
    
    moveArmToSide:function(){
	var that=this;
	this.pickandplace_manager.moveArm(jQuery("input[name='selected_arm']:checked").val(), function(e){
		if(e.success==true){
	    if(that.currarm=='l')
		that.leftArmController.moveArmToSideResult();
	    else
		that.rightArmController.moveArmToSideResult();
		}
	    that.create_divHTML(0);
	});
    },

    placeObject:function(){
	var that=this;
	this.pickandplace_manager.placeObject(jQuery("input[name='selected_arm']:checked").val(), function(e){
		if(e.success==true){
	    if(that.currarm=='l')
		that.leftArmController.placeObjectResult();
	    else
		that.rightArmController.placeObjectResult();
	    }
	    that.create_divHTML(0);
	});
    },

    detach: function(){
	var that=this;
	this.pickandplace_manager.detachObject(jQuery("input[name='selected_arm']:checked").val(),function(e){
		if(e==true){
	    if(that.currarm=='l')
		that.leftArmController.detachResult();
	    else
		that.rightArmController.detachResult();
		}
	    that.create_divHTML(0);
	});
    },    


});