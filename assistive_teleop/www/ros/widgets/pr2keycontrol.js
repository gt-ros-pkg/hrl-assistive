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

ros.widgets.PR2KeyControl = Class.extend({
    init: function(node, left_armmessage, left_armtype, right_armmessage, right_armtype, basemessage, basetype, left_grippermessage, right_grippermessage,grippertype, headmessage, headtype, head_currpose, tf, source_frame, lefthand_target, righthand_target)
  {
    // members
    this.node = node;
    this.armmessage;
    this.armtype;
      this.grippermessage;
      this.grippertype = grippertype;
    this.left_armmessage=left_armmessage;
    this.left_armtype=left_armtype;
    this.right_armmessage=right_armmessage;
    this.right_armtype=right_armtype;
    this.headmessage=headmessage;
    this.headtype=headtype;
    this.basemessage=basemessage;
    this.basetype=basetype;
    this.tf=tf;
      this.source_frame=source_frame;
      this.lefthand_target=lefthand_target;
      this.righthand_target=righthand_target;
      this.left_grippermessage=left_grippermessage;
      this.right_grippermessage=right_grippermessage;


    this.z= 0;
    this.x = 0;
    this.headX = 0;
    this.headY = 0,
      this.currheadX = 0;
      this.currheadY = 0,

    this.headDeltaX = 0;
    this.headDeltaY = 0;
    this.handX = 0.4;
    this.handDeltaX = 0;
    this.handY = 0.2;
    this.handDeltaY = 0;
    this.handZ = 0.2;
    this.handDeltaZ = 0;
    this.handClosed = false;
    this.handToggle = true;
    this.handXr = .4;
    this.handYr = .2;
    this.handZr = .2;
    this.baseX = 0;
    this.baseZ = 0;
    this.leftcallback=false;
    this.rightcallback=false;
    this.headcallback=false;
    this.firsttime=true;
    this.initializedcallback=null;
    this.json = function(obj) {return JSON.stringify(obj);};
    this.intval = "";
      
    // add event handler
    that = this;
    $('body').keydown(function(e) {that.handleKey(e.keyCode,true);});
    $('body').keyup(function(e) {that.handleKey(e.keyCode,false);});

    this.leftX=.4;
    this.leftY=.2;
    this.leftZ=.2;
    this.rightX=.0;
    this.rightY=0;
    this.rightZ=0;
      
    var that=this;

/*    this.l_arm_currpose=this.node.subscribe(l_arm_currpose, function(msg){
	var position=msg.pose.position;
//	console.log(position.x)
	that.leftX=position.x;
	that.leftY=position.y;
	that.leftZ=position.z; 
	that.leftcallback=true;
	//console.log(that.firsttime)
	if(that.rightcallback==true && that.headcallback==true &&that.firsttime==true && that.initializedcallback) 
	{that.initializedcallback();}
    });


      this.r_arm_currpose=this.node.subscribe(r_arm_currpose, function(msg){
	  var position=msg.pose.position;
	//  console.log(that.rightX)
	  that.rightX=position.x;
	  that.rightY=position.y;
	  that.rightZ=position.z;
	  that.rightcallback=true;
	  //console.log(that.firsttime)
	  if(that.leftcallback==true && that.headcallback==true && that.firsttime==true && that.initializedcallback)
	  {that.initializedcallback();}
      });
    */
      this.curr_headpose=this.node.subscribe(head_currpose, function(msg){
	  var position=msg.desired.positions;
	  //console.log(msg);
	  that.currheadX=position[0];
	  that.currheadY=position[1];
	  that.headcallback=true;
          document.log(that.currheadX);
	  if(that.leftcallback==true &&  that.rightcallback==true && that.firsttime==true && that.initializedcallback)
	  {that.initializedcallback();}
      });

  },
 
  start: function(arm)
  {
      that=this;
      console.log("started keycontrol");
      this.switch_arm(arm);
      this.keyControlOn=true;   
      if(this.intval=="") {
	  this.intval = setInterval(function() {that.handler(); }, 50);
	  
	  //  console.log(this.lefthand_target);
	  //  console.log(this.source_frame);
	  
	  this.headX=this.currheadX;
	  this.headY=this.currheadY;
	  
	  console.log(this.currheadX + " " + this.currheadY);
	  //SET VALUES
	  
	  
      
      }
  },
 
  stop: function()
  {
      this.keyControlOn=false;
      console.log('stop please');
    if(this.intval!="") {
	clearInterval(this.intval);
	this.intval="";
	console.log("stopped keycontrol");
    }
  },
  
    setKeyControlOn:function()
    {
	this.keyControlOn=true;
    },
    
  switch_arm:function(arm)
  {      
      if(arm=='l') {
	
	  this.armmessage=this.left_armmessage;
	  this.armtype=this.left_armtype;

	  this.grippermessage=this.left_grippermessage;
	  leftarmTf=this.tf.lookupTransform(this.lefthand_target, this.source_frame);
	  
	  console.log(leftarmTf);
	  this.handX=leftarmTf.translation.x;//this.leftX;
	  this.handY=leftarmTf.translation.y;//this.leftY;
	  this.handZ=leftarmTf.translation.z;//this.leftZ;
	  console.log(this.handX + " " + this.handY + " " + this.handZ);
      }
      else {
	  this.armmessage=this.right_armmessage;
	  this.armtype=this.right_armtype;

	  this.grippermessage=this.right_grippermessage;
	  //alert(this.leftX);

	  rightarmTf=this.tf.lookupTransform(this.righthand_target, this.source_frame);
	  
	  this.handX=rightarmTf.translation.x;//this.rightX;
	  this.handY=rightarmTf.translation.y;//this.rightY;
	  this.handZ=rightarmTf.translation.z;//this.rightZ;
	  
      }
  },
    
  setOnInitialized:function(e)  {
      this.initializedcallback=e;
  },
 
    set_firsttime:function(){
	console.log("setting first time");
	this.firsttime=false;
    },
    
  handleKey: function (code, down)
  {
//      if(this.keyControlOn){	  
	  ros_debug('Key Code ' + code);
	
  	  switch(code) {
  	  case 86:
  	      //pan left
  	      if (down) {
  		  this.headDeltaX = .01;
  	      } else {
  		  this.headDeltaX = 0;
  	      }
  	      break;
  	  case 66:
  	      //pan right
  	      if (down) {
  		  this.headDeltaX = -.01;
  	      } else {
  		  this.headDeltaX = 0;
  	      }
  	      break;
  	  case 78:
  	      //tilt up
  	      if (down) {
  		  this.headDeltaY = .01;
  	      } else {
  		  this.headDeltaY = 0;
  	      }
  	      break;
  	  case 77:
  	      //tilt down
  	      if (down) {
  		  this.headDeltaY = -.01;
  	      } else {
  		  this.headDeltaY = 0;
  	      }
  	      break;
  	  case 87:
  	      if (down) {
  		  this.handDeltaX = .01;
  	      } else {
  		  this.handDeltaX = 0;
  	      }
  	      break;
  	  case 83:
  	      if (down) {
  		  this.handDeltaX = -.01;
  	      } else {
  		  this.handDeltaX = 0;
  	      }
  	      break;
  	  case 68:
  	      if (down) {
  		  this.handDeltaY = -0.01;
  	      } else {
  		  this.handDeltaY = 0;
  	      }
  	      break;
  	  case 65:
  	      if (down) {
  		  this.handDeltaY = 0.01;
  	      } else {
  		  this.handDeltaY = 0;
  	      }
  	      break;
  	  case 81:
  	      if (down) {
  		  this.handDeltaZ = 0.01;
  	      } else {
  		  this.handDeltaZ = 0;
  	      }
  	      break;
  	  case 69:
  	      if (down) {
  		  this.handDeltaZ = -0.01;
  	      } else {
  		  this.handDeltaZ = 0;
  	      }
  	      break;
  	  case 37:
  	      if (down) {
  		  this.baseZ = 1;
  	      } else {
  		  this.baseZ = 0;
  	      }
  	      break;
  	  case 38:
  	      if (down) {
  		  this.baseX = .5;
  	      } else {
  		  this.baseX = 0;
  	      }
  	      break;
  	  case 39:
  	      if (down) {
  		  this.baseZ = -1;
  	      } else {
  		  this.baseZ = 0;
  	      }
  	      break;
  	  case 40:
  	      if (down) {
  		  this.baseX = -.5;
  	      } else {
  		  this.baseX = 0;
  	      }
  	      break;
  	  case 67:
  	      if (down) {
  		  this.handToggle = true;
  		  if (this.handClosed) {
  		      this.handClosed = false;
  		  } else {
  		      this.handClosed = true;
  		  }
  	      }
  	      break;
  	  case 81:
  	      if (down) {
  		
  		  this.handX = Math.round(this.handXr*10)/10;
  		  this.handY = Math.round(this.handYr*10)/10;
  		  this.handZ = Math.round(this.handZr*10)/10;
  	      }
  	      break;
  //	  }
	//  console.log("hand x " + this.handX);
	//  this.handler();
      }
  },

  look: function (x,y)
  {
    this.node.publish(this.headmessage, this.headtype, this.json(
        	    {
          		'joint_names':["head_pan_joint", "head_tilt_joint"],
          		'points':[{
          		    'positions':[x,y],
          		    'velocities':[0.0, 0.0],
          		    'time_from_start':{'nsecs':0,'secs':0},
          		}]
        	    }
    ));
  },
    
  hand: function(x,y,z,ox,oy,oz,ow) 
  {
      //      console.log('publishing');
     this.node.publish(this.armmessage,this.armtype,this.json(
               {
                 'header':{'frame_id':'torso_lift_link'}, //may need time
                 'pose':{
                 'position':{'x':x,'y':y,'z':z},
                 'orientation':{'x':ox,'y':oy,'z':oz,'w':ow},
                 }
             }
     ));
  },
    
  handler: function() 
  {
  	//	console.log('sigh');
  	this.headX += this.headDeltaX;
  	this.headY += this.headDeltaY;
  	if (this.headX < -2.8) {
  	    this.headX = -2.8;
  	} else {
  	    if (this.headX > 2.8) this.headX = 2.8;
  	}
  	
  	if (this.headY < -.24) {
  	    this.headY = -.24;
  	} else {
  	    if (this.headY > 1.16) this.headY = 1.16;
  	}
  	
  	//	console.log(this.headX);
  	this.look(this.headX,this.headY);	
  
  
  	this.handX += this.handDeltaX;
  	this.handY += this.handDeltaY;
  	this.handZ += this.handDeltaZ;
  	
  	this.hand(this.handX, this.handY, this.handZ,0,0,0,1);
  	
  	this.node.publish(this.basemessage, this.basetype,this.json(
  	    {'linear':	{'x': this.baseX,
  			 'y': 0,
  			 'z': 0},
  	     'angular':	{'x': 0,
  			 'y': 0,
  			 'z': this.baseZ}}
  	));
  	
  	if (this.handToggle) {
  	    this.handToggle = false;
  	    var position = 0.08;
  	    if (this.handClosed) {
      		position = -100.00;
      		this.handClosed = true;
  	    }
  	    this.node.publish(this.grippermessage, this.grippertype, this.json(
		//'pr2_controllers_msgs/Pr2GripperCommand', this.json(
                {
                    'position':position,
                    'max_effort':-1.0,
                }
  	    ));
  	}
	
  },

   
  
});
	  
