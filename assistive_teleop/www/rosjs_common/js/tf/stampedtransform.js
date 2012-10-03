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

ros.tf.StampedTransform = Class.extend(
/** @lends ros.tf.StampedTransform# */
{
	init: function(){
  // members
	this.matrix = sglIdentityM4();
	this.time = new ros.Time();
	this.frame_id = "";
	this.child_frame_id = "";
	},

	updateFromMessage : function (msg)
	{
		var translationMsg = msg.transform.translation;
		var translation = sglTranslationM4V([translationMsg.x, translationMsg.y, translationMsg.z]);
		var rotationMsg = msg.transform.rotation;
		var rotation = sglGetQuatRotationM4([rotationMsg.x, rotationMsg.y, rotationMsg.z, rotationMsg.w]);
		this.matrix = sglMulM4(this.matrix,translation);
		this.matrix = sglMulM4(this.matrix,rotation);
  
		this.time.updateFromMessage(msg.header.stamp);
		this.frame_id = msg.header.frame_id;
		this.child_frame_id = msg.child_frame_id;
	},

	fromMatrix : function (matrix)
	{
		var translationMsg = msg.transform.translation;
		var translation = sglTranslationM4V([translationMsg.x, translationMsg.y, translationMsg.z]);
		var rotationMsg = msg.transform.rotation;
		var rotation = sglGetQuatRotationM4([rotationMsg.x, rotationMsg.y, rotationMsg.z, rotationMsg.w]);
		this.matrix = sglMulM4(this.matrix,translation);
		this.matrix = sglMulM4(this.matrix,rotation);
  
		this.time.updateFromMessage(msg.header.stamp);
		this.frame_id = msg.header.frame_id;
		this.child_frame_id = msg.child_frame_id;
	},

});





