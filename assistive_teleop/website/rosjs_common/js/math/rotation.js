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

/**
 * Class to handle rotations
 * @class
 * @augments Class
 */
ros.math.Rotation = Class.extend(
/** @lends ros.math.Rotation# */
{
	
/**
 * Initializes to empty pose 
 */
		
	init : function() {
  
		// members
		this.x = 0;
		this.y = 0;
		this.z = 0;
		this.w = 1.0;
	},

	/**
	 * Sets pose to be empty
	 */
	clear : function ()
	{
		this.x = 0;
		this.y = 0;
		this.z = 0;
		this.w = 1.0;
	},

	/**
	 * Initialize the pose given the rotation in a string
	 */
	
	initString : function (str)
	{
		this.clear();
  
		var rpy = new ros.math.Vector3();
  
		if (!rpy.initString(str))
			return false;
		else
		{
			this.setFromRPY(rpy.x,rpy.y,rpy.z);
			return true;
		}
		delete rpy;
  
		return true;
	},

	/**
	 * Set from roll, pitch, yaw
	 */
	setFromRPY : function(roll, pitch, yaw)
	{
		var phi = roll / 2.0;
		var the = pitch / 2.0;
		var psi = yaw / 2.0;

		this.x = Math.sin(phi) * Math.cos(the) * Math.cos(psi) - Math.cos(phi) * Math.sin(the) * Math.sin(psi);
		this.y = Math.cos(phi) * Math.sin(the) * Math.cos(psi) + Math.sin(phi) * Math.cos(the) * Math.sin(psi);
		this.z = Math.cos(phi) * Math.cos(the) * Math.sin(psi) - Math.sin(phi) * Math.sin(the) * Math.cos(psi);
		this.w = Math.cos(phi) * Math.cos(the) * Math.cos(psi) + Math.sin(phi) * Math.sin(the) * Math.sin(psi);

		this.normalize();
	},

	normalize : function()
	{
		var s = Math.sqrt(this.x * this.x +
				this.y * this.y +
				this.z * this.z +
				this.w * this.w);
		if (s == 0.0)
		{
			this.x = 0.0;
			this.y = 0.0;
			this.z = 0.0;
			this.w = 1.0;
		}
		else
		{
			this.x /= s;
			this.y /= s;
			this.z /= s;
			this.w /= s;
		}
	},
});
