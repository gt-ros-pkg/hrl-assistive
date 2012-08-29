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
 * Handles material information for the URDF
 * @class
 * @augments Class
 */
ros.urdf.Material = Class.extend(
/** @lends ros.urdf.Material# */
{
	init: function() {
	
	// members
	this.name = "";
	this.texture_filename = "";
	this.color = new ros.urdf.Color();
	},

// methods
	clear : function () 
	{
		this.name = "";
		this.texture_filename = "";
		this.color.clear();
	},

	initXml : function (xml) 
	{
		var has_rgb = false;
		var has_filename = false;

		this.clear();

		if (!xml.getAttribute("name"))
		{
			ros_error("Material must contain a name attribute");
			return false;
		}

		this.name = xml.getAttribute("name");

		// texture
		var textures = xml.getElementsByTagName("texture");
		if(textures.length>0)
		{
			var texture = textures[0];
			if (texture.getAttribute("filename"))
			{
				this.texture_filename = texture.getAttribute("filename");
				has_filename = true;
			}
			else
			{
				ros_error("texture has no filename for Material " + this.name);
			}
		}

		// color
		var colors = xml.getElementsByTagName("color");
		if(colors.length>0)
		{
			var c = colors[0];
			if (c.getAttribute("rgba"))
			{
				if (!this.color.initString(c.getAttribute("rgba")))
				{
					ros_error("Material " + this.name + " has malformed color rgba values.");
					this.color.clear();
					return false;
				}
				else
					has_rgb = true;
			}
			else
			{
				ros_error("Material " + this.name + " color has no rgba");
			}
		}


//		if(has_rgb == false && has_filename ==false)
//		{
//		ros_error("material xml is not initialized correctly");
//		this.color.clear();
//		has_rgb = true;
//		}

		return (has_rgb || has_filename);
	}
});