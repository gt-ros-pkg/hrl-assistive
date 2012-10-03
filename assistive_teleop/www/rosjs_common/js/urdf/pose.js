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
 *  Stores urdf pose information
 *  @class
 * @augments Class
 */

ros.urdf.Pose =  Class.extend(
/** @lends ros.urdf.Pose# */
{
	init: function() {
  
		// members
		this.position = new ros.urdf.Vector3();
		this.rotation = new ros.urdf.Rotation();
	}, 

// methods
	clear : function ()
	{
		this.position.clear();
		this.rotation.clear();
	},

	initXml : function (xml)
	{
		this.clear();

		var xyz_str = xml.getAttribute("xyz");
		if (!xyz_str)
		{
			ros_debug("parsing pose: no xyz, using default values.");
			return true;
		}
		else
		{
			if (!this.position.initString(xyz_str))
			{
				ros_error("malformed xyz");
				this.position.clear();
				return false;
			}
		}

		rpy_str = xml.getAttribute("rpy");
		if (!rpy_str)
		{
			ros_debug("parsing pose: no rpy, using default values.");
			return true;
		}
		else
		{
			if (!this.rotation.initString(rpy_str))
			{
				ros_error("malformed rpy");
				return false;
				this.rotation.clear();
			}
		}

		return true;
	}
});
