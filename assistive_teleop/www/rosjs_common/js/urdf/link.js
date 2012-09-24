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
 * Class to help store urdf link structure information for the urdf tree 
 * @class
 * @augments Class
 **/
ros.urdf.Link = Class.extend(
/** @lends ros.urdf.Link# */
{
	init: function() {
		// members
		this.name = "";
		this.inertial = null;
		this.visual = null;
		this.collision = null;
		this.parent_link = null;
		this.parent_joint = null;
		this.child_joints = [];
		this.child_links = [];
	},

// methods
	clear : function() {
		this.name = "";	
		this.inertial = null;
		this.visual = null;
		this.collision = null;
		this.parent_joint = null;
		this.child_joints = [];
		this.child_links = [];
	},

	getParent : function() {
		return this.parent_link;
	},

	setParent : function(parent) {
		this.parent_link = parent;
	},

	getParentJoint : function() {
		return this.parent_joint;
	},

	setParentJoint : function(parent) {
		this.parent_joint = parent;
	},

	addChild : function(child) {
		this.child_links.push(child);
	},

	addChildJoint : function(child) {
		this.child_joints.push(child);
	},

	initXml : function(xml) {

		this.clear();

		var name = xml.getAttribute("name");
		if (!name)
		{
			ros_error("No name given for the link.");
			return false;
		}
		this.name = name;

		// Inertial (optional)
//		TiXmlElement *i = config->FirstChildElement("inertial");
//		if (i)
//		{
//		inertial.reset(new Inertial);
//		if (!inertial->initXml(i))
//		{
//		ROS_ERROR("Could not parse inertial element for Link '%s'", this->name.c_str());
//		return false;
//		}
//		}

		// Visual (optional)
		var visuals = xml.getElementsByTagName("visual");
		if(visuals.length>0)
		{
			var visual_xml = visuals[0];
			var visual = new ros.urdf.Visual();
			if (!visual.initXml(visual_xml))
			{
				ros_error("Could not parse visual element for Link " + this.name);
				return false;
			}
			this.visual = visual;
		}

		// Collision (optional)
//		TiXmlElement *col = xml.FirstChildElement("collision");
//		if (col)
//		{
//		collision.reset(new Collision);
//		if (!collision.initXml(col))
//		{
//		console.log("Could not parse collision element for Link '%s'", this.name.c_str());
//		return false;
//		}
//		}

		return true;
	},
});