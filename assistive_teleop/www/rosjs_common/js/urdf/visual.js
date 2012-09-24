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
 * Class to handle visualizing the urdf
 * @class
 * @augments Class
 */

ros.urdf.Visual = Class.extend(
/** @lends ros.urdf.Visual# */
{
	init:function() {	
		// members
		this.origin = null;
		this.geometry = null;
		this.material_name = "";
		this.material = null;
	},

// methods
	clear : function ()
	{
		this.origin = null;
		this.geometry = null;
		this.material_name = "";
		this.material = null;
	}, 

	initXml : function (visual_xml)
	{
		function parseGeometry(geometry_xml)
		{
			var geometry = null;
			var shape = null;

			for (n in geometry_xml.childNodes) {
				var node = geometry_xml.childNodes[n];
				if(node.nodeType == 1) {
					shape = node;
					break;
				}
			}

			if (!shape)
			{
				ros_error("Geometry tag contains no child element.");
			}

			var type_name = shape.nodeName;
			if (type_name == "sphere")
				geometry = new ros.urdf.Sphere();
			else if (type_name == "box")
				geometry = new ros.urdf.Box();
			else if (type_name == "cylinder")
				geometry = new ros.urdf.Cylinder();
			else if (type_name == "mesh")
				geometry = new ros.urdf.Mesh();
			else
			{
				ros_error("Unknown geometry type " + type_name);
				return geometry;
			}

			// clear geom object when fails to initialize
			if (!geometry.initXml(shape)){
				ros_error("Geometry failed to parse");
				geometry = null;
			}

			return geometry;
		}


		this.clear();

		// Origin
		var origins = visual_xml.getElementsByTagName("origin");
		if(origins.length==0)
		{
			ros_debug("Origin tag not present for visual element, using default (Identity)");
			this.origin = new ros.urdf.Pose();
		}
		else 
		{
			var origin = new ros.urdf.Pose();
			if (!origin.initXml(origins[0])) {
				ros_error("Visual has a malformed origin tag");
				return false;
			}
			this.origin = origin;
		}

		// Geometry
		var geoms = visual_xml.getElementsByTagName("geometry");
		if(geoms.length==0)
		{
			ros_debug("Geometry tag not present for visual element.");
		}
		else 
		{
			var geometry = parseGeometry(geoms[0]);    
			if (!geometry) {
				ros_error("Malformed geometry for Visual element.");
				return false;
			}
			this.geometry = geometry;
		}

		// Material
		var materials = visual_xml.getElementsByTagName("material");
		if(materials.length==0)
		{
			ros_debug("visual element has no material tag.");
		}
		else
		{
			// get material name
			var material_xml = materials[0];
			var material = new ros.urdf.Material();

			if (!material_xml.getAttribute("name"))
			{
				ros_error("Visual material must contain a name attribute");
				return false;
			}
			this.material_name = material_xml.getAttribute("name");

			// try to parse material element in place
			if (!material.initXml(material_xml))
			{
			    ros_debug(this.material_name);
			    
				ros_debug("Could not parse material element in Visual block, maybe defined outside.");
			}
			else
			{
				ros_debug("Parsed material element in Visual block.");
				this.material = material;
			}
		}
		return true;
	}
});




