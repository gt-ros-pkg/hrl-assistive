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
 * Handles urdf model and initialization from XML file
 * @class
 * @augments Class
 */
ros.urdf.Model = Class.extend(
/** @lends ros.urdf.Model# */
{
	init: function() {

		// members
		this.name_ = "";
		this.links_ = new ros.Map();
		this.joints_ = new ros.Map();
		this.materials_ = new ros.Map();
	},

	clear : function() {
		this.name_ = '';
		this.links_.clear();
		this.joints_.clear();
		this.materials_.clear();
	},

	initFile : function(src, callback) {
		var xhr = new XMLHttpRequest();
		var self = this;
		xhr.onreadystatechange = function() {
			// Status of 0 handles files coming off the local disk
			if (xhr.readyState == 4 && (xhr.status == 200 || xhr.status == 0)) {
				ros.runSoon(function() {
					var xml = xhr.responseXML;
					console.log(xhr);
					console.log(xml)
					xml.getElementById = function(id) {
						console.log(id);
						return xpathGetElementById(xml, id);
					};
					self.initXml(xml);
        
					if (callback) {
						callback(self);
					}
				});
			}
		};

		console.log(src);
		xhr.open("GET", src, true);
		console.log(xhr);
		xhr.overrideMimeType("text/xml");
		xhr.setRequestHeader("Content-Type", "text/xml");
		xhr.send(null);
		
	},

	initXml : function(xml_doc) {
		function nsResolver(prefix) {
			var ns = {
					'c' : 'http://www.collada.org/2005/11/COLLADASchema'
			};
			return ns[prefix] || null;
		}

		function getNode(xpathexpr, ctxNode) {
			if (ctxNode == null)
				ctxNode = xml_doc;
			return xml_doc.evaluate(xpathexpr, ctxNode, null,
					XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
		}

		var robot_xml = getNode('//robot');
		if (!robot_xml) {
			ros_error("Could not find the 'robot' element in the xml file");
			return false;
		}

		this.clear();

		ros_debug("Parsing robot xml");

		// Get robot name
		var name = robot_xml.getAttribute("name");
		if (!name) {
			ros_error("No name given for the robot.");
			return false;
		}
		this.name_ = name;

//		var nodes = robot_xml.getElementsByTagName("*");
//		for ( var c in nodes) {
//		var node = nodes[c];
//		var name = node.getAttribute("name");
//		ros_error(name);
//		}

		// Get all Material elements
		for (n in robot_xml.childNodes) {
			var node = robot_xml.childNodes[n];
			if(node.tagName != "material") continue;
			var material_xml = node;
			var material = new ros.urdf.Material();

			if (material.initXml(material_xml)) {
				if (this.getMaterial(material.name)) {
					ros_error("material " + material.name + " is not unique.");
					return false;
				} else {
					this.materials_.insert(material.name, material);
					ros_debug("successfully added a new material " + material.name);
				}
			} else {
				ros_error("material xml is not initialized correctly");
				return false;
			}
		}

		// Get all Link elements
		for (n in robot_xml.childNodes) {
		    var node = robot_xml.childNodes[n];
			if(node.tagName != "link") continue;
			var link_xml = node;
			var link = new ros.urdf.Link();

			if (link.initXml(link_xml)) {
				if (this.getLink(link.name)) {
					ros_error("link " + link.name + " is not unique.");
					return false;
				} else {
					// set link visual material
					ros_debug("setting link " + link.name + " material");
					if (link.visual) {
						if (link.visual.material_name.length > 0) {
							if (this.getMaterial(link.visual.material_name)) {
								ros_debug("setting link " + link.name + " material to " + link.visual.material_name);
								link.visual.material = this.getMaterial(link.visual.material_name);
							} else {
								if (link.visual.material) {
									ros_debug("link " + link.name + " material " + link.visual.material_name + " defined in Visual.");
									this.links_.insert(link.visual.material.name,link.visual.material);
								} else {
									ros_error("link " + link.name + " material " + link.visual.material_name + " undefined.");
									return false;
								}
							}
						}
					}

					this.links_.insert(link.name, link);
					ros_debug("successfully added a new link " + link.name);
				}
			} else {
				ros_error("link xml is not initialized correctly");
				return false;
			}
		}
		if (this.links_.empty()) {
			ros_error("No link elements found in urdf file");
			return false;
		}

		// Get all Joint elements
		for (n in robot_xml.childNodes) {
			var node = robot_xml.childNodes[n];
			if(node.tagName != "joint") continue;
			var joint_xml = node;
			var joint = new ros.urdf.Joint();

			if (joint.initXml(joint_xml)) {
				if (this.getJoint(joint.name)) {
					ros_error("joint " + joint.name + " is not unique.");
					return false;
				} else {
					this.joints_.insert(joint.name, joint);
					ros_debug("successfully added a new joint " + joint.name);
				}
			} else {
				ros_error("joint xml is not initialized correctly");
				return false;
			}
		}
		return true;
	},

	getMaterial : function(name) {
		return this.materials_.find(name);
	},

	getLink : function(name) {
		return  this.links_.find(name);
	},

	getLinks : function() {
		return this.links_;
	},

	getJoint : function(name) {
		return this.joints_.find(name);
	}

});

