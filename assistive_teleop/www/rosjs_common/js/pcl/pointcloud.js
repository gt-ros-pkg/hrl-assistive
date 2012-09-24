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
 * Class to handle and store pointcloud messages
 * @class
 * @augments Class
 */ 
ros.pcl.PointCloud=Class.extend(
/** @lends ros.pcl.PointCloud# */
{
/**
 * Initializes Pointcloud members
 */
	init : function() {
		this.header = new ros.roslib.Header();
		this.points = new Array();
		this.width = 0;
		this.height = 0;
		this.is_dense = false;
	},

/**
 * Updates point cloud based upon new incoming message
 */
	updateFromMessage : function (pointcloud_msg)
	{ 
		this.clear();
		this.header.updateFromMessage(pointcloud_msg.header);
		for ( var p in pointcloud_msg.points) {
			var point_msg = pointcloud_msg.points[p];
			var point = new ros.math.Vector3();
			point.x = point_msg.x;
			point.y = point_msg.y;
			point.z = point_msg.z;
			this.points.push(point);
		}
		this.width = pointcloud_msg.width;
		this.height = pointcloud_msg.height;
		this.is_dense = pointcloud_msg.is_dense;
	},

	clear : function() 
	{
		this.points = new Array();
		this.width = 0;
		this.height = 0;
		this.is_dense = false;
	}
});
