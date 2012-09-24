/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Robert Bosch LLC.
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
 * A class to create a ray.
 * 
 * @class
 * @augments Class
 */
ros.geometry.Ray = Class.extend(
/** @lends ros.geometry.Ray# */
{
	
/**
 * Initializes Vectors for the origin and direction
 * Can be called with either classes with properties x, y, z : var new_ray= new ros.geometry.Ray(points1, points2) the origin is set to be at points1
 *  or with the the points specified individually:var new_ray= new ros.geometry.Ray(x1, y1, z1, x2, y2, z2) the origin is set to be x2, y2, z2
 *  
 */	
  init: function()
  {
    var n = arguments.length;  
    if (n == 2) {
      this.origin = new ros.math.Vector3(arguments[0].x, arguments[0].y, arguments[0].z);
      this.direction = new ros.math.Vector3(arguments[1].x, arguments[1].y, arguments[1].z);
    }
    else if(n == 6)
    {
      this.origin = new ros.math.Vector3(arguments[3],arguments[4],arguments[5]);
      this.direction = new ros.math.Vector3(arguments[0] - arguments[3],arguments[1] - arguments[4],arguments[2] - arguments[5]);
    }
  },
 
  /**
   * Returns the direction of the Ray
   * 
   */	
  getDirection : function()
  {
    return this.direction;
  },
/**
 * 	Returns the origin of the array
 * 
 */
  getOrigin : function()
  {
    return this.origin;
  },

  getPoint : function(t)
  {
    var o = new ros.math.Vector3(this.origin.x, this.origin.y, this.origin.z);
    var d = new ros.math.Vector3(this.direction.x,this.direction.y,this.direction.z);

    return o.add(d.multiplyScalar(t));    
  }

});
