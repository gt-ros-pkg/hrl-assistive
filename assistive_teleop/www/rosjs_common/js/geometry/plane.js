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
 * A class to create a plane.
 * 
 * @class
 * @augments Class
 */
ros.geometry.Plane = Class.extend(
/** @lends ros.geometry.Plane# */
{
	
/**
 *  Initializes the Plane.  If two parameters are not provided the origin and normal are set to be empty vectors
 *   
 *  @param origin origin of plane 
 *  @param normal normal vector of the plane
 *   	
*/	
  init: function(origin, normal)
  {
    var n = arguments.length;

    if( n == 2)
    {
      this.origin = origin;
      this.normal = normal;
    }
    else {
      this.origin = new ros.math.Vector3();
      this.normal = new ros.math.Vector3();
    }
  },


  // returns distance. point return will be implemented later.
  /**
   *  Returns distance
   *   
   *  @param origin origin of plane 
   *  @param normal normal vector of the plane
   *   	
  */	
  intersectRay: function(ray)
  {
    var tmp = this.normal.dotProduct(ray.getDirection());

    if(tmp == 0)
    {
      ros_error("Cannot intersect plane with a parallel line");
      return null;
    }

    var po = this.normal.dotProduct(this.origin);
    var ro = this.normal.dotProduct(ray.getOrigin());
    var dist = (po - ro) / tmp;

    return dist;
  },

});

