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
 * A class to create and handle 2D Vectors.
 * 
 * @class
 * @augments Class
 */
ros.math.Vector2 = Class.extend(

{
  init: function(x, y) 
  {
    var n = arguments.length;
    if (n == 2) {
      this.x = x;
      this.y = y;
    }
    else {
      this.x = 0;
      this.y = 0;
    }
  },
  
  initString: function (str)
  {
    this.clear();
    
    var xy = ros.parseFloatListString(str);
  
    if (xy.length != 3) {
      ros_error("Vector contains " + xyz.length + " elements instead of 3 elements"); 
      return false;
    }
  
    this.x = xy[0];
    this.y = xy[1];
  
    return true;
  },

  /** 
   * Adds another vector to this vector
   * @param
   *     other Vector to add to this vector
   * @returns
   *     This vector with the provided vector added
   */
  add : function(other)
  {
    this.x = this.x - (-other.x);
    this.y = this.y - (-other.y);

    return this;
  },
  
  /** 
   * Subtracts another vector from this vector
   * @param
   *     other Vector to subtract from this vector
   * @returns
   *     This vector with the provided vector subtracted
   */
  subtract: function(other)
  {
    this.x = this.x - other.x;
    this.y = this.y - other.y;
    return this;
  },
  
  /** 
   * Returns the length (magnitude) of the vector.
   * @warning
   *     This operation requires a square root and is expensive in
   *     terms of CPU operations. If you don't need to know the exact
   *     length (e.g. for just comparing lengths) use squaredLength()
   *     instead.
   */
  length: function()
  {
    return Math.sqrt( this.x * this.x + this.y * this.y);
  },
  
  /**
   * Normalises the vector.
   *  
   * @remarks
   *     This method normalises the vector such that it's
   *     length / magnitude is 1. The result is called a unit vector.
   * @note
   *     This function will not crash for zero-sized vectors, but there
   *     will be no changes made to their components.
   * @returns The previous length of the vector.
   *  
   */
  normalise: function ()
  {
    var fLength = Math.sqrt( this.x * this.x + this.y * this.y); 
  
    // Will also work for zero-sized vectors, but will change nothing
    if ( fLength > 1e-08 )
    {
        var fInvLength = 1.0 / fLength;
        this.x *= fInvLength;
        this.y *= fInvLength;
    }
  
    return fLength;
  },

  /** 
   * Calculates the dot (scalar) product of this vector with another.
   * @remarks
   *     The dot product can be used to calculate the angle between 2
   *     vectors. If both are unit vectors, the dot product is the
   *     cosine of the angle; otherwise the dot product must be
   *     divided by the product of the lengths of both vectors to get
   *     the cosine of the angle. This result can further be used to
   *     calculate the distance of a point from a plane.
   * @param
   *     vec Vector with which to calculate the dot product (together
   *     with this one).
   * @returns
   *     A float representing the dot product value.
   */
  dotProduct: function(vec)
  {
    return this.x * vec.x + this.y * vec.y;
  },
  
  /** 
   * Checks if the vector has a zero length
   * 
   * @returns a bool indicating if the vector has zero length. 
   */
  isZeroLength: function()
  {
      var sqlen = (this.x * this.x) + (this.y * this.y);
      return (sqlen < (1e-06 * 1e-06));
  },
  
  /**
   * Multiplies the vector by a scalar
   * @param scalar Scalar with which to multiply this vector
   * @returns This vector scaled
   */
  multiplyScalar : function(scalar)
  { 
    this.x = scalar * this.x;
    this.y = scalar * this.y;

    return this;
  },
  
});

ros.math.Vector2.ZERO = new ros.math.Vector2( 0, 0);
ros.math.Vector2.UNIT_X = new ros.math.Vector2( 1, 0);
ros.math.Vector2.UNIT_Y = new ros.math.Vector2( 0, 1);
ros.math.Vector2.NEGATIVE_UNIT_X = new ros.math.Vector2( -1,  0);
ros.math.Vector2.NEGATIVE_UNIT_Y = new ros.math.Vector2(  0, -1);
ros.math.Vector2.UNIT_SCALE = new ros.math.Vector2(1, 1);


