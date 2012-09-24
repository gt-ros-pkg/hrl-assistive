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
 * A class to create and handle 3D Vectors.
 * 
 * @class
 * @augments Class
 */
ros.math.Vector3 = Class.extend(
/** @lends ros.math.Vector2# */
{
  init: function() 
  {
    var n = arguments.length;
    if (n == 3) {
      this.x = arguments[0];
      this.y = arguments[1];
      this.z = arguments[2];
    }
    else {
      this.x = 0;
      this.y = 0;
      this.z = 0;
    }
  },
  
  initString: function (str)
  {
    this.clear();
    
    var xyz = ros.parseFloatListString(str);
  
    if (xyz.length != 3) {
      ros_error("Vector contains " + xyz.length + " elements instead of 3 elements"); 
      return false;
    }
  
    this.x = xyz[0];
    this.y = xyz[1];
    this.z = xyz[2];
  
    return true;
  },

  toArray : function()
  {
    return [this.x, this.y, this.z];
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
    this.z = this.z - (-other.z);

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
    this.z = this.z - other.z;
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
    return Math.sqrt( this.x * this.x + this.y * this.y + this.z * this.z );
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
    var fLength = Math.sqrt( this.x * this.x + 
                             this.y * this.y + 
                             this.z * this.z );
  
    // Will also work for zero-sized vectors, but will change nothing
    if ( fLength > 1e-08 )
    {
        var fInvLength = 1.0 / fLength;
        this.x *= fInvLength;
        this.y *= fInvLength;
        this.z *= fInvLength;
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
    return this.x * vec.x + this.y * vec.y + this.z * vec.z;
  },
  
  /** 
   * Calculates the cross-product of 2 vectors, i.e. the vector that
   * lies perpendicular to them both.
   * @remarks
   *     The cross-product is normally used to calculate the normal
   *     vector of a plane, by calculating the cross-product of 2
   *     non-equivalent vectors which lie on the plane (e.g. 2 edges
   *     of a triangle).
   * @param
   *     vec Vector which, together with this one, will be used to
   *     calculate the cross-product.
   * @returns
   *     A vector which is the result of the cross-product. This
   *     vector will <b>NOT</b> be normalised, to maximise efficiency
   *     - call Vector3::normalise on the result if you wish this to
   *     be done. As for which side the resultant vector will be on, the
   *     returned vector will be on the side from which the arc from 'this'
   *     to rkVector is anticlockwise, e.g. UNIT_Y.crossProduct(UNIT_Z)
   *     = UNIT_X, whilst UNIT_Z.crossProduct(UNIT_Y) = -UNIT_X.
   *     This is because OGRE uses a right-handed coordinate system.
   * @par
   *     For a clearer explanation, look a the left and the bottom edges
   *     of your monitor's screen. Assume that the first vector is the
   *    left edge and the second vector is the bottom edge, both of
   *     them starting from the lower-left corner of the screen. The
   *     resulting vector is going to be perpendicular to both of them
   *     and will go <i>inside</i> the screen, towards the cathode tube
   *     (assuming you're using a CRT monitor, of course).
   */
  crossProduct: function(rkVector)
  {
    return new ros.math.Vector3(
        this.y * rkVector.z - this.z * rkVector.y,
        this.z * rkVector.x - this.x * rkVector.z,
        this.x * rkVector.y - this.y * rkVector.x);
  },

  /** 
   * Returns true if this vector is zero length. 
   */
  isZeroLength: function()
  {
      var sqlen = (this.x * this.x) + (this.y * this.y) + (this.z * this.z);
      return (sqlen < (1e-06 * 1e-06));
  },
  
  /** 
   * Gets the shortest arc quaternion to rotate this vector to the destination vector.
   * 
   * @remarks
   * If you call this with a dest vector that is close to the inverse
   * of this vector, we will rotate 180 degrees around the 'fallbackAxis'
   * (if specified, or a generated axis if not) since in this case ANY axis of rotation is valid.
  */
  getRotationTo: function (dest, fallbackAxis)
  {
    fallbackAxis = typeof(fallbackAxis) != 'undefined' ? fallbackAxis : ros.math.Vector3.ZERO;
    
    // Based on Stan Melax's article in Game Programming Gems
    var q = new ros.math.Quaternion();
    // Copy, since cannot modify local
    var v0 = this;
    var v1 = dest;
    v0.normalise();
    v1.normalise();
    
    var d = v0.dotProduct(v1);
    // If dot == 1, vectors are the same
    if (d >= 1.0)
    {
        return ros.math.Quaternion.IDENTITY;
    }
    if (d < (1e-6 - 1.0))
    {
      if (fallbackAxis != ros.math.Vector3.ZERO)
      {
        // rotate 180 degrees about the fallback axis
        q.fromAngleAxis(Math.PI, fallbackAxis);
      }
      else
      {
        // Generate an axis
        var axis = ros.math.Vector3.UNIT_X.crossProduct(this);
        if (axis.isZeroLength()) // pick another if colinear
          axis = ros.math.Vector3.UNIT_Y.crossProduct(this);
        axis.normalise();
        q.fromAngleAxis(Math.PI, axis);
      }
    }
    else
    {
      var s = Math.sqrt( (1+d)*2 );
      var invs = 1 / s;
    
      var c = v0.crossProduct(v1);
    
      q.x = c.x * invs;
      q.y = c.y * invs;
      q.z = c.z * invs;
      q.w = s * 0.5;
      q.normalise();
    }
    return q;
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
    this.z = scalar * this.z;

    return this;
  },

  toString : function()
  {
    return ('x : ' + this.x.toFixed(4) + ' y: ' + this.y.toFixed(4) + ' z: ' + this.z.toFixed(4));
  },

  copy : function()
  {
    return new ros.math.Vector3(this.x,this.y, this.z);
  }
  
});

ros.math.Vector3.ZERO = new ros.math.Vector3( 0, 0, 0 );
ros.math.Vector3.UNIT_X = new ros.math.Vector3( 1, 0, 0 );
ros.math.Vector3.UNIT_Y = new ros.math.Vector3( 0, 1, 0 );
ros.math.Vector3.UNIT_Z = new ros.math.Vector3( 0, 0, 1 );
ros.math.Vector3.NEGATIVE_UNIT_X = new ros.math.Vector3( -1,  0,  0 );
ros.math.Vector3.NEGATIVE_UNIT_Y = new ros.math.Vector3(  0, -1,  0 );
ros.math.Vector3.NEGATIVE_UNIT_Z = new ros.math.Vector3(  0,  0, -1 );
ros.math.Vector3.UNIT_SCALE = new ros.math.Vector3(1, 1, 1);


