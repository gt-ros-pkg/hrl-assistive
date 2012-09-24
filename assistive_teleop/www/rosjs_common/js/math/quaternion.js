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
 * A class to create and handle a quaternions and the associated math.
 * 
 * @class
 * @augments Class
 */


ros.math.Quaternion = Class.extend(
/** @lends ros.math.Quaternion# */
{

/**
* Initializes quaternion
* 
*/
	
init: function(w, x, y, z) 
  {
    var n = arguments.length;
    if (n == 4) {
      this.w = w;//arguments[0];
      this.x = x;//arguments[1];
      this.y = y;//arguments[2];
      this.z = z;//arguments[3];

      if(this.w == 0 && this.x == 0 && this.y == 0 && this.z == 0 )
        this.w = 1;
      this.normalise();
    }
    else if(n == 3)
    {
      ros_debug('Quaternion Error');
      this.x = w;//arguments[0];
      this.y = x;//arguments[1];
      this.z = y;//arguments[2];
    }
    else {
      this.w = 1;
      this.x = 0;
      this.y = 0;
      this.z = 0;
    }
  },

 /**
  * Returns a new Quaternion with the same w,x,y,z
  * 
*/
  copy : function()
  {
    return new ros.math.Quaternion(this.w, this.x, this.y, this.z);
  },

  /**
   * Creates the Quaternion from a set of axes
   * 
 */   
  fromAxes : function(xaxis, yaxis, zaxis)
  {
    var rot = new Array(9);

    rot[0] = xaxis.x; // 0,0
    rot[1] = yaxis.x; // 0,1
    rot[2] = zaxis.x; // 0,2

    rot[3] = xaxis.y; // 1,0
    rot[4] = yaxis.y; // 1,1
    rot[5] = zaxis.y; // 1,2

    rot[6] = xaxis.z; // 2,0
    rot[7] = yaxis.z; // 2,1
    rot[8] = zaxis.z; // 2,2

    this.fromRotationMatrix(rot);
  },

  /**
   * Creates the Quaternion from a rotation matrix
   * 
 */  
  fromRotationMatrix : function(rot)
  {
    var trace = rot[0] + rot[4] + rot[8]; 
    var root;

    if(trace > 0.0)
    {
      root = Math.sqrt(trace + 1.0);
      this.w = 0.5 * root;
      root = 0.5 / root;

      this.x = (rot[7] - rot[5]) * root;
      this.y = (rot[2] - rot[6]) * root;
      this.z = (rot[3] - rot[1]) * root;
    }
    else 
    {
      var inext = [1,2,0];
      var i = 0;

      if(rot[4] > rot[0])
        i = 1;
      if(rot[8] > rot[i * 3 + i])
        i = 2;
      var j = inext[i];
      var k = inext[j];

      root = Math.sqrt(rot[i * 3 + i] - rot[j * 3 + j] - rot[k * 3 + k] + 1.0);

      var apkQuat = [this.x,this.y, this.z];
      
      apkQuat[i] = 0.5 * root;
      root = 0.5 / root;
      
      this.w = (rot[k* 3 + i] - rot[j * 3 + k]) * root;
      apkQuat[j] = (rot[j * 3 + i] + rot[i * 3 + j]) * root;
      apkQuat[k] = (rot[k * 3 + i] + rot[i * 3 + k]) * root;
      
      this.x = apkQuat[0];
      this.y = apkQuat[1];
      this.z = apkQuat[2];
    }
  },

  
  /**
   * Creates the Quaternion from an axis angle
   * 
 */ 
  fromAngleAxis: function(rfAngle, rkAxis)
  {
    rkAxis.normalise();
    var fHalfAngle = 0.5*rfAngle;
    var fSin = Math.sin(fHalfAngle);

    this.w = Math.cos(fHalfAngle);
    this.x = fSin*rkAxis.x;
    this.y = fSin*rkAxis.y;
    this.z = fSin*rkAxis.z;
    this.normalise();
  },

  /**
   * Normalizes the quaternion to be of length 1.  Returns the previous length
   * 
 */
  normalise: function()
  {
      var len = this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w ;
      var factor = 1.0 / Math.sqrt(len);
      this.multiply(factor);

      return len;
  },
  /**
   * Returns the quaternion that results from the multiplication of the quaternion stored in the current object and another quaternion
   * 
   */
  multiplyQuat : function(quat)
  {
    var w = this.w;
    var x = this.x;
    var y = this.y;
    var z = this.z;

    this.w = w * quat.w - x * quat.x - y * quat.y - z * quat.z;
    this.x = w * quat.x + x * quat.w - y * quat.z + z * quat.y;
    this.y = w * quat.y + x * quat.z + y * quat.w - z * quat.x;
    this.z = w * quat.z - x * quat.y + y * quat.x + z * quat.w;

    this.normalise();

    return this;
  },

  /**
   * Returns the result of multiplying the quaternion stored in this object by a vector
   */
  multiplyVector : function(v)
  {
    var uv = new ros.math.Vector3();
    var uuv = new ros.math.Vector3();
    var qvec = new ros.math.Vector3(this.x, this.y, this.z);

    uv = qvec.crossProduct(v);
    uuv = qvec.crossProduct(uv);
    uv.multiplyScalar(2.0 * this.w);
    uuv.multiplyScalar(2.0);

    var out = new ros.math.Vector3();
    out.add(v);
    out.add(uv);
    out.add(uuv);

    return out;
  },
  
  /**
   * Multiplies Quaternion by a scalar and returns the object
   */
  multiply: function(fScalar)
  {
    
    this.w = fScalar*this.w;
    this.x = fScalar*this.x;
    this.y = fScalar*this.y;
    this.z = fScalar*this.z;

    return this;
  },


  getAxis: function(v)
  {
    var rotation = sglGetQuatRotationM4([this.x, this.y, this.z, this.w]);
    var v4 = sglMulM4V4(rotation, sglV4C(v.x,v.y,v.z,0.0));
    return new ros.math.Vector3(v4[0],v4[1],v4[2]);
  },

  xAxis: function()
  {
    var y = 2.0 * this.y;
    var z = 2.0 * this.z;
    var wy = y * this.w;
    var wz = z * this.w;
    var xy = y * this.x;
    var xz = z * this.x;
    var yy = y * this.y;
    var zz = z * this.z;

    return new ros.math.Vector3(1.0 - (yy + zz) , (xy + wz), (xz - wy));
  },

  yAxis: function()
  {
    var x = 2.0 * this.x;
    var y = 2.0 * this.y;
    var z = 2.0 * this.z;
    var wx = x * this.w;
    var wz = z * this.w;
    var xx = x * this.x;
    var xy = y * this.x;
    var yz = z * this.y;
    var zz = z * this.z;

    return new ros.math.Vector3((xy - wz), (1.0 - (xx + zz)), (yz + wx));
  },

  zAxis: function()
  {
    var x = 2.0 * this.x;
    var y = 2.0 * this.y;
    var z = 2.0 * this.z;
    var wx = x * this.w;
    var wy = y * this.w;
    var xx = x * this.x;
    var xz = z * this.x;
    var yy = y * this.y;
    var yz = z * this.z;

    return new ros.math.Vector3((xz + wy),(yz - wx), (1.0 - (xx + yy)));
  },

  neg_zAxis: function()
  {
    var neg_z = ros.math.Vector3.NEGATIVE_UNIT_Z;
    return this.getAxis(neg_z);
  },

  /**
   * Returns the string of the quaternion
   * 
   * 
   */
  toString : function()
  {
    return ('w: ' + this.w.toFixed(4) + ' x: ' + this.x.toFixed(4) + ' y: ' + this.y.toFixed(4) + ' z: ' + this.z.toFixed(4));
  }
  
});

/**
 * Returns the quaternion that results from the multiplication of two quaternions 
 * 
 * @param q The first quaternion
 * @param quat The second quaternion
 * 
 */
ros.math.Quaternion.MULTIPLY_TWO_QUAT = function(q,quat) 
{
   var t = new ros.math.Quaternion();

   t.w = q.w * quat.w - q.x * quat.x - q.y * quat.y - q.z * quat.z;
   t.x = q.w * quat.x + q.x * quat.w - q.y * quat.z + q.z * quat.y;
   t.y = q.w * quat.y + q.x * quat.z + q.y * quat.w - q.z * quat.x;
   t.z = q.w * quat.z - q.x * quat.y + q.y * quat.x + q.z * quat.w;
   /*  
       this.w = this.w * quat.w - this.x * quat.x - this.y * quat.y - this.z * qua
       this.x = this.w * quat.x + this.x * quat.w + this.y * quat.z - this.z * qua
       this.y = this.w * quat.y - this.x * quat.z + this.y * quat.w + this.z * qua
       this.z = this.w * quat.z + this.x * quat.y - this.y * quat.x + this.z * qua
    */
   t.normalise();
   return t;
};

ros.math.Quaternion.ZERO     = new ros.math.Quaternion( 0.0, 0.0, 0.0, 0.0 );
ros.math.Quaternion.IDENTITY = new ros.math.Quaternion( 1.0, 0.0, 0.0, 0.0 );




