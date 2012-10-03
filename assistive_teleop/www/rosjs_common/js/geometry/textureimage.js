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
 * A class to create a textured image.
 * 
 * @class
 * @augments Class
 */

ros.geometry.TextureImage = Class.extend(
/** @lends ros.geometry.TextureImage# */
{

/**
 * Initializes the textured image, can be called with the three optional parameters or without
 * 
 * @param pixarray Optional parameter of the rgba data of the image
 * @param img_width Optional parameter to specify image width
 * @param img_height Optional parameter to specify the image height
 * 
 */	
  init : function(pixarray, img_width, img_height)
  {
    this.width = 1;
    this.height = 1;
    this.pixels = [];

    if(arguments.length == 3)
    {
      this.fromImage(arguments[0],arguments[1],arguments[2]);
    }

  },


  // assume pixarray contains rgba
  /**
   * Creates a texture (with a size that is a power of two) from an image
   * 
   * @param pixarray   rgba data of the image
   * @param img_width  image width
   * @param img_height image height
   * 
   */
  fromImage : function(pixarray,img_width,img_height)
  {
    var width = 1;
    var height = 1;

    while(width < img_width)
      width *= 2;

    while(height < img_height)
      height *= 2;


    this.resize(pixarray,width,height,img_width,img_height);
  },

  /**
   * Resizes the image 
   * 
   * @param pix_array   rgba data of the image
   * @param w  desired image width
   * @param h  desired image height
   * @param oldw current image width
   * @param oldh current image height
   * 
   */
  
  resize : function(pix_array,w,h,oldw,oldh)
  {
    var x,y;
    var fx,fy;
    var dx = (oldw - 1)/(w - 1);
    var dy = (oldh - 1)/(h - 1);
    var index;
    var array = new Array();

    for(fy = 0.0, y = 0; y < h; ++y, fy+=dy)
      for(fx = 0.0, x = 0; x < w; ++x, fx+=dx)
      {
        index = (y * w + x) * 4;
/*
        if(index < pix_array.length) {
          array[index] = pix_array[index];
          array[index+1] = pix_array[index+1];
          array[index+2] = pix_array[index+2];
          array[index+3] = pix_array[index+3];
        }
        else {
          array[index] = 0;
          array[index+1] = 0;
          array[index+2] = 0;
          array[index+3] = 0;
        }
*/
        
        array[index] = this.interpolate(pix_array,fx,fy,oldw,oldh,0);
        array[index+1] = this.interpolate(pix_array,fx,fy,oldw,oldh,1);
        array[index+2] = this.interpolate(pix_array,fx,fy,oldw,oldh,2);
        array[index+3] = 0;
      }

    this.width = w;
    this.height = h;
    this.pixels = array;
  },

  /**
   * Stretches the image to fit the texture 
   * 
   * 
   */
  
  interpolate : function(pix,fx,fy,oldw,oldh,n)
  {
    var truncR = this.clamp(Math.floor(fx),0,oldw - 1);
    var truncR1 = this.clamp(truncR + 1, 0, oldw -1);
    var fractR = oldw - Math.floor(truncR);
    var truncC = this.clamp(Math.floor(fy),0, oldh-1);
    var truncC1 = this.clamp(truncC +1,0, oldh -1);
    var fractC = oldh - Math.floor(truncC);

    var index = oldw * truncC + truncR;
    var syx   = pix[(oldw * truncC  + truncR) * 4 + n];
    var syx1  = pix[(oldw * truncC1 + truncR) * 4 + n];
    var sy1x  = pix[(oldw * truncC  + truncR1) * 4 + n];
    var sy1x1 = pix[(oldw * truncC1 + truncR1) * 4 + n];

    var tmp1 = syx + (syx1 - syx) * fractC;
    var tmp2 = sy1x + (sy1x1 - sy1x) * fractC;
    
    return (tmp1 + (tmp2 - tmp1) * fractR);
  },

  
  toUint8Array : function()
  {
    return new Uint8Array(this.pixels);
  },

/**
 * Forces the input to be between two values.  
 * 
 * 
 */ 
  
  clamp : function(input, min, max)
  {
    if(input < min)
      return min;

    return input < max?input:max;
  },
});
