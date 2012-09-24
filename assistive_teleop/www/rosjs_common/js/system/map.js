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
 * Class to create and handle a HashMap
 * @class
 * @augments Class
 */
ros.Map = Class.extend(
/** @lends ros.Map# */
{
	init : function()
	{
	    // members
	    this.keyArray = new Array(); // Keys
	    this.valArray = new Array(); // Values  
	},
	
/**
 * Checks if the Map contains any items
 * @returns true if empty else false
 */	
	empty : function () 
	{
	  return(this.size() == 0);
	},

/**
 * Inserts an key value pair into the Map
 * 
 */
	insert :function (key,val) 
	{
	    var elementIndex = this.getIndex( key );
	    
	    if( elementIndex == (-1) )
	    {
	        this.keyArray.push( key );
	        this.valArray.push( val );
	    }
	    else
	    {
	        this.valArray[ elementIndex ] = val;
	    }
	},
/**
 * Returns the object associated with the provided key returns null if there is no mapping for the key
 * 
 */
	find : function (key) 
	{
	    var result = null;
	    var elementIndex = this.getIndex( key );

	    if( elementIndex != (-1) )
	    {   
	        result = this.valArray[ elementIndex ];
	    }  
	    
	    return result;
	},

/**
 * Removes the mapping for the specified key from this map if present.
 */
	remove: function (key) 
	{
	    var result = null;
	    var elementIndex = this.getIndex( key );

	    if( elementIndex != (-1) )
	    {
	        this.keyArray.splice(elementIndex,1);
	        this.valArray.splice(elementIndex,1);
	    }  
	    
	    return ;
	},

/**
 * Returns the number of key-value mappings in this map
 */
	size : function () 
	{
	    return (this.keyArray.length);  
	},
/**
 * Removes all key-value mappings from this map
 */
	clear :function () 
	{
	    for( var i = 0; i < this.keyArray.length; i++ )
	    {
	        this.keyArray.pop(); this.valArray.pop();   
	    }
	},
/**
 * Returns the array  of the keys contained in this map.
 */
	keySet : function () 
	{
	    return (this.keyArray);
	},
	
/**
 * Returns the array  of the values contained in this map.
 */
	valSet : function () 
	{
	    return (this.valArray);   
	},


/**
 * Returns a string of all key-value mappings
 */
		
	showme : function () 
	{
	    var result = "";
	    
	    for( var i = 0; i < this.keyArray.length; i++ )
	    {
	        result += "Key: " + this.keyArray[ i ] + "\tValues: " + this.valArray[ i ] + "\n";
	    }
	    return result;
	},

/**
 * Returns the index for the key in the key array
 */
	getIndex : function (key) 
	{
	    var result = (-1);

	    for( var i = 0; i < this.keyArray.length; i++ )
	    {
	        if( this.keyArray[ i ] == key )
	        {
	            result = i;
	            break;
	        }
	    }
	    return result;
	},

	
/**
 * Removes the key at a specified index
 */	
	removeAt : function (index)
	{
	  var part1 = this.slice( 0, index);
	  var part2 = this.slice( index+1 );

	  return( part1.concat( part2 ) );
	},

	
});






