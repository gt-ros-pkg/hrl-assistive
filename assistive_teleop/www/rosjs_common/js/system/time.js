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
 * Class to handle time messages
 * @class
 * @augments Class
 */
ros.Time = Class.extend(
		/** @lends ros.Time# */
		{
		init: function(){
		// members
		this.secs = 0;
		this.nsecs = 0;
		},

		updateFromMessage : function (msg)
		{
			this.secs = msg.secs;
			this.nsecs = msg.nsecs;
		},

		/**
		 * returns current time in seconds
		 */
		now : function()
		{
			var milliceconds = (new Date).getTime();
			return this.fromMilliseconds(milliceconds);
		},

		add : function(time)
		{
			this.secs += time.secs;
			this.nsecs += time.nsecs;
			this.normalize();
			return this;
		},

		subtract : function(time)
		{
			this.secs -= time.secs;
			this.nsecs -= time.nsecs;
			this.normalize();
			return this;
		},

		fromSeconds : function(seconds)
		{
			this.secs = Math.floor(seconds);
			this.nsecs = Math.round((seconds-this.secs) * 1000000000);
			return this;
		},

		fromMilliseconds : function(milliseconds)
		{
			this.secs = Math.floor(milliseconds / 1000);
			this.nsecs = Math.round((milliseconds / 1000 - this.secs) * 1000000000);
			return this;
		},

		toMilliseconds : function()
		{
			return this.secs*1000+this.nsecs/1000000;
		},

		toSec : function()
		{
			return this.secs+this.nsecs/1000000000;
		},

		normalize : function()
		{
			var nsec_part = this.nsecs;
			var sec_part = this.secs;

			while (nsec_part >= 1000000000)
			{
				nsec_part -= 1000000000;
				++sec_part;
			}
			while (nsec_part < 0)
			{
				nsec_part += 1000000000;
				--sec_part;
			}

			this.secs = Math.round(sec_part);
			this.nsecs = Math.round(nsec_part);
		},

		});

