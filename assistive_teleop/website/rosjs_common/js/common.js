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
 * Reference to the global context.  In most cases this will be 'window'.
 */
ros.global = this;

/**
 * Path for included scripts.
 * @type {string}
 */
ros.basePath = 'rosjs_common/js/';

/**
 * Tries to detect the base path of the base.js script that
 * bootstraps the ros libraries.
 * @private
 */
ros.findBasePath_ = function() {
  var doc = ros.global.document;
  if (typeof doc == 'undefined') {
    return;
  }
  if (ros.global.BASE_PATH) {
    ros.basePath = ros.global.BASE_PATH;
    return;
  } else {
    // HACK to hide compiler warnings :(
    ros.global.BASE_PATH = null;
  }
  var scripts = doc.getElementsByTagName('script');
  for (var script, i = 0; script = scripts[i]; i++) {
    var src = script.src;
    var l = src.length;
    var s = 'ros/ros.js';
    var sl = s.length;
    if (src.substr(l - sl) == s) {
      ros.basePath = src.substr(0, l - sl) + 'ros/';
      return;
    }
  }
};

/**
 * Writes a script tag for the given ros source file name
 * to the document.  (Must be called at execution time.)
 * @param {string} src The full path to the source file.
 * @private
 */
ros.writeScriptTag_ = function(src) {
  var doc = ros.global.document;
  if (typeof doc != 'undefined') {
    doc.write('<script type="text/javascript" src="' +
              src + '"></' + 'script>');
  }
};

/**
 * Filters any "ros." prefix from the given type name.
 * @param {string} type_name The type name to filter.
 * @return {string} Filtered type name.
 * @private
 */
ros.filterTypeName_ = function(type_name) {
  if (type_name.length >= 4 && type_name.substr(0, 4) == 'ros.') {
    type_name = type_name.substr(4);
  }
  return type_name;
};

/**
 * Includes the file indicated by the rule by adding a script tag.
 * @param {string} rule Rule to include, in the form ros.package.part.
 */
ros.include = function(rule) {
  var parts = rule.split('.');
  var path = parts[parts.length - 1] + '.js';
  ros.writeScriptTag_(ros.basePath + path);
};


/**
 * Makes one class inherit from another.  Adds the member variables superClass
 * and superClassName to the prototype of the sub class.
 * @param {string} subClass Class that wants to inherit.
 * @param {string} superClass Class to inherit from.
 */
ros.inherit = function(subClassName, superClassName) {
  var superClass = ros.global.ros[superClassName];
  var subClass = ros.global.ros[subClassName];

  if (!superClass)
    throw ('Invalid superclass: ' + superClassName);
  if (!subClass)
    throw ('Invalid subclass: ' + subClassName);

  subClass.prototype = new superClass;
  subClass.prototype.superClassName = superClassName;
  subClass.prototype.superClass = superClass;
  subClass.prototype.className = subClassName;
};


/**
 * Utility function to remove an object from an array.
 * @param {!Array} array The array.
 * @param {Object} object The thing to be removed.
 */
ros.removeFromArray = function(array, object) {
  var i = array.indexOf(object);
  if (i >= 0) {
    array.splice(i, 1);
  }
}

/**
 * If an ros function has not been implemented in javascript yet, it should
 * call this function to throw an error because it's better than doing
 * nothing.
 */
ros.notImplemented = function() {
  debugger;
  throw 'Not implemented.';
};

/**
 * Parses a list of floats from a string
 */
ros.parseFloatListString = function(s) {
    if (s == "")
      return [];

    // first trim
    var ts = ros.trim(s);
    
    // this is horrible
    var ss = ts.split(/\s+/);
    var res = Array(ss.length);
    for (var i = 0, j = 0; i < ss.length; i++) {
      if (ss[i].length == 0)
        continue;
      res[j++] = parseFloat(ss[i]);
    }
    return res;
}

/**
 * Runs a function f 
 */
ros.runSoon = function (f) {
  setTimeout(f, 0);
}

/**
 * Trims a string
 */
ros.trim = function(str) {
  str = str.replace(/^\s+/, '');
  for (var i = str.length - 1; i >= 0; i--) {
    if (/\S/.test(str.charAt(i))) {
      str = str.substring(0, i + 1);
      break;
    }
  }
  return str;
}

/**
 * Sleep function
 */
ros.sleep = function(ms)
{
  var dt = new Date();
  dt.setTime(dt.getTime() + ms);
  while (new Date().getTime() < dt.getTime());
}

if (!("console" in window)) {
    window.console = {
      log : function(s) {
        var l = document.getElementById('log');
        if (l) {
          l.innerHTML = l.innerHTML + "<span>" + s.toString() + "</span><br>";
        }
      }
    };
  }

ros_error = function(string)
{
  console.log(string);
}

ros_debug = function(string)
{
  console.log(string);
}

ros_info = function(string)
{
  console.log(string);
}

ros.nop = function()
{  
}

ros.json = function(obj) 
{
  return JSON.stringify(obj);
}


/**
 * Prints a value the console or log or wherever it thinks is appropriate
 * for debugging.
 * @param {string} string String to print.
 */
ros.dump = function(string) {
  ros_info(string);
};

// First find the path to the directory where all ros-webgl sources live.
ros.findBasePath_();

//include all files at once
ros.include('class');
ros.include('system/time');
ros.include('system/map');
ros.include('system/tree');
ros.include('core/core');
ros.include('math/math');
ros.include('geometry/geometry');
ros.include('roslib/roslib');
ros.include('pcl/pcl');
ros.include('urdf/urdf');
ros.include('tf/tf');
ros.include('actionlib/actionlib');
ros.include('widgets/widgets');

