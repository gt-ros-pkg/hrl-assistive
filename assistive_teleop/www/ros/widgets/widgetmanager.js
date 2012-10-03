/*******************************************************************************
 * 
 * Software License Agreement (BSD License)
 * 
 * Copyright (c) 2010, Robert Bosch LLC. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met: *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer. * Redistributions in binary
 * form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials provided
 * with the distribution. * Neither the name of the Robert Bosch nor the names
 * of its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 ******************************************************************************/

ros.widgets.WidgetManager = Class.extend({
  init: function(node) {
    this.node = node;
    this.widgets = [];
    this.widget_hash = [];
    this.topicListeners = new ros.Map();
    this.subscribers = new ros.Map();
    this.checkWidgets();
  },
  
  registerWidget: function(widget, type) {
    widget.manager = this;

    this.widgets.push(widget);
    
    if (type) {
        this.widget_hash[type] = widget;
    }

    for(var i=0; i<widget.topics.length; i++) {
        var topic = widget.topics[i];
        this.registerListener(widget, topic);
    }
  },
  
  registerListener: function(listener, topic) {
    var listeners = this.topicListeners.find(topic);
    if(listeners == null) {
      listeners = [];
      this.topicListeners.insert(topic, listeners);
      var that = this;
      this.node.subscribe(topic,function(msg){
        that.receiveMessage(topic,msg);
      });
    }
    listeners.push(listener);
  },
  
  receiveMessage: function(topic, msg) {
    var listeners = this.topicListeners.find(topic);
    if(listeners) {
      for(var j=0; j<listeners.length; j++) {
        try {
          listeners[j].receive(topic, msg);
        } catch (e) {
          ros_debug("Error while processing topic: " + e);
        }
      }
    }
  },
  
  setupWidgets:  function() {
    var widget = null;
    var allHTMLTags=document.getElementsByTagName("*");
    for (i=0; i<allHTMLTags.length; i++) {
      var domobj = allHTMLTags[i];
      var objtype = domobj.getAttribute("objtype");
      if(objtype) {
          var clss = window[objtype];
          if(clss) {
            widget = new clss(domobj);
            this.registerWidget(widget, domobj.getAttribute("objtype"));
          }
      }
    }
    var urlprefix = this.urlprefix;
  },
  
  checkWidgets: function() {
    for(var i=0; i<this.widgets.length; i++) {
      var widget = this.widgets[i];
      if (widget) {
        try {
          widget.check();
        } catch (exception) {
          //ros_debug("error checking widget: " + exception);
        }
      }
    }
    var that = this;
    setTimeout(function() {
      that.checkWidgets();
    }, 5000);
  },
  
  
});