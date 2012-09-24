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

ros.webgl.BitrateGraphWidget = ros.widgets.Widget.extend({
  init: function(domobj) {
    this._super(domobj);
    this.startTime = new Date();
    
//    this.canvas = document.createElement('canvas');
//    jQuery(this.canvas).attr({ id    : "bitrate_graph_canvas",
//                               width : "600",
//                               height: "250"});
//    this.domobj.appendChild(this.canvas);
    this.canvas = document.getElementById("bitrate_graph_canvas");
    this.canvas_id = "bitrate_graph_canvas";
     
    this.graph = null;
    this.topics = [];
    this.topics_bits = [];
    this.topics_bitrate_history = [];
     
  },


  getGraph: function ()
  {
    var graph;
    switch(this.topics.length)
    {
    case 1:
      graph = new RGraph.Line(this.canvas_id, 
                              this.topics_bitrate_history[0]);
      //graph.Set('chart.colors', ['rgb(169, 222, 244)']);
      //graph.Set('chart.fillstyle', ['#daf1fa']);
      
      break;
    case 2:
      graph = new RGraph.Line(this.canvas_id, 
                              this.topics_bitrate_history[0], 
                              this.topics_bitrate_history[1]);
      //graph.Set('chart.colors', ['rgb(169, 222, 244)', 'red']);
      //graph.Set('chart.fillstyle', ['#daf1fa', '#faa']);
      break;
    case 3:
      graph = new RGraph.Line(this.canvas_id, 
                              this.topics_bitrate_history[0], 
                              this.topics_bitrate_history[1], 
                              this.topics_bitrate_history[2]);
     // graph.Set('chart.colors', ['rgb(169, 222, 244)', 'red', 'green']);
      break;
    case 4:
      graph = new RGraph.Line(this.canvas_id, 
                              this.topics_bitrate_history[0], 
                              this.topics_bitrate_history[1], 
                              this.topics_bitrate_history[2], 
                              this.topics_bitrate_history[3]);
      //graph.Set('chart.colors', ['rgb(169, 222, 244)', 'red', 'green', 'yellow']);
      break;
    default:
      graph = new RGraph.Line(this.canvas_id, 
          this.topics_bitrate_history[0], 
          this.topics_bitrate_history[1], 
          this.topics_bitrate_history[2], 
          this.topics_bitrate_history[3]);
      //graph.Set('chart.colors', ['rgb(169, 222, 244)', 'red', 'green', 'yellow']);
      break;
    }
     
    //graph = new RGraph.Line(this.canvas_id, this.d1, this.d2);
    graph.Set('chart.gutter', 25);
    graph.Set('chart.background.barcolor1', 'white');
    graph.Set('chart.background.barcolor2', 'white');
    graph.Set('chart.title.xaxis', 'Time');
    graph.Set('chart.title.yaxis', 'kbit/s');
    graph.Set('chart.filled', true);
    //graph.Set('chart.fillstyle', ['#daf1fa', '#faa']);
    //graph.Set('chart.colors', ['rgb(169, 222, 244)', 'red']);
    graph.Set('chart.linewidth', 3);
    //graph.Set('chart.ymax', 250);
    graph.Set('chart.xticks', 25);
    graph.Set('chart.key', this.topics);

    return graph;
  },
  
  drawGraph: function ()
  {
    if(this.topics.length > 0) {
      var curTime = new Date();
      var startTime = this.startTime;
      var diff = curTime.getTime() - startTime.getTime();
      
      // calculate bitrates
      for (i in this.topics) {
        var topic = this.topics[i];
        var bitrate = (this.topics_bits[i] / diff);
        this.topics_bitrate_history[i] = this.topics_bitrate_history[i];
        this.topics_bitrate_history[i].push(bitrate);
        // Get rid of the first values of the arrays
        while (this.topics_bitrate_history[i].length > 100) {
          this.topics_bitrate_history[i] = RGraph.array_shift(this.topics_bitrate_history[i]);
        }
      }
      
      // Clear the canvas and redraw the graph
      RGraph.Clear(this.canvas);
      this.graph = this.getGraph();
      this.graph.Draw(); 
    
      // reset bit counters
      for (i in this.topics) {
        var topic = this.topics[i];
        this.topics_bits[i] = 0;
      }
      this.startTime = curTime;
   }

    var that = this;
    setTimeout(function(e){that.drawGraph();},2000);
  },
   
  findTopicID: function (topic) 
  {
    for( var i = 0; i < this.topics.length; i++ ) {
      if( this.topics[ i ] == topic ) {
        return i;
      }
    }
    return -1;
  },
  
  /**
   * Updates this BitrateWidget.
   * @return {boolean} whether this sample counter actually updated this tick.
   */
  update: function(msg) {
      var topic = "/tf_changes";
      var call = ''; 
      try {
        eval('call = ' + msg.data);
      } catch(err) {
        return;
      }
      topic = call.receiver;
      var id = this.findTopicID(topic);
      if(id == -1) {
        return;
//        id = this.topics.length;
//        this.topics.push(topic);
//        this.topics_bits[id] = 0;
//        this.topics_bitrate_history[id] = [];
//        // Pre-pad the arrays with 100 null values
//        for (var i=0; i< 100; ++i) {
//          this.topics_bitrate_history[id].push(null);
//        }
      }
      this.topics_bits[id] += msg.data.length * 8.0;
      return false;
  },

  setTopics: function(topics) {
    this.topics = topics;
    this.topics_bits = [];
    this.topics_bitrate_history = [];
    for (i in this.topics) {
      var topic = this.topics[i];
      this.topics_bits[i] = 0;
      this.topics_bitrate_history[i] = [];
      // Pre-pad the arrays with 100 null values
      for (var j=0; j< 100; ++j) {
        this.topics_bitrate_history[i].push(null);
      }
    }
  },

});  