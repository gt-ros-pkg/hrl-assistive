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

ros.widgets.Widget = Class.extend({
  init: function(domobj) {
    this.domobj = domobj;
    this.id = domobj.getAttribute("id");
    this.jquery = jQuery(this.domobj);
    this.topics = [domobj.getAttribute("topic")];
    this.manager = null;
  },
  
  writeListNode: function(key, value, table, indent) {
    if (key == "header") {
      return;
    }
    tr = document.createElement("tr");
    if (value.constructor == Object) {
      th = document.createElement("tr");
      jQuery(th).css({"font-weight": "bold"});
      td = document.createElement("td");
      td.appendChild(document.createTextNode(indent + key));
      th.appendChild(td);
      table.appendChild(th);    
      
      for (var v in value) {
        this.writeListNode(v, value[v], table, " - ");
      }
    } else {
      td1 = document.createElement("td");
      td1.appendChild(document.createTextNode(indent + key));
      td2 = document.createElement("td");
      td2.appendChild(document.createTextNode(String(value).replace(/,/g, ', ')));
      tr.appendChild(td1);
      tr.appendChild(td2);
      table.appendChild(tr);  
    }
  },
  
  writeMenuTable: function(msg, target) {
    if (!target) { target = ".menu_pane"; }
    var title = this.jquery.attr('title');
    var table = document.createElement("table");
    if (title) {
      th = document.createElement("tr");
      jQuery(th).css({"font-weight": "bold"});
      td = document.createElement("td")
      td.colSpan = "2";
      td.appendChild(document.createTextNode(title));
      th.appendChild(td);
      table.appendChild(th);
    }
    for (var v in msg) {
      this.writeListNode(v, msg[v], table, "");
    }
    this.jquery.find(target)[0].innerHTML = "";
    this.jquery.find(target).html(table);
  },
  
});

jQuery.fn.menu = function(options) {
  jQuery(this).append('<div class="menu_pane"><div class="data">(no data)</div></div>');

  jQuery(this).hover(
    function() {
      jQuery(this).find('.menu_pane').show();
    },
    function() {
      jQuery(this).find('.menu_pane').hide();
    }
  );
};
