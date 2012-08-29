ft_viewer_html = '<table><tr><td>\
            <table border=1 style="height:450px; width:10px">\
        <tr id="ft_ref_danger" style="height:33%;background-color:red">\
          <td id="ft_ref_danger_label">[Danger Force Level]</td>\
        </tr><tr id="ft_ref_act" style="height:50%;background-color:green">\
          <td id="ft_ref_act_label">[Activity Force Level]</td>\
        </tr><tr id="ft_ref_null" style="height:100%;background-color:gray">\
         <td><div id="ft_readout"></div></td></tr></table>\
        </td><td><div id="ft_bar_wrapper">\
          <div id="ft_bar_value"></div></div></td></tr></table>'
          
$(function(){
$.widget("rfh.ft_viewer",{
    options:{
        yellow_pct:50,
        max_force:12
    },
    _create: function(){
        var self = this;
        this.element.html(ft_viewer_html);
        node.subscribe('/wt_force_out_throttle',
                        function(ws){self.element.ft_viewer('refresh',ws)});
        this.load_params();
    },
    load_params: function(){
        if (window.get_param_free){
            window.get_param_free = false;
            console.log("Force viewer has locked get_param");
            node.rosjs.callService('/rosbridge/get_param',
                                   '["face_adls_manager"]',
              function(params){
                  var dt = params['dangerous_force_thresh'];
                  var at = params['activity_force_thresh'];
                  var f_max = $('#ft_view_widget').ft_viewer('option','max_force');
                  var danger_pct=((f_max-dt)/f_max)*100;
                  var act_pct = ((f_max-at)/f_max)*100-danger_pct;
                  $("#ft_ref_danger").height(danger_pct.toString()+'%');
                  $("#ft_ref_danger_label").text("Danger\r\n >"+params.dangerous_force_thresh.toString()+" N");
                  $("#ft_ref_act").height(act_pct.toString()+'%');
                  $("#ft_ref_act_label").text("Activity\r\n  >"+params.activity_force_thresh.toString()+ "N");
                  window.get_param_free=true;
                  console.log('Force viewer has released lock on get_params');
          })} else {
              console.log("Ft viewer widget waiting for rosparam service");
              setTimeout('$("#'+this.element.attr("id").toString()+'").ft_viewer("load_params");',500);
          };

    },
    _destroy: function(){
        this.element.html('');
    },
    refresh: function(ws){
       var mag = Math.sqrt(Math.pow(ws.wrench.force.x,2) + 
                       Math.pow(ws.wrench.force.y,2) + 
                       Math.pow(ws.wrench.force.z,2))
       var pct = 100*Math.min(mag, this.options.max_force)/this.options.max_force;
       $('#ft_readout').html('<p>'+mag.toFixed(1)+' N </p>')
       var g = "FF";
       var r = "FF";
       if (pct > this.options.yellow_pct){
           g = Math.round((255*(1-(pct-this.options.yellow_pct)/(100-this.options.yellow_pct)))).toString(16);
           if (g.length==1){g="0"+g};
       };
       if (pct < this.options.yellow_pct){
           r = (Math.round(255*(pct/this.options.yellow_pct))).toString(16);
           if (r.length==1){ r="0"+r};
       };
       var color = "#"+r+g+'00';
       $('#ft_bar_wrapper').css('background-color', color);
       $('#ft_bar_value').css('height', Math.round(100-pct).toString()+'%');
        
    },
});
});

