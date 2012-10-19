var ForceDisplay = function () {
    'use strict';
    var forceDisplay = this;
    forceDisplay.wrench = {};
    forceDisplay.magnitude = function () {
        var x = forceDisplay.wrench.force.x
        var y = forceDisplay.wrench.force.y
        var z = forceDisplay.wrench.force.z
        return   Math.sqrt(x*x+y*y+z*z)
    };
    forceDisplay.setState = function (msg) {
        forceDisplay.wrench = msg.wrench;
    };
    forceDisplay.stateSubCBList = [forceDisplay.setState];
    forceDisplay.stateSubCB = function (msg) {
        for (var i = 0; i<forceDisplay.stateSubCBList.length; i += 1) {
            forceDisplay.stateSubCBList[i](msg);
        }
    };
    forceDisplay.stateSub = new window.ros.Topic({
        name: 'wt_force_out_throttle',
        messageType: 'geometry_msgs/WrenchStamped'});
    forceDisplay.stateSub.subscribe(function (msg) {
        forceDisplay.stateSubCB(msg);
    });

    forceDisplay.rezeroPub = new window.ros.Topic({
        name: 'pr2_ft_zeroing/rezero_wrench',
        messageType: 'std_msgs/Bool'});
    forceDisplay.rezeroPub.advertise();
    forceDisplay.rezero = function () {
        var msg = new window.ros.Message({data:true});
        forceDisplay.rezeroPub.publish(msg);
        console.log('Publishing rezero_wrench msg');
    };
    
    forceDisplay.activityThresh = new window.ros.Param({
        name: 'face_adls_manager/activity_force_thresh'});
    forceDisplay.dangerThresh = new window.ros.Param({
        name: 'face_adls_manager/dangerous_force_thresh'});
};

var initFTDisplay = function (divId, options) {
    window.ftDisplay = new ForceDisplay();
    var yellowPercent = options.yellowPercent || 50;
    var maxForce = options.maxForce || 15;
    var height = options.height || '450px';
    var width = options.width || '20px';
    $('#'+divId).append('<table><tr><td>'+
                        '<table id="'+divId+'FTRefTable">'+
                        '<tr id="'+divId+'FTDangerRef">'+
                        '<td id="'+divId+'FTDangerLabel">[FDL]</td>'+
                        '<tr id="'+divId+'FTActivityRef">'+
                        '<td id="'+divId+'FTActivityLabel">[FAL]'+
                        '</td></tr>'+
                        '<tr id="'+divId+'FTNullRef">'+
                        '<td id="'+divId+'FTTextDisplay">'+
                        '</td></tr></table>'+
                        '<td style="height:'+height+'">'+
                        '<div id="'+divId+'FTColorWrapper">'+
                        '<div id="'+divId+'FTColorBar"></div></div>'+
                        '</td></tr>'+
                        '<tr><td colspan=2><button id="'+divId+
                        'FTRezeroButton">Rezero</button>'+
                        '</td></tr></table>');
    $('#'+divId+'FTRefTable').css({'border-style':'solid',
                                   'border-width':'2px',
                                   'border-color':'black',
                                   'border-spacing':'0px',
                                   'height':height});
    $('#'+divId+'FTColorWrapper').css({'width':width,
                                       'height':'100%',
                                       'background-color':'blue'});
    $('#'+divId+'FTColorBar').css({'height':'95%',
                                   'background-color':'white'});
    $('#'+divId+'FTDangerRef').css({'width':width,
                                    'height':'33%',
                                    'background-color':'red'});
    $('#'+divId+'FTActivityRef').css({'width':width,
                                      'height':'50%',
                                      'background-color':'green'});
    $('#'+divId+'FTNullRef').css({'width':width,
                                  'height':'100%',
                                  'background-color':'gray'});
    $('#'+divId+'FTTextDisplay').html('## N');
    $('#'+divId+'FTRezeroButton').css('width','70px');
   
    var updateReadout = function (ws) {
       var pct = (window.ftDisplay.magnitude/maxForce)*100;
       $('#'+divId+'FTTextDisplay').html('<p>'+mag.toFixed(1)+' N </p>')
       var g = "FF";
       var r = "FF";
       if (pct > yellowPercent){
           g = Math.round(255*(1-(pct-yellowPercent)/(100-yellowPercent))).toString(16);
           if (g.length==1){g="0"+g};
       };
       if (pct < yellowPercent){
           r = Math.round(255*(pct/yellowPercent)).toString(16);
           if (r.length==1){r="0"+r};
       };
       var color = "#"+r+g+'00';
       $('#'+divId+'FTColorWrapper').css('background-color', color);
       $('#'+divId+'FTColorBar').css('height', Math.round(100-pct)+'%');
       }; 

    window.ftDisplay.stateSubCBList.push(updateReadout);

    window.ftDisplay.activityThresh.get(function (val) {
        window.ftDisplay.activityThresh.value = val;
        var dangerPct = (
        $('#'+divId+'FTDangerRef').css('height',); 
        $('#'+divId+'FTActivityRef').css('height',); 
        console.log('Param: '+ ftDisplay.activityThresh.name +'\r\n'+
                    ' Value: ' + val.toString());
    });
    

};

          
//    options:{
//        yellow_pct:50,
//        max_force:12
//    },
//    _create: function(){
//        var self = this;
//        this.element.html(ft_viewer_html);
//        node.subscribe('/wt_force_out_throttle',
//                        function(ws){self.element.ft_viewer('refresh',ws)});
//        this.load_params();
//    },
//    load_params: function(){
//            node.rosjs.callService('/rosbridge/get_param',
//                                   '["face_adls_manager"]',
//              function(params){
//                  var dt = params['dangerous_force_thresh'];
//                  var at = params['activity_force_thresh'];
//                  var f_max = $('#ft_view_widget').ft_viewer('option','max_force');
//                  var danger_pct=((f_max-dt)/f_max)*100;
//                  var act_pct = ((f_max-at)/f_max)*100-danger_pct;
//                  $("#ft_ref_danger").height(danger_pct.toString()+'%');
//                  $("#ft_ref_danger_label").text("Danger\r\n >"+params.dangerous_force_thresh.toString()+" N");
//                  $("#ft_ref_act").height(act_pct.toString()+'%');
//                  $("#ft_ref_act_label").text("Activity\r\n  >"+params.activity_force_thresh.toString()+ "N");
//                  window.get_param_free=true;
//                  console.log('Force viewer has released lock on get_params');
//    },

//ft_viewer_html = '<table>
//                    <tr>
//                      <td>
//                            <table border=1 style="height:450px; width:10px">
//                              <tr id="ft_ref_danger" style="height:33%;background-color:red">\
//                                <td id="ft_ref_danger_label">[Danger Force Level]</td>\
//                              </tr>
//                              <tr id="ft_ref_act" style="height:50%;background-color:green">\
//                                <td id="ft_ref_act_label">
//                                [Activity Force Level]
//                                </td>\
//                              </tr>
//                              <tr id="ft_ref_null" style="height:100%;background-color:gray">\
//                                <td>
//                                  <div id="ft_readout"></div>
//                                </td>
//                              </tr>
//                            </table>\
//                      </td>
//                      <td>
//                          <div id="ft_bar_wrapper">\
//                            <div id="ft_bar_value">
//                            </div>
//                          </div>
//                      </td>
//                    </tr>
//                  </table>'
