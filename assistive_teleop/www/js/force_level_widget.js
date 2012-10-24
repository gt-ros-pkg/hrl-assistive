var ForceDisplay = function (ros) {
    'use strict';
    var forceDisplay = this;
    forceDisplay.ros = ros;
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
    forceDisplay.stateSub = new forceDisplay.ros.Topic({
        name: 'wt_force_out_throttle',
        messageType: 'geometry_msgs/WrenchStamped'});
    forceDisplay.stateSub.subscribe(function (msg) {
        forceDisplay.stateSubCB(msg);
    });

    forceDisplay.rezeroPub = new forceDisplay.ros.Topic({
        name: 'netft_gravity_zeroing/rezero_wrench',
        messageType: 'std_msgs/Bool'});
    forceDisplay.rezeroPub.advertise();
    forceDisplay.rezero = function () {
        var msg = new forceDisplay.ros.Message({data:true});
        forceDisplay.rezeroPub.publish(msg);
        console.log('Publishing rezero_wrench msg');
    };
    
    forceDisplay.activityThresh = new forceDisplay.ros.Param({
        name: 'face_adls_manager/activity_force_thresh'});
    forceDisplay.dangerThresh = new forceDisplay.ros.Param({
        name: 'face_adls_manager/dangerous_force_thresh'});
};

var initFTDisplay = function (divId, options) {
    window.ftDisplay = new ForceDisplay(window.ros);
    var yellowPercent = options.yellowPercent || 50;
    var maxForce = options.maxForce || 15;
    var height = options.height || '450px';
    var width = options.width || '20px';
    
    // Produce basic layout and default options
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
    $('#'+divId+'FTColorBar').css({'height':'0%',
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
    $('#'+divId+'FTDangerLabel').css({'vertical-align':'bottom',
                                      'text-align':'center'});
    $('#'+divId+'FTActivityLabel').css({'vertical-align':'bottom',
                                       'text-align':'center'});
    $('#'+divId+'FTTextDisplay').html('## N');
    $('#'+divId+'FTRezeroButton').css('width','70px');
   
    // Readjust layout based on parameters
    window.ftDisplay.dangerThresh.get(function (val) {
        window.ftDisplay.dangerThresh.value = val;
        console.log('Param: '+ ftDisplay.dangerThresh.name +'\r\n'+
                    ' Value: ' + val.toString());
        window.ftDisplay.activityThresh.get(function (val) {
            window.ftDisplay.activityThresh.value = val;
            var dangerThr = window.ftDisplay.dangerThresh.value;
            $('#'+divId+'FTDangerLabel').html(dangerThr.toString()+' N');
            var dangerPct = 100*(maxForce-dangerThr)/maxForce;
            $('#'+divId+'FTActivityLabel').html(val.toString()+' N');
            var actPct = 100*(maxForce - val)/maxForce - dangerPct;
            $('#'+divId+'FTDangerRef').css('height',dangerPct+'%'); 
            $('#'+divId+'FTActivityRef').css('height',actPct+'%'); 
            console.log('Param: '+ ftDisplay.activityThresh.name +'\r\n'+
                        ' Value: ' + val.toString());
        });
    });

    // Update Display based upon published data
    var updateReadout = function (ws) {
       var mag = window.ftDisplay.magnitude();
       var pct = (mag/maxForce)*100;
       if (pct > 100.0) {pct = 100.0};
       $('#'+divId+'FTTextDisplay').html('<p><strong>'+mag.toFixed(1)+' N </strong></p>')
       var g = "FF";
       var r = "FF";
       if (pct > yellowPercent) {
           g = Math.round(255*(1-(pct-yellowPercent)/(100-yellowPercent))).toString(16);
           if (g.length==1){g="0"+g};
       } else if (pct < yellowPercent) {
           r = Math.round(255*(pct/yellowPercent)).toString(16);
           if (r.length==1){r="0"+r};
       };
       var color = "#"+r+g+'00';
       $('#'+divId+'FTColorWrapper').css('background-color', color);
       $('#'+divId+'FTColorBar').css('height', Math.round(100-pct)+'%');
       }; 
    window.ftDisplay.stateSubCBList.push(updateReadout);

    $('#'+divId+'FTRezeroButton').click(function () {
        window.ftDisplay.rezeroPub.publish({data:true});
        log("Sending command to Re-zero Force/Torque Sensor");
    });
};
