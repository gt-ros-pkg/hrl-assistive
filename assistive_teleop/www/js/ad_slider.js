var Pr2AD = function (ros, task) {
    'use strict';
    var ad = this;
    ad.ros = ros;
    ad.state = 0.0;
    ad.ros.getMsgDetails('std_msgs/Float64');

    ad.statePub = new ad.ros.Topic({
        name: task.concat('/manipulation_task/ad_sensitivity_request'),
        messageType: 'std_msgs/Float64'
    });
    ad.statePub.advertise();

    ad.stateSub = new ad.ros.Topic({
        name: task.concat('/manipulation_task/ad_sensitivity_state'),
        messageType: 'std_msgs/Float64'
    });
    ad.setState = function (msg) {
        //TODO: Is this right or wrong???
        ad.state = msg.data;
    } ;   

    ad.stateCBList = [ad.setState];
    ad.stateCB = function(msg) {
        for (var i=0; i<ad.stateCBList.length; i++){
            ad.stateCBList[i](msg);
        };
    };
    ad.stateSub.subscribe(ad.stateCB);

    ad.setSensitivity = function (z) {
        
        var cmd = new ad.ros.Message({
            data: z
        });
        //assistive_teleop.log(z);
        
        ad.statePub.publish(cmd);
    };

};

var initAdSlider = function (orientation, task) {
    var str1 = "#ad_";
    var str2 = "ad_";
    $(str1.concat(task, '_slider')).slider({
        min: 0.0,
        max: 1.0,
        step: 0.05,
        orientation: orientation
    });
    if (task == "feeding"){
        var ad = assistive_teleop.feeding_ad;
    }else{
        var ad = assistive_teleop.scooping_ad;
    }
    var adStateDisplay = function (msg) {
        if ($(str1.concat(task, '_slider > .ui-slider-handle')).hasClass('ui-state-active') !== true) {
            $(str1.concat(task, '_slider')).show().slider('option', 'value', ad.state);
        }
    };
    ad.stateCBList.push(adStateDisplay);
    $(str1.concat(task, '_slider')).unbind("slidestop").bind("slidestop", function (event, ui) {
        ad.setSensitivity($(str1.concat(task, '_slider')).slider("value"));
    });

    
    document.getElementById(str2.concat(task,'_sense_max')).addEventListener('click', function (e) {
        newVal1 = $(str1.concat(task, '_slider')).slider("value")+0.05;
        if (newVal1 > 1) {
            newVal1 = 1;
        }
 
        ad.setSensitivity(newVal1);
        //        assistive_teleop.ad.setSensitivity($('#ad_feeding_slider').slider("value")+5);
    });
    document.getElementById(str2.concat(task,'_sense_min')).addEventListener('click', function (e) {

        newVal2 = $(str1.concat(task, '_slider')).slider("value")-0.05;
        if (newVal2 < 0) {
            newVal2 = 0;
        }
        ad.setSensitivity(newVal2);
        //assistive_teleop.ad.setSensitivity($('#ad_feeding_slider').slider("value")-5);
    });


}

var initAdGUI = function (task) {  
    if (task == "scooping"){
        assistive_teleop.scooping_ad = new Pr2AD(assistive_teleop.ros, task);
    }else{
        assistive_teleop.feeding_ad = new Pr2AD(assistive_teleop.ros, task);
    }
    initAdSlider('horizontal', task);
    //initAdSlider('horizontal', 'feeding');

}


