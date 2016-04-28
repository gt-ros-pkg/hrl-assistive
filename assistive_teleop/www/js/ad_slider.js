var Pr2AD = function (ros) {
    'use strict';
    var ad = this;
    ad.ros = ros;
    ad.state = 0.0;
    ad.ros.getMsgDetails('std_msgs/Float64');
    ad.statePub = new ad.ros.Topic({
        name: 'manipulation_task/ad_sensitivity_request',
        messageType: 'std_msgs/Float64'
    });

    ad.statePub.advertise();

    ad.stateSub = new ad.ros.Topic({
        name: 'manipulation_task/ad_sensitivity_state',
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

var initAdSlider = function (orientation) {
    $('#ad_slider').slider({
        min: 0.0,
        max: 1.0,
        step: 0.05,
        orientation: orientation
    });
    var adStateDisplay = function (msg) {
        if ($('#ad_slider > .ui-slider-handle').hasClass('ui-state-active') !== true) {
            $('#ad_slider').show().slider('option', 'value', assistive_teleop.ad.state);
        }
    };
    assistive_teleop.ad.stateCBList.push(adStateDisplay);
    $('#ad_slider').unbind("slidestop").bind("slidestop", function (event, ui) {
        assistive_teleop.ad.setSensitivity($('#ad_slider').slider("value"));
    });

    
    document.getElementById('ad_sense_max').addEventListener('click', function (e) {
        newVal1 = $('#ad_slider').slider("value")+0.05;
        if (newVal1 > 1) {
            newVal1 = 1;
        }
 
        assistive_teleop.ad.setSensitivity(newVal1);

//        assistive_teleop.ad.setSensitivity($('#ad_slider').slider("value")+5);
    });
    document.getElementById('ad_sense_min').addEventListener('click', function (e) {

        newVal2 = $('#ad_slider').slider("value")-0.05;
        if (newVal2 < 0) {
            newVal2 = 0;
        }
        assistive_teleop.ad.setSensitivity(newVal2);


        //assistive_teleop.ad.setSensitivity($('#ad_slider').slider("value")-5);
    });


}

var initAdGUI = function () {
    assistive_teleop.ad = new Pr2AD(assistive_teleop.ros);
}


