var BodyRegistration = function (ros) {
    'use strict';
    var bodyReg = this;
    bodyReg.ros = ros;
    bodyReg.headRegCB = function () {
        $('#img_act_select').val('seedReg');
        window.mjpeg.setCamera('head_registration/confirmation');
        window.log("Click on your check to begin head registration");
    };

    bodyReg.headRegServiceClient = new bodyReg.ros.Service({
        name:'/initialize_registration',
        serviceType:'hrl_head_registration/InitializeRegistration'});

    bodyReg.registerHead = function (u,v) {
        bodyReg.headRegServiceClient.callService({u:u,v:v}, function (resp) {
            console.log('Initialize Head Registration Service Returned.');
            $('#img_act_select').val('looking');
            $("#confirm_reg").show();
        });
    };

    bodyReg.faceSideParam = new ros.Param({
        name: "face_side"
        });

    bodyReg.setSideParam = function () {
        var side = $("input[name=face_side]:checked").val();
        bodyReg.faceSideParam.set(side);
        console.log("Setting Param: " + bodyReg.faceSideParam.name + " = " + side);
    };

    bodyReg.regConfirmServiceClient = new bodyReg.ros.Service({
        name:"/confirm_registration",
        serviceType:"hrl_head_registration/ConfirmRegistration"
        });

    bodyReg.confirmRegistration = function () {
        bodyReg.regConfirmServiceClient.callService({}, function (resp) {
        if (resp) {
            console.log("Head Registration Confirmed.");
        }});
    }
}

var initBodyRegistration = function (tabDivId) {
    window.bodyReg = new BodyRegistration(window.ros);
    divRef = "#"+tabDivId;
    $(divRef).append('<table id="' + tabDivId +
                     '_T0"><tr><td id="' + tabDivId +
                     '_R0C0"></td><td id="' + tabDivId +
                     '_R0C1"></td></tr></table>');
    $(divRef+'_T0').append('<tr><td id="' + tabDivId + '_R1C0"></td></tr>')
    $(divRef+'_R0C0').append('<button class="centered" id="reg_head"> Register Head </button>');
    $(divRef+'_R0C1').append('<button class="centered" id="confirm_reg"> Confirm </button>');
    $(divRef+'_R1C0').append('<form id="face_side_form">' + 
                             '<input id="face_radio_left" name="face_side" type="radio" value="l">'+
                             '<label for="face_radio_left"> Left </label>' +
                             '<input id="face_radio_right" name="face_side" type="radio" value="r" checked="checked">' +
                             '<label for="face_radio_right"> Right </label>' +
                             '</form>');
    $("#reg_head").button();
    $("#confirm_reg").button().hide();
    $("#face_side_form").buttonset();
    $("#face_side_form").change(window.bodyReg.setSideParam);
    $(divRef+'_R0C0').click(window.bodyReg.headRegCB);
    $(divRef+'_R0C1').click(window.bodyReg.confirmRegistration);
    window.bodyReg.setSideParam();
}

