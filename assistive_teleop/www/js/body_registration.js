var BodyRegistration = function (ros) {
    'use strict';
    var bodyReg = this;
    bodyReg.ros = ros;
    bodyReg.head_reg_cb = function () {
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
        });
    };




}

var initBodyRegistration = function (tabDivId) {
    window.body_reg = new BodyRegistration(window.ros);
    divRef = "#"+tabDivId;
    $(divRef).append('<table><tr><td id="'+tabDivId+'_R0C0"></td><td id="'+tabDivId+'_R0C1"></td></tr></table>');
    $(divRef+'_R0C0').append('<button class="centered" id="reg_head"> Register Head </button>');
    $("#reg_head").button();
    $(divRef+'_R0C0').click(window.body_reg.head_reg_cb);

}

