var BodyRegistration = function () {
    'use strict';
    var bodyReg = this;
    bodyReg.head_reg_cb = function () {
        $('#img_act_select').val('seedReg');
        window.mjpeg.setCamera('head_registration/confirmation');
        window.log("Click on your check to begin head registration");
    };
}

var initBodyRegistration = function (tabDivId) {
    window.body_reg = new BodyRegistration();
    divRef = "#"+tabDivId;
    $(divRef).append('<table><tr><td id="'+tabDivId+'_R0C0"></td><td id="'+tabDivId+'_R0C1"></td></tr></table>');
    $(divRef+'_R0C0').append('<button class="centered" id="reg_head"> Register Head </button>');
    $("#reg_head").button();
    $(divRef+'_R0C0').click(window.body_reg.head_reg_cb);

}

