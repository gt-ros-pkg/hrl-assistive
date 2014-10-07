var initGripper = function (orientation) {
    $('#r_gripper_slider').slider({
        min: 0.0,
        max: 0.09,
        step: 0.001,
        orientation:orientation 
    });
    var rGripperStateDisplay = function (msg) {
        if ($('#r_gripper_slider > .ui-slider-handle').hasClass('ui-state-active') !== true) {
            $('#r_gripper_slider').show().slider('option', 'value', RFH.gripper[1].state);
        }
    };
    RFH.gripper[1].stateCBList.push(rGripperStateDisplay);
    $('#r_gripper_slider').unbind("slidestop").bind("slidestop", function (event, ui) {
        RFH.gripper[1].setPosition($('#r_gripper_slider').slider("value"));
        if ($('#r_gripper_slider').slider("value") > RFH.gripper[1].state) {
            log('Opening Right Gripper');
        } else {
            log('Closing Right Gripper');
        }
    });
    document.getElementById('r_gripper_open').addEventListener('click', function (e) {
        RFH.gripper[1].open();
        log('Opening Right Gripper');
    });
    document.getElementById('r_gripper_close').addEventListener('click', function (e) {
        RFH.gripper[1].close();
        log('Closing Right Gripper');
    });

    $('#l_gripper_slider').slider({
        min: 0.0,
        max: 0.09,
        step: 0.001,
        orientation: orientation 
    });
    var lGripperStateDisplay = function (msg) {
        if ($('#l_gripper_slider > .ui-slider-handle').hasClass('ui-state-active') !== true) {
            $('#l_gripper_slider').show().slider('option', 'value', RFH.gripper[0].state);
        }
    };
    RFH.gripper[0].stateCBList.push(lGripperStateDisplay);
    $('#l_gripper_slider').unbind("slidestop").bind("slidestop", function (event, ui) {
        RFH.gripper[0].setPosition($('#l_gripper_slider').slider("value"));
        if ($('#l_gripper_slider').slider("value") > RFH.gripper[0].state) {
            log('Opening Left Gripper');
            RFH.skinUtil.addTaxelArray('/pr2_pps_left_sensor/taxels/forces');
            RFH.skinUtil.addTaxelArray('/pr2_pps_right_sensor/taxels/forces');
        } else {
            log('Closing Left Gripper');
            RFH.skinUtil.removeTaxelArray('/pr2_pps_left_sensor/taxels/forces');
            RFH.skinUtil.removeTaxelArray('/pr2_pps_right_sensor/taxels/forces');
        }
    });
    document.getElementById('l_gripper_open').addEventListener('click', function (e) {
        RFH.gripper[0].open();
        RFH.skinUtil.addTaxelArray('/pr2_pps_left_sensor/taxels/forces');
        RFH.skinUtil.addTaxelArray('/pr2_pps_right_sensor/taxels/forces');
        log('Opening Left Gripper');
    });
    document.getElementById('l_gripper_close').addEventListener('click', function (e) {
        RFH.gripper[0].close();
        RFH.skinUtil.removeTaxelArray('/pr2_pps_left_sensor/taxels/forces');
        RFH.skinUtil.removeTaxelArray('/pr2_pps_right_sensor/taxels/forces');
        log('Closing Left Gripper');
    });
}

var teleopHead = function () {
    $('#bpd_default_rot, #cart_frame_select, #cart_frame_select_label, #cart_controller').hide();
    log('Controlling Head');
    $('#scale_slider').show().slider("option", "value", RFH.head.sliderScale);
    $('#scale_slider').unbind("slidestop").bind("slidestop", function (event, ui) {
        RFH.head.sliderScale = $('#scale_slider').slider("value");
    });

    $('#bpd_default').find(':button').unbind('.rfh').text('');
    $('#b9, #b7', '#bpd_default').hide();
    $('#b8, #b6, #b5, #b4, #b2', '#bpd_default').bind('click.rfh', function (e) {
        clearInterval(RFH.head.pubInterval);
    });
    $('#bpd_default #b8').show().bind('click.rfh', function (e) { //head up 
        RFH.head.delPosition(0.0, -RFH.head.sliderScale);
    });
    $('#bpd_default #b6').show().bind('click.rfh', function (e) { //head right
        RFH.head.delPosition(-RFH.head.sliderScale, 0.0);
    });
    $('#bpd_default #b5').show().text("_|_").bind('click.rfh', function (e) { //center head to (0,0)
        RFH.head.pointHead(0.8, 0.0, -0.25, '/base_footprint');
    });
    $('#bpd_default #b4').show().bind('click.rfh', function (e) { //head left
        RFH.head.delPosition(RFH.head.sliderScale, 0.0);
    });
    $('#bpd_default #b3').show().removeClass('arrow_rot_x_pos').text("Track Right Hand").bind('click.rfh', function (e) {
        clearInterval(RFH.head.pubInterval);
        RFH.head.pubInterval = setInterval(function () {
            RFH.head.pointHead(0, 0, 0, 'r_gripper_tool_frame');
        }, 100);
    });
    $('#bpd_default #b2').show().bind('click.rfh', function (e) { //head down
        RFH.head.delPosition(0.0, RFH.head.sliderScale);
    });
    $('#bpd_default #b1').show().removeClass('arrow_rot_x_neg').text("Track Left Hand").bind('click.rfh', function (e) {
        clearInterval(RFH.head.pubInterval);
        RFH.head.pubInterval = setInterval(function () {
            RFH.head.pointHead(0, 0, 0, 'l_gripper_tool_frame');
        }, 100);
    });
}

var teleopBase = function () {
    log("Controlling Base");
    $('#bpd_default_rot, #cart_frame_select, #cart_frame_select_label, #cart_controller').hide();
    $('#scale_slider').show().slider("option", "value", RFH.base.scaleSlider);
    $('#scale_slider').unbind("slidestop").bind("slidestop", function (event, ui) {
        RFH.base.scaleSlider = $('#scale_slider').slider("value");
    });

    $('#bpd_default').find(':button').unbind('.rfh').text('');
    $('#b9, #b7, #b5', '#bpd_default').hide();

    $('#bpd_default #b8').show().bind('mousedown.rfh', function (e) {
        RFH.base.drive("#bpd_default #" + e.target.id, 0.2 * RFH.base.scaleSlider, 0, 0);
    });
    $('#bpd_default #b6').show().bind('mousedown.rfh', function (e) {
        RFH.base.drive("#bpd_default #" + e.target.id, 0, -0.2 * RFH.base.scaleSlider, 0);
    });
    $('#bpd_default #b4').show().bind('mousedown.rfh', function (e) {
        RFH.base.drive("#bpd_default #" + e.target.id, 0, 0.2 * RFH.base.scaleSlider, 0);
    });
    $('#bpd_default #b3').show().addClass('arrow_rot_x_pos').bind('mousedown.rfh', function (e) {
        RFH.base.drive("#bpd_default #" + e.target.id, 0, 0, -0.6 * RFH.base.scaleSlider);
    });
    $('#bpd_default #b2').show().bind('mousedown.rfh', function (e) {
        RFH.base.drive("#bpd_default #" + e.target.id, -0.2 * RFH.base.scaleSlider, 0, 0);
    });
    $('#bpd_default #b1').show().addClass('arrow_rot_x_neg').bind('mousedown.rfh', function (e) {
        RFH.base.drive("#bpd_default #" + e.target.id, 0, 0, 0.6 * RFH.base.scaleSlider);
    });
}

var initTorsoSlider = function (orientation) {
    $('#torso_slider').slider({
        min: 0.0,
        max: 0.3,
        step: 0.01,
        orientation: orientation
    });
    var torsoStateDisplay = function (msg) {
        if ($('#torso_slider > .ui-slider-handle').hasClass('ui-state-active') !== true) {
            $('#torso_slider').show().slider('option', 'value', msg.actual.positions[0]);
        }
    };
    RFH.torso.stateCBList.push(torsoStateDisplay);
    $('#torso_slider').unbind("slidestop").bind("slidestop", function (event, ui) {
        RFH.torso.setPosition($('#torso_slider').slider("value"));
    });
    document.getElementById('torso_max').addEventListener('click', function (e) {
        RFH.torso.setPosition(0.3);
    });
    document.getElementById('torso_min').addEventListener('click', function (e) {
        RFH.torso.setPosition(0.0);
    });
}

var initPr2 = function () {
    RFH.head = new Pr2Head(RFH.ros); 
    RFH.head.sliderScale = 0.5;
    RFH.base = new Pr2Base(RFH.ros);    
    RFH.base.scaleSlider = 0.5;
    RFH.gripper = [new Pr2Gripper('left', RFH.ros),
                      new Pr2Gripper('right', RFH.ros)];
    RFH.torso = new Pr2Torso(RFH.ros);
    RFH.wtLog = new RFH.ros.Topic({
        name: 'wt_log_out',
        messageType:'std_msgs/String'});
    RFH.wtLog.subscribe(function (msg) {
        RFH.log(msg.data);});
}
