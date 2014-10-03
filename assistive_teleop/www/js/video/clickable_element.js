var ClickableElement = function(elementID) {
    'use strict';
    this.elementID = elementID;
    this.onClickCBList = [];
    this.getClickPixel = function (e) {
        //FIXME: This should be made cross-browser. 
        //Clickable Element doesn't seem to work in firefox.
        var posx = 0;
        var posy = 0;
        if (!e) var e = window.event;
        if (e.pageX || e.pageY) 	{
            posx = e.pageX;
            posy = e.pageY;
        }
        else if (e.clientX || e.clientY) 	{
            posx = e.clientX + document.body.scrollLeft
                + document.documentElement.scrollLeft;
            posy = e.clientY + document.body.scrollTop
                + document.documentElement.scrollTop;
        }
        var offsetLeft = 0;
        var offsetTop = 0;
        var element = document.getElementById(e.target.id);
        while (element && !isNaN(element.offsetLeft)
                && !isNaN(element.offsetTop)) {
            offsetLeft += element.offsetLeft;
            offsetTop += element.offsetTop;
            element = element.offsetParent;
        }
        posx -= offsetLeft;
        posy -= offsetTop;
        console.log('Element '+e.target.id+' clicked at (x,y) = ('+posx.toString() +','+ posy.toString()+')');
        return [posx, posy]
    };

    this.convertDisplayToCameraPixel = function (pixel, display, camera) {
        var px = pixel[0];
        var py = pixel[1];
        var dw = display.activeParams['width'];
        var dh = display.activeParams['height'];
        var cw = display.cameraData[camera].width;
        var ch = display.cameraData[camera].height;
        return [Math.round(px/dw*cw), Math.round(py/dh*ch)];
    };

    this.getClickPixelWRTCamera= function (e) {
        var pixel = this.getClickPixel(e);
        var camera = $('#'+assistive_teleop.mjpeg.selectBoxId+" :selected").val();
        return this.convertDisplayToCameraPixel(pixel, assistive_teleop.mjpeg, camera);
    };

    this.onClickCB = function (e) {
        var camera = $('#cameraSelect :selected').text();
        if (assistive_teleop.mjpeg.cameraData[camera].clickable) {
            var pixel = this.getClickPixelWRTCamera(e)
            for (var i = 0; i < this.onClickCBList.length; i += 1) {
                this.onClickCBList[i](pixel);
            }
        }
    }
}

var PoseSender = function (ros, topic) {
    'use strict';
    var poseSender = this;
    poseSender.ros = ros;
    poseSender.posePub = new poseSender.ros.Topic({
        name: topic,
        messageType: 'geometry_msgs/PoseStamped'})
    poseSender.posePub.advertise();

    poseSender.sendPose = function (poseStamped) {
        var msg = new poseSender.ros.Message(poseStamped);
        poseSender.posePub.publish(msg);
    }
}

var LookatIk = function (ros, goalTopic) {
    'use strict';
    this.ros = ros;

    this.targetPublisher = new this.ros.Topic({
        name: goalTopic,
        messageType: 'geometry_msgs/PointStamped'});
    this.targetPublisher.advertise();

    this.publishTarget = function (pointStamped) {
        var pt = new this.ros.Message(pointStamped);
        this.targetPublisher.publish(pt);
    }

    this.callPoseStamped = function (poseStamped) {
        var msg = this.ros.composeMsg('geometry_msgs/PointStamped');
        msg.header = poseStamped.header
        msg.point = poseStamped.pose.position
        this.publishTarget(msg);
    }
}

var Pixel23DClient = function (ros) {
    'use strict';
    var p23D = this;
    p23D.ros = ros;
    p23D.serviceClient =  new p23D.ros.Service({
                                        name: '/pixel_2_3d',
                                        serviceType: 'Pixel23d'});
    p23D.call = function (u, v, cb) {
        var req = new p23D.ros.ServiceRequest({'pixel_u':u, 'pixel_v':v});
        p23D.serviceClient.callService(req, cb);
    }
}


var initClickableActions = function () {
    assistive_teleop.rPoseSender = new PoseSender(assistive_teleop.ros, 'wt_r_click_pose');
    assistive_teleop.lPoseSender = new PoseSender(assistive_teleop.ros, 'wt_l_click_pose');
    assistive_teleop.rCamPointSender = new LookatIk(assistive_teleop.ros, '/rightarm_camera/lookat_ik/goal')
    //assistive_teleop.poseSender = new PoseSender(assistive_teleop.ros);
    assistive_teleop.clickableCanvas = new ClickableElement(assistive_teleop.mjpeg.imageId);
    assistive_teleop.p23DClient = new Pixel23DClient(assistive_teleop.ros);

    $('#'+assistive_teleop.mjpeg.imageId).on('click.rfh', assistive_teleop.clickableCanvas.onClickCB.bind(assistive_teleop.clickableCanvas));
    $('#image_click_select').html('<select id="img_act_select"> </select>');
    //Add flag option for looking around on click
    $('#img_act_select').append('<option id="looking" '+
                                'value="looking">Look</option>')
    var lookCB = function (pixel) { //Callback for looking at image
        if ($('#img_act_select :selected').val() === 'looking') {
            var camera = $('#'+assistive_teleop.mjpeg.selectBoxId+" :selected").val();
            if (camera === 'Right Arm' || camera === 'Left Arm' || camera === 'AR Tag') {
                var cm = assistive_teleop.mjpeg.cameraModels[assistive_teleop.mjpeg.cameraData[camera].cameraInfo];
                var xyz =  cm.projectPixel(pixel[0], pixel[1], 2);
                var psm = assistive_teleop.ros.composeMsg('geometry_msgs/PointStamped'); 
                psm.header.frame_id = cm.frame_id;
                psm.point.x = xyz[0];
                psm.point.y = xyz[1];
                psm.point.z = xyz[2];
                assistive_teleop.rCamPointSender.publishTarget(psm);
            } else {
                var resp_cb = function (result) {
                    if (result.error_flag !== 0) {
                        log('Error finding 3D point.');
                    } else {
                        clearInterval(assistive_teleop.head.pubInterval);
                        assistive_teleop.head.pointHead(result.pixel3d.pose.position.x,
                                              result.pixel3d.pose.position.y,
                                              result.pixel3d.pose.position.z,
                                              result.pixel3d.header.frame_id);
                        log("Looking at click.");
                    };
                }
                assistive_teleop.p23DClient.call(pixel[0], pixel[1], resp_cb);
            }
        }
    }
    //Add callback to list of callbacks for clickable element
    assistive_teleop.clickableCanvas.onClickCBList.push(lookCB);

    $('#img_act_select').append('<option id="reachLeft" value="reachLeft">Left Hand Goal</option>');
    var reachLeftCB = function (pixel) { //Callback for looking at image
        if ($('#img_act_select :selected').val() ===   'reachLeft') {
            var resultCB = function(result){
                    if (result.error_flag !== 0) {
                        log('Error finding 3D point');
                    } else {
                        assistive_teleop.lPoseSender.sendPose(result.pixel3d);
                        log("Sending Left Arm Reach point command");
                        $('#img_act_select').val('looking');
                    };
                }
            assistive_teleop.p23DClient.call(pixel[0], pixel[1], resultCB);
        }
    }
    //Add callback to list of callbacks for clickable element
    assistive_teleop.clickableCanvas.onClickCBList.push(reachLeftCB);

    // Right hand reach goal on clicked position
    $('#img_act_select').append('<option id="reachRight" value="reachRight">Right Hand Goal</option>');
    var reachRightCB = function (pixel) { //Callback for looking at image
        if ($('#img_act_select :selected').val() ===   'reachRight') {
            var resultCB = function (result) {
                if (result.error_flag !== 0) {
                    log('Error finding 3D point');
                } else {
                    assistive_teleop.rPoseSender.sendPose(result.pixel3d);
                    log("Sending Right Arm Reach point command");
                    $('#img_act_select').val('looking');
                };
            }
            assistive_teleop.p23DClient.call(pixel[0], pixel[1], resultCB);
        }
    }
    //Add callback to list of callbacks for clickable element
    assistive_teleop.clickableCanvas.onClickCBList.push(reachRightCB);

    $('#img_act_select').append('<option id="seedReg" value="seedReg">Register Head</option>');
    var seedRegCB = function (pixel) { //Callback for registering the head
        if ($('#img_act_select :selected').val() ===  'seedReg') {
            var camera = $('#'+assistive_teleop.mjpeg.selectBoxId+" :selected").val();
            cw = assistive_teleop.mjpeg.cameraData[camera].width;
            ch = assistive_teleop.mjpeg.cameraData[camera].height;
            cw_border = Math.round(cw*0.20);
            ch_border = Math.round(ch*0.20);
            if (pixel[0] < cw_border || pixel[0] > (cw-cw_border) ||
                pixel[1] < ch_border || pixel[1] > (ch-ch_border)) {
              assistive_teleop.log("Please center the head in the camera before registering the head");
              $('#img_act_select').val('looking');
            } else {
              assistive_teleop.bodyReg.registerHead(pixel[0], pixel[1]);
              log("Sending head registration command.");
            }
        }
    }
    //Add callback to list of callbacks for clickable element
    assistive_teleop.clickableCanvas.onClickCBList.push(seedRegCB);

    //Add callback for Bowl Registration
    $('#img_act_select').append('<option id="BowlReg" value="BowlReg">Register Bowl</option>');
    var LookBowlCB = function (pixel) { //Callback for registering the bowl
        if ($('#img_act_select :selected').val() ===  'BowlReg') {
            var camera = $('#'+assistive_teleop.mjpeg.selectBoxId+" :selected").val();
            cw = assistive_teleop.mjpeg.cameraData[camera].width;
            ch = assistive_teleop.mjpeg.cameraData[camera].height;
            cw_border = Math.round(cw*0.20);
            ch_border = Math.round(ch*0.20);
            if (pixel[0] < cw_border || pixel[0] > (cw-cw_border) ||
                pixel[1] < ch_border || pixel[1] > (ch-ch_border)) {
              assistive_teleop.log("Please center the bowl in the camera before registering the bowl");
              $('#img_act_select').val('looking');
            } else {
            assistive_teleop.ryds.RegisterBowl(pixel[0], pixel[1]);
            log("Sending bowl registration command.");
            }
        }
    }
    //Add callback to list of callbacks for clickable element
    assistive_teleop.clickableCanvas.onClickCBList.push(LookBowlCB);

    $('#img_act_select').append('<option id="rArmCamLook" value="rArmCamLook">Look: Right Arm Camera</option>')
    var rArmCamLookCB = function (pixel) { //Callback for looking at point with right arm camera
        if ($('#img_act_select :selected').val() === 'rArmCamLook') {
            var resultCB = function(result){
                if (result.error_flag !== 0) {
                    log('Error finding 3D point');
                } else {
                    assistive_teleop.rCamPointSender.callPoseStamped(result.pixel3d);
                };
                $('#img_act_select').val('looking');
            }
            assistive_teleop.p23DClient.call(pixel[0], pixel[1], resultCB);
        }
    }
    //Add callback to list of callbacks
    assistive_teleop.clickableCanvas.onClickCBList.push(rArmCamLookCB);
};
