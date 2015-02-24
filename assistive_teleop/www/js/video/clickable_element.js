var ClickableElement = function(elementID) {
    'use strict';
    this.elementID = elementID;
    this.onClickCBList = [];
    this.getClickPixel = function (e) {
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
        var camera = $('#'+RFH.mjpeg.selectBoxId+" :selected").val();
        return this.convertDisplayToCameraPixel(pixel, RFH.mjpeg, camera);
    };

    this.onClickCB = function (e) {
        var camera = $('#cameraSelect :selected').text();
        if (RFH.mjpeg.cameraData[camera].clickable) {
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
    var self = this;
    self.ros = ros;
    self.serviceClient =  new self.ros.Service({
                                        name: '/pixel_2_3d',
                                        serviceType: 'Pixel23d'});
    self.call = function (u, v, cb) {
        var req = new self.ros.ServiceRequest({'pixel_u':u, 'pixel_v':v});
        self.serviceClient.callService(req, cb);
    }
}


var initClickableActions = function () {
    RFH.rPoseSender = new PoseSender(RFH.ros, 'wt_r_click_pose');
    RFH.lPoseSender = new PoseSender(RFH.ros, 'wt_l_click_pose');
    RFH.rCamPointSender = new LookatIk(RFH.ros, '/rightarm_camera/lookat_ik/goal')
    //RFH.poseSender = new PoseSender(RFH.ros);
    RFH.clickableCanvas = new ClickableElement(RFH.mjpeg.imageId);
    RFH.p23DClient = new Pixel23DClient(RFH.ros);

    $('#'+RFH.mjpeg.imageId).on('click.rfh', RFH.clickableCanvas.onClickCB.bind(RFH.clickableCanvas));
    $('#image_click_select').html('<select id="img_act_select"> </select>');
    //Add flag option for looking around on click
    $('#img_act_select').append('<option id="looking" '+
                                'value="looking">Look</option>')
    var lookCB = function (pixel) { //Callback for looking at image
        if ($('#img_act_select :selected').val() === 'looking') {
            var camera = $('#'+RFH.mjpeg.selectBoxId+" :selected").val();
            if (camera === 'Right Arm' || camera === 'Left Arm' || camera === 'AR Tag') {
                var cm = RFH.mjpeg.cameraModels[RFH.mjpeg.cameraData[camera].cameraInfo];
                var xyz =  cm.projectPixel(pixel[0], pixel[1], 2);
                var psm = RFH.ros.composeMsg('geometry_msgs/PointStamped'); 
                psm.header.frame_id = cm.frame_id;
                psm.point.x = xyz[0];
                psm.point.y = xyz[1];
                psm.point.z = xyz[2];
                RFH.rCamPointSender.publishTarget(psm);
            } else {
                var resp_cb = function (result) {
                    if (result.error_flag !== 0) {
                        log('Error finding 3D point.');
                    } else {
                        clearInterval(RFH.head.pubInterval);
                        RFH.head.pointHead(result.pixel3d.pose.position.x,
                                              result.pixel3d.pose.position.y,
                                              result.pixel3d.pose.position.z,
                                              result.pixel3d.header.frame_id);
                        log("Looking at click.");
                    };
                }
                RFH.p23DClient.call(pixel[0], pixel[1], resp_cb);
            }
        }
    }
    //Add callback to list of callbacks for clickable element
    RFH.clickableCanvas.onClickCBList.push(lookCB);

    $('#img_act_select').append('<option id="reachLeft" value="reachLeft">Left Hand Goal</option>');
    var reachLeftCB = function (pixel) { //Callback for looking at image
        if ($('#img_act_select :selected').val() ===   'reachLeft') {
            var resultCB = function(result){
                    if (result.error_flag !== 0) {
                        log('Error finding 3D point');
                    } else {
                        RFH.lPoseSender.sendPose(result.pixel3d);
                        log("Sending Left Arm Reach point command");
                        $('#img_act_select').val('looking');
                    };
                }
            RFH.p23DClient.call(pixel[0], pixel[1], resultCB);
        }
    }
    //Add callback to list of callbacks for clickable element
    RFH.clickableCanvas.onClickCBList.push(reachLeftCB);

    // Right hand reach goal on clicked position
    $('#img_act_select').append('<option id="reachRight" value="reachRight">Right Hand Goal</option>');
    var reachRightCB = function (pixel) { //Callback for looking at image
        if ($('#img_act_select :selected').val() ===   'reachRight') {
            var resultCB = function (result) {
                if (result.error_flag !== 0) {
                    log('Error finding 3D point');
                } else {
                    RFH.rPoseSender.sendPose(result.pixel3d);
                    log("Sending Right Arm Reach point command");
                    $('#img_act_select').val('looking');
                };
            }
            RFH.p23DClient.call(pixel[0], pixel[1], resultCB);
        }
    }
    //Add callback to list of callbacks for clickable element
    RFH.clickableCanvas.onClickCBList.push(reachRightCB);

    $('#img_act_select').append('<option id="seedReg" value="seedReg">Register Head</option>');
    var seedRegCB = function (pixel) { //Callback for registering the head
        if ($('#img_act_select :selected').val() ===  'seedReg') {
            var camera = $('#'+RFH.mjpeg.selectBoxId+" :selected").val();
            cw = RFH.mjpeg.cameraData[camera].width;
            ch = RFH.mjpeg.cameraData[camera].height;
            cw_border = Math.round(cw*0.20);
            ch_border = Math.round(ch*0.20);
            if (pixel[0] < cw_border || pixel[0] > (cw-cw_border) ||
                pixel[1] < ch_border || pixel[1] > (ch-ch_border)) {
              RFH.log("Please center the head in the camera before registering the head");
              $('#img_act_select').val('looking');
            } else {
              RFH.bodyReg.registerHead(pixel[0], pixel[1]);
              log("Sending head registration command.");
            }
        }
    }
    //Add callback to list of callbacks for clickable element
    RFH.clickableCanvas.onClickCBList.push(seedRegCB);
    
    $('#img_act_select').append('<option id="rArmCamLook" value="rArmCamLook">Look: Right Arm Camera</option>')
    var rArmCamLookCB = function (pixel) { //Callback for looking at point with right arm camera
        if ($('#img_act_select :selected').val() === 'rArmCamLook') {
            var resultCB = function(result){
                if (result.error_flag !== 0) {
                    log('Error finding 3D point');
                } else {
                    RFH.rCamPointSender.callPoseStamped(result.pixel3d);
                };
                $('#img_act_select').val('looking');
            }
            RFH.p23DClient.call(pixel[0], pixel[1], resultCB);
        }
    }
    //Add callback to list of callbacks
    RFH.clickableCanvas.onClickCBList.push(rArmCamLookCB);
};
