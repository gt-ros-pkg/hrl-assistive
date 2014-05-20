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
        var camera = $('#'+window.mjpeg.selectBoxId+" :selected").val();
        return this.convertDisplayToCameraPixel(pixel, window.mjpeg, camera);
    };

    this.onClickCB = function (e) {
        var camera = $('#cameraSelect :selected').text();
        if (window.mjpeg.cameraData[camera].clickable) {
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
    window.rPoseSender = new PoseSender(window.ros, 'wt_r_click_pose');
    window.lPoseSender = new PoseSender(window.ros, 'wt_l_click_pose');
    window.rCamPointSender = new LookatIk(window.ros, '/rightarm_camera/lookat_ik/goal')
    //window.poseSender = new PoseSender(window.ros);
    window.clickableCanvas = new ClickableElement(window.mjpeg.imageId);
    window.p23DClient = new Pixel23DClient(window.ros);

    $('#'+window.mjpeg.imageId).on('click.rfh', window.clickableCanvas.onClickCB.bind(window.clickableCanvas));
    $('#image_click_select').html('<select id="img_act_select"> </select>');
    //Add flag option for looking around on click
    $('#img_act_select').append('<option id="looking" '+
                                'value="looking">Look</option>')
    var lookCB = function (pixel) { //Callback for looking at image
        if ($('#img_act_select :selected').val() ===   'looking') {
            var resp_cb = function (result) {
                if (result.error_flag !== 0) {
                    log('Error finding 3D point.');
                } else {
                    clearInterval(window.head.pubInterval);
                    window.head.pointHead(result.pixel3d.pose.position.x,
                                          result.pixel3d.pose.position.y,
                                          result.pixel3d.pose.position.z,
                                          result.pixel3d.header.frame_id);
                    log("Looking at click.");
                };
            }
            window.p23DClient.call(pixel[0], pixel[1], resp_cb);
        }
    }
    //Add callback to list of callbacks for clickable element
    window.clickableCanvas.onClickCBList.push(lookCB);

    $('#img_act_select').append('<option id="reachLeft" value="reachLeft">Left Hand Goal</option>');
    var reachLeftCB = function (pixel) { //Callback for looking at image
        if ($('#img_act_select :selected').val() ===   'reachLeft') {
            var resultCB = function(result){
                    if (result.error_flag !== 0) {
                        log('Error finding 3D point');
                    } else {
                        window.lPoseSender.sendPose(result.pixel3d);
                        log("Sending Left Arm Reach point command");
                        $('#img_act_select').val('looking');
                    };
                }
            window.p23DClient.call(pixel[0], pixel[1], resultCB);
        }
    }
    //Add callback to list of callbacks for clickable element
    window.clickableCanvas.onClickCBList.push(reachLeftCB);

    // Right hand reach goal on clicked position
    $('#img_act_select').append('<option id="reachRight" value="reachRight">Right Hand Goal</option>');
    var reachRightCB = function (pixel) { //Callback for looking at image
        if ($('#img_act_select :selected').val() ===   'reachRight') {
            var resultCB = function (result) {
                if (result.error_flag !== 0) {
                    log('Error finding 3D point');
                } else {
                    window.rPoseSender.sendPose(result.pixel3d);
                    log("Sending Right Arm Reach point command");
                    $('#img_act_select').val('looking');
                };
            }
            window.p23DClient.call(pixel[0], pixel[1], resultCB);
        }
    }
    //Add callback to list of callbacks for clickable element
    window.clickableCanvas.onClickCBList.push(reachRightCB);

    $('#img_act_select').append('<option id="seedReg" value="seedReg">Register Head</option>');
    var seedRegCB = function (pixel) { //Callback for registering the head
        if ($('#img_act_select :selected').val() ===  'seedReg') {
            var camera = $('#'+window.mjpeg.selectBoxId+" :selected").val();
            cw = window.mjpeg.cameraData[camera].width;
            ch = window.mjpeg.cameraData[camera].height;
            cw_border = Math.round(cw*0.20);
            ch_border = Math.round(ch*0.20);
            if (pixel[0] < cw_border || pixel[0] > (cw-cw_border) ||
                pixel[1] < ch_border || pixel[1] > (ch-ch_border)) {
              window.log("Please center the head in the camera before registering the head");
              $('#img_act_select').val('looking');
            } else {
              window.bodyReg.registerHead(pixel[0], pixel[1]);
              log("Sending head registration command.");
            }
        }
    }
    //Add callback to list of callbacks for clickable element
    window.clickableCanvas.onClickCBList.push(seedRegCB);

    
    $('#img_act_select').append('<option id="rArmCamLook" value="rArmCamLook">Look: Right Arm Camera</option>')
    var rArmCamLookCB = function (pixel) { //Callback for looking at point with right arm camera
        if ($('#img_act_select :selected').val() === 'rArmCamLook') {
            var resultCB = function(result){
                if (result.error_flag !== 0) {
                    log('Error finding 3D point');
                } else {
                    window.rCamPointSender.callPoseStamped(result.pixel3d);
                };
                $('#img_act_select').val('looking');
            }
            window.p23DClient.call(pixel[0], pixel[1], resultCB);
        }
    }
    //Add callback to list of callbacks
    window.clickableCanvas.onClickCBList.push(rArmCamLookCB);
};

//<!--<option id="ell_global_move" value="ell_global_move">Move around Ellipse</option>-->\
//<option id="skin_linear_move" value="skin_linear_move">Move to point with skin</option>\
//<!--<option id="na" value="norm_approach">Normal Approach</option>-->\
//<!--<option id="touch" value="touch">Touch</option>-->\
//<!--<option id="wipe" value="wipe">Wipe</option>-->\
//<!--<option id="swipe" value="swipe">Swipe</option>-->\
//<!--<option id="poke" value="poke">Poke</option>-->\
//<!--<option id="surf_wipe" value="surf_wipe">Surface Wipe</option>-->\
//<!--<option id="grasp" value="grasp">Grasp</option>-->\
//<!--<option id="reactive_grasp" value="reactive_grasp">Reactive Grasp</option>-->\
//<!--<option id="contact_approach" value="contact_approach">Approach until Contact</option>-->\
//<!--<option id="hfc_contact_approach" value="hfc_contact_approach">Approach until Contact HFC</option>\
//<option id="hfc_swipe" value="hfc_swipe">Swipe HFC</option>\
//<option id="hfc_wipe" value="hfc_wipe">Wipe HFC</option>-->\
//</select>'
    
//clickableElement.Response = function(result_pose){
//    var  _element = document.getElementById(clickableElement.elementID);
//    var _selector = document.getElementById(clickableElement.selectorID);
//    switch (_selector[_selector.selectedIndex].value){
//        case 'looking':
//            log("Sending look to point command");
//            clearInterval(window.head.pubInterval);
//            window.head.pointHead(result_pose.pose.position.x, result_pose.pose.position.y,
//                                  result_pose.pose.position.z, result_pose.header.frame_id);
//            break
//        case 'head_nav_goal':
//                log("Sending navigation seed position");
//                node.publish('head_nav_goal', 'geometry_msgs/PoseStamped', json(result_pose));
//            break
//        case 'norm_approach':
//            log('Sending '+window.arm().toUpperCase()+ ' Arm Normal Approach Command')
//                node.publish('norm_approach_'+window.arm(), 'geometry_msgs/PoseStamped', json(result_pose));
//            break
//        case 'grasp':
//            log('Sending command to attempt to grasp object with '+window.arm().toUpperCase()+' arm')
//                node.publish('wt_grasp_'+window.arm()+'_goal', 'geometry_msgs/PoseStamped', json(result_pose));
//            break
//        case 'reactive_grasp':
//            log('Sending command to grasp object with reactive grasping with '+window.arm().toUpperCase()+' arm');
//            node.publish('wt_rg_'+window.arm()+'_goal', 'geometry_msgs/PoseStamped', json(result_pose));
//            break
//        case 'wipe':
//            node.publish('wt_wipe_'+window.arm()+'_goals', 'geometry_msgs/PoseStamped', json(result_pose));
//            if (window.force_wipe_count == 0) {
//                window.force_wipe_count = 1;
//                log('Sending start position for force-sensitive wiping')
//            } else if (window.force_wipe_count == 1) {
//                window.force_wipe_count = 0;
//                log('Sending end position for force-sensitive wiping')
//            };    
//            break
//        case 'swipe':
//            log('Sending command to swipe from start to finish with '+window.arm().toUpperCase+' arm')
//            node.publish('wt_swipe_'+window.arm()+'_goals', 'geometry_msgs/PoseStamped', json(result_pose));
//            break
//        case 'poke':
//            log('Sending command to poke point with '+window.arm().toUpperCase()+' arm')
//            node.publish('wt_poke_'+window.arm()+'_point', 'geometry_msgs/PoseStamped', json(result_pose));
//            break
//        case 'contact_approach':
//            log('Sending command to approaching point by moving until contact with '+window.arm().toUpperCase()+' arm')
//            node.publish('wt_contact_approach_'+window.arm(), 'geometry_msgs/PoseStamped', json(result_pose));
//            break
//        case 'ell_global_move':
//            log('Sending command to approach head point by moving around ellipse');
//            window.ellControl.sendClickedMove(result_pose);
//            break
//        case 'skin_linear_move':
//            log('Sending command to approach point using skin');
//            window.poseSender.sendPose(result_pose);
//            break
//    };
//    if (window.force_wipe_count == 0){
//    $('#img_act_select').val('looking');
//    };
//};

