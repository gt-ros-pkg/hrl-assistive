var ClickableElement = function(elementID){
    'use strict';
    var clickableElement = this;
    clickableElement.elementID = elementID;
	var _element = document.getElementById(clickableElement.elementID);
    clickableElement.onClickCBList = [];
    clickableElement.onClickCB = function (e) {
        for (var i = 0; i<clickableElement.onClickCBList.length; i += 1) {
            clickableElement.onClickCBList[i](e);
        }
    }
    _element.addEventListener('click', clickableElement.onClickCB);
}

var PoseSender = function (ros) {
    'use strict';
    var poseSender = this;
    poseSender.ros = ros;
    poseSender.posePub = new poseSender.ros.Topic({
        name: '/wt_clicked_pose',
        messageType: 'geometry_msgs/PoseStamped'})
    poseSender.posePub.advertise();
    poseSender.sendPose = function (poseStamped) {
        var msg = new poseSender.ros.Message(poseStamped);
        poseSender.posePub.publish(msg);
    }
}

var pixel23DClient = function (ros) {
    'use strict';
    var p23D = this;
    p23D.ros = ros;
    p23D.serviceClient =  new p23D.ros.Service({
                                        name: '/pixel_2_3d',
                                        serviceType: 'Pixel23d'});
}

var clickInElement = function (e) {
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
    return [posx,posy]
};

var initClickableActions = function () {
    $('#image_click_select').html('<select id="img_act_select"> </select>');
    //Add flag option for looking around on click
    $('#img_act_select').append('<option id="looking" '+
                                'value="looking">Look</option>')
    var lookCB = function (e) { //Callback for looking at image
        var sel = document.getElementById('img_act_select');
        if (sel[sel.selectedIndex].value === 'looking') {
            pointUV = window.clickInElement(e);
            var request = new window.ros.ServiceRequest({
                                        'pixel_u':pointUV[0],
                                        'pixel_v':pointUV[1]});
            window.p23DClient.serviceClient.callService(request,
                function(result){
                    if (result.error_flag !== 0) {
                        log('Error finding 3D point');
                        return
                    } else {
                        result_pose = result.pixel3d;
                        log('pixel_2_3d response received');
                        clearInterval(window.head.pubInterval);
                        window.head.pointHead(result_pose.pose.position.x,
                                              result_pose.pose.position.y,
                                              result_pose.pose.position.z,
                                              result_pose.header.frame_id);
                        log("Sending look to point command");
                    };
                }
            )
        }
    }
    //Add callback to list of callbacks for clickable element
    window.clickableCanvas.onClickCBList.push(lookCB);

    $('#img_act_select').append('<option id="reachLeft" '+
                                'value="reachLeft">Left Hand Goal</option>')
    var reachLeftCB = function (e) { //Callback for looking at image
        var sel = document.getElementById('img_act_select');
        if (sel[sel.selectedIndex].value === 'reachLeft') {
            pointUV = window.clickInElement(e);
            var request = new window.ros.ServiceRequest({
                                        'pixel_u':pointUV[0],
                                        'pixel_v':pointUV[1]});
            window.p23DClient.serviceClient.callService(request,
                function(result){
                    if (result.error_flag !== 0) {
                        log('Error finding 3D point');
                        return
                    } else {
                        result_pose = result.pixel3d;
                        log('pixel_2_3d response received');
                        window.poseSender.sendPose(result_pose);
                        log("Sending Left Arm Reach point command");
                    };
                }
            )
        }
    }
    //Add callback to list of callbacks for clickable element
    window.clickableCanvas.onClickCBList.push(reachLeftCB);

    $('#img_act_select').append('<option id="seedReg" '+
                                'value="seedReg">Register Head</option>')
    var seedRegCB = function (e) { //Callback for looking at image
        var sel = document.getElementById('img_act_select');
        if (sel[sel.selectedIndex].value === 'seedReg') {
            pointUV = window.clickInElement(e);
            window.ellControl.registerHead(pointUV[0], pointUV[1]);
            log("Sending Head Registration Command");
        }
    }
    //Add callback to list of callbacks for clickable element
    window.clickableCanvas.onClickCBList.push(seedRegCB);
}
//<option id="seed_reg" value="seed_reg">Register Head</option>
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

