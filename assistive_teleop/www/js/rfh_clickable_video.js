var count_surf_wipe_right=count_surf_wipe_left=force_wipe_count=0;

image_click_select_html = 
'<select id="img_act_select">\
    <option id="looking" selected="selected" value="looking">Look</option>\
    <option id="seed_reg" value="seed_reg">Register Head</option>\
    <option id="skin_goal" value="skin_goal">Move with Skin</option>\
    <!--<option id="na" value="norm_approach">Normal Approach</option>-->\
    <!--<option id="touch" value="touch">Touch</option>-->\
    <!--<option id="wipe" value="wipe">Wipe</option>-->\
    <!--<option id="swipe" value="swipe">Swipe</option>-->\
    <!--<option id="poke" value="poke">Poke</option>-->\
    <!--<option id="surf_wipe" value="surf_wipe">Surface Wipe</option>-->\
    <!--<option id="grasp" value="grasp">Grasp</option>-->\
    <!--<option id="reactive_grasp" value="reactive_grasp">Reactive Grasp</option>-->\
    <!--<option id="contact_approach" value="contact_approach">Approach until Contact</option>-->\
    <!--<option id="hfc_contact_approach" value="hfc_contact_approach">Approach until Contact HFC</option>\
    <option id="hfc_swipe" value="hfc_swipe">Swipe HFC</option>\
    <option id="hfc_wipe" value="hfc_wipe">Wipe HFC</option>-->\
    </select>'

$(function(){
        $('#image_click_select').html(image_click_select_html);
        });

window.addEventListener('click', function(e){
        if (e.target.id == 'mjpeg_canvas'){
            var canvasOffset = $('#'+e.target.id).offset()
            var clickX 
            var clickY
            if (e.pageX|| e.pageY){
                clickX = e.pageX - canvasOffset.left
                clickY = e.pageY - canvasOffset.top
            } else if (e.clientX || e.clientY) {
                clickX = 1+e.clientX + document.body.scrollLeft
                            + document.documentElement.scrollLeft;
                clickY = 1+e.clientY + document.body.scrollTop
                            + document.documentElement.scrollTop;
            };
        image_click(clickX, clickY);
        };
});

function image_click(pixX, pixY){
    var im_pixel = [pixX, pixY]
    if ($('#img_act_select option:selected').val() == 'surf_wipe') {
        surf_points_out = window.gm_point
            surf_points_out.x = im_pixel[0]
            surf_points_out.y = im_pixel[1]
            log('Surface Wipe');
        log('Surface Wipe '+window.arm().toUpperCase()+' '+window.count_surf_wipe_right.toString())
            node.publish('wt_surf_wipe_'+window.arm()+'_points', 'geometry_msgs/Point', json(surf_points_out));
        if (window.count_surf_wipe == 0){
            log("Sending start position for surface-aware wiping");
            window.count_surf_wipe = 1;
        } else if (window.count_surf_wipe == 1){
            log("Sending end position for surface-aware wiping");
            window.count_surf_wipe = 0;
            $('#img_act_select').val('looking');
        }
    } else if ($('#img_act_select option:selected').val() == 'seed_reg'){
        console.log("Calling Registration Service with "+im_pixel[0].toString() +", "+im_pixel[1].toString())
            node.rosjs.callService('/initialize_registration',
                    '['+json(im_pixel[0])+','+json(im_pixel[1])+']',
                    function(msg){console.log("Registration Service Returned")})
            $('#img_act_select').val('looking');
    } else {
        get_im_3d(im_pixel[0],im_pixel[1])
    };
};

function get_im_3d(x,y){
    window.point_2d.pixel_u = x;
    window.point_2d.pixel_v = y;
    log('Sending Pixel_2_3d request, awaiting response');
    node.rosjs.callService('/pixel_2_3d',
            '['+json(window.point_2d.pixel_u)+','+json(window.point_2d.pixel_v)+']',
            function(msg){
            log('pixel_2_3d response received');
            point_3d=msg.pixel3d;
            p23d_response(point_3d)
            })
};

function p23d_response(point_3d){
    switch ($('#img_act_select option:selected').val()){
        case 'looking':
            log("Sending look to point command");
            window.head_pub = window.clearInterval(head_pub);
            pub_head_goal(point_3d.pose.position.x, point_3d.pose.position.y, point_3d.pose.position.z, point_3d.header.frame_id);
            break
        case 'head_nav_goal':
                log("Sending navigation seed position");
                node.publish('head_nav_goal', 'geometry_msgs/PoseStamped', json(point_3d));
            break
        case 'norm_approach':
            log('Sending '+window.arm().toUpperCase()+ ' Arm Normal Approach Command')
                node.publish('norm_approach_'+window.arm(), 'geometry_msgs/PoseStamped', json(point_3d));
            break
        case 'grasp':
            log('Sending command to attempt to grasp object with '+window.arm().toUpperCase()+' arm')
                node.publish('wt_grasp_'+window.arm()+'_goal', 'geometry_msgs/PoseStamped', json(point_3d));
            break
        case 'reactive_grasp':
            log('Sending command to grasp object with reactive grasping with '+window.arm().toUpperCase()+' arm');
            node.publish('wt_rg_'+window.arm()+'_goal', 'geometry_msgs/PoseStamped', json(point_3d));
            break
        case 'wipe':
            node.publish('wt_wipe_'+window.arm()+'_goals', 'geometry_msgs/PoseStamped', json(point_3d));
            if (window.force_wipe_count == 0) {
                window.force_wipe_count = 1;
                log('Sending start position for force-sensitive wiping')
            } else if (window.force_wipe_count == 1) {
                window.force_wipe_count = 0;
                log('Sending end position for force-sensitive wiping')
            };    
            break
        case 'swipe':
            log('Sending command to swipe from start to finish with '+window.arm().toUpperCase+' arm')
            node.publish('wt_swipe_'+window.arm()+'_goals', 'geometry_msgs/PoseStamped', json(point_3d));
            break
        case 'poke':
            log('Sending command to poke point with '+window.arm().toUpperCase()+' arm')
            node.publish('wt_poke_'+window.arm()+'_point', 'geometry_msgs/PoseStamped', json(point_3d));
            break
        case 'contact_approach':
            log('Sending command to approaching point by moving until contact with '+window.arm().toUpperCase()+' arm')
            node.publish('wt_contact_approach_'+window.arm(), 'geometry_msgs/PoseStamped', json(point_3d));
        case 'skin_goal':
            log('Sending pose as goal to approach with skin')
            node.publish('wt_skin_goal', 'geometry_msgs/PoseStamped', json(point_3d));
            break
    };
    if (window.force_wipe_count == 0){
        $('#img_act_select').val('looking');
    };
};
