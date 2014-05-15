var MjpegClient = function (options) {
    var mjpegClient = this;
    var options = options || {};
    mjpegClient.divId = options.divId;
    mjpegClient.host = options.host;
    mjpegClient.port = options.port;
    mjpegClient.selectBoxId = options.selectBoxId;
    mjpegClient.width = options.width || 640;
    mjpegClient.height = options.height || 480;
    mjpegClient.quality = options.quality || 90;

    mjpegClient.cameraData = {'Head': {topic:'/head_mount_kinect/rgb/image_color',
                                       optgroup:'Default',
                                       cameraInfo:'/head_mount_kienct/rgb/camera_info',
                                       clickable:true,
                                       width:1280,
                                       height:1024},
                              'Right Arm': {topic: '/r_forearm_cam/image_color_rotated',
                                            optgroup:'Default',
                                            cameraInfo: '/r_forearm_cam/camera_info',
                                            clickable:false,
                                            width:640,
                                            height:480},
                              'Left Arm': {topic: '/l_forearm_cam/image_color_rotated',
                                           optgroup:'Default',
                                           cameraInfo: '/l_forearm_cam/camera_info',
                                           clickable:false,
                                           width:640,
                                           height:480},
                              'AR Tag': {topic:'/ar_servo/confirmation_rotated',
                                         optgroup:'Special',
                                         cameraInfo:'/r_forearm_cam/camera_info',
                                         clickable:false,
                                         width:640,
                                         height:480},
                              'Head Registration': {topic: '/head_registration/confirmation',
                                                    optgroup:'Special',
                                                    cameraInfo: 'head_mount_kinect/rgb/camera_info',
                                                    clickable: true,
                                                    width: 1280,
                                                    height:1024}
    }

    mjpegClient.activeParams = {'topic':mjpegClient.cameraData['Head'].topic,
                                'width':mjpegClient.width,
                                'height':mjpegClient.height,
                                'quality':mjpegClient.quality}

    mjpegClient.server = "http://"+mjpegClient.host+":"+mjpegClient.port;
    mjpegClient.imageId = mjpegClient.divId + "Image";
    $("#"+mjpegClient.divId).append("<img id="+mjpegClient.imageId+"></img>");

    mjpegClient.update = function () {
        var srcStr = mjpegClient.server+ "/stream"
        for (param in mjpegClient.activeParams)
        {
            srcStr += "?" + param + '=' + mjpegClient.activeParams[param]
        }
        $("#"+mjpegClient.imageId).attr("src", srcStr);
    };

    // Set parameter value
    mjpegClient.setParam = function (param, value) {
      mjpegClient.activeParams[param] = value;
      mjpegClient.update();
    };

    // Return parameter value
    mjpegClient.getParam = function (param) {
      return mjpegClient.activeParams[param];
    };

    // Convenience function for back compatability to set camera topic
    mjpegClient.setCamera = function (cameraName) {
      $('#'+mjpegClient.selectBoxId+" :selected").attr("selected", "");
      $('#'+mjpegClient.selectBoxId+" option[value='"+cameraName+"']" ).attr('selected', 'selected').change();
    };

    mjpegClient.createCameraMenu = function (divRef) {
        $(divRef).append("<select id='"+mjpegClient.selectBoxId+"'></select>");
        for (camera in mjpegClient.cameraData) {
          var optgroupLabel = mjpegClient.cameraData[camera].optgroup;
          var optgroupID = "cameraGroup"+optgroupLabel;
          if ($('#'+optgroupID).length === 0) {
            $('#cameraSelect').append("<optgroup id='"+optgroupID+"' label='"+optgroupLabel+"'></optgroup>");
          }
          $('#'+optgroupID).append("<option value='"+camera+"'>"+camera+"</option>");
        };
    };
};

var initMjpegCanvas = function (divId) {
    var divRef = '#' + divId;
    $(divRef).off('click'); //Disable click detection so clickable_element catches it
    // Build the html for image feed and controls below
    $(divRef).append("<table>"+
                       "<tr><td colspan='4'><div id='mjpegDiv'></div></td></tr>" +
                       "<tr id='underVideoBar'>" + 
                         "<td style='text-align:right'>On Image Click:</td>" +
                         "<td id='image_click_select'></td>" + 
                         "<td style='text-align:right'>Camera:</td>" +
                         "<td id='cameraSelectCell'></td>" + 
                       "</tr>" +
                     "</table>");

    // Initialize the mjpeg client
    window.mjpeg = new MjpegClient({'divId': 'mjpegDiv',
                                    "host": window.ROBOT,
                                    "port": 8080,
                                    "selectBoxId": 'cameraSelect',
                                    "width": 640,//1280,
                                    "height": 512,//1024,//480,
                                    "quality": 85});
    // Initialize the camera selection menu
    window.mjpeg.createCameraMenu('#cameraSelectCell');
    $('#cameraSelect').on('change', function() {
      var topic = window.mjpeg.cameraData[$('#cameraSelect :selected').text()].topic;
        window.mjpeg.setParam('topic', topic);
    });
    // Apply these initial settings
    window.mjpeg.update();    

    // Make the image resizeable
    var resizeStopCB = function (event, ui) {
      window.mjpeg.setParam('height', Math.round(ui.size.height));
      window.mjpeg.setParam('width', Math.round(ui.size.width));
      window.mjpeg.update()
    };
    $('#'+window.mjpeg.divId).resizable({aspectRatio:true,
                                         alsoResize:'#'+window.mjpeg.imageId,
                                         autoHide:true,
                                         ghost:true,
                                         delay:250,
                                         handles:'se',
                                         distance:7,
                                         maxWidth:1280,
                                         minWidth:320,
                                         maxHeight:1024,
                                         minHeight:240,
                                         stop:resizeStopCB});
};
