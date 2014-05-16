var MjpegClient = function (options) {
    var mjpegClient = this;
    var options = options || {};
    this.divId = options.divId;
    this.host = options.host;
    this.port = options.port;
    this.selectBoxId = options.selectBoxId;
    this.width = options.width || 640;
    this.height = options.height || 480;
    this.quality = options.quality || 90;

    this.cameraData = {'Head': {topic:'/head_mount_kinect/rgb/image_color',
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

    this.activeParams = {'topic':this.cameraData['Head'].topic,
                                'width':this.width,
                                'height':this.height,
                                'quality':this.quality}

    this.server = "http://"+this.host+":"+this.port;
    this.imageId = this.divId + "Image";
    $("#"+this.divId).append("<img id="+this.imageId+"></img>");

    this.update = function () {
        var srcStr = this.server+ "/stream"
        for (param in this.activeParams)
        {
            srcStr += "?" + param + '=' + this.activeParams[param]
        }
        $("#"+this.imageId).attr("src", srcStr)
                                  .width(this.activeParams['width'])
                                  .height(this.activeParams['height']);
    };

    // Set parameter value
    this.setParam = function (param, value) {
      this.activeParams[param] = value;
      this.update();
    };

    // Return parameter value
    this.getParam = function (param) {
      return this.activeParams[param];
    };

    // Convenience function for back compatability to set camera topic
    this.setCamera = function (cameraName) {
      $('#'+this.selectBoxId+" :selected").attr("selected", "");
      $('#'+this.selectBoxId+" option[value='"+cameraName+"']" ).attr('selected', 'selected').change();
    };

    this.createCameraMenu = function (divRef) {
      $(divRef).append("<select id='"+this.selectBoxId+"'></select>");
      for (camera in this.cameraData) {
        var optgroupLabel = this.cameraData[camera].optgroup;
        var optgroupID = "cameraGroup"+optgroupLabel;
        if ($('#'+optgroupID).length === 0) {
          $('#cameraSelect').append("<optgroup id='"+optgroupID+"' label='"+optgroupLabel+"'></optgroup>");
        }
        $('#'+optgroupID).append("<option value='"+camera+"'>"+camera+"</option>");
      };
    };

    this.onSelectChange = function () {
      var topic = this.cameraData[$('#'+this.selectBoxId+' :selected').text()].topic;
      this.setParam('topic', topic);
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
    $('#cameraSelect').on('change', window.mjpeg.onSelectChange.bind(window.mjpeg));
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
