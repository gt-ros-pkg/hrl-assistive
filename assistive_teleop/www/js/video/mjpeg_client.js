var MjpegClient = function (options) {
    var self = this;
    var options = options || {};
    self.ros = options.ros;
    self.divId = options.divId;
    self.host = options.host;
    self.port = options.port;
    self.selectBoxId = options.selectBoxId;
    self.width = options.width || 640;
    self.height = options.height || 480;
    self.quality = options.quality || 90;

    self.cameraData = {'Head': {topic:'/head_mount_kinect/qhd/image_color',//kinect 1 is image_color 
                                       optgroup:'Default',
                                       cameraInfo:'/head_mount_kinect/qhd/camera_info', //
                                       clickable:true,
                                       rotated: false,
                                       width: 960, //960, //1280, originally 640
                                       height: 540},// 540},//1024, originally 480}
                              'Right Arm': {topic: '/r_forearm_cam/image_color_rotated',
                                            optgroup:'Default',
                                            cameraInfo: '/r_forearm_cam/camera_info',
                                            clickable:true,
                                            rotated: true,
                                            width:640,
                                            height:480},
                              'Left Arm': {topic: '/l_forearm_cam/image_color_rotated',
                                           optgroup:'Default',
                                           cameraInfo: '/l_forearm_cam/camera_info',
                                           clickable: false,
                                           rotated: true,
                                           width:640,
                                           height:480},
                              'Wrist': {topic: '/SR300/rgb/image_raw',
                                           optgroup:'Default',
                                           cameraInfo: '/SR300/rgb/camera_info',
                                           clickable: false,
                                           rotated: false,
                                           width:640,
                                           height:480},
                              'AR Tag': {topic:'/ar_servo/confirmation_rotated',
                                         optgroup:'Special',
                                         cameraInfo:'/r_forearm_cam/camera_info',
                                         clickable:false,
                                         rotated: true,
                                         width:640,
                                         height:480},
                              'Feedback': {topic: '/manipulation_task/overlay',
                                                    optgroup:'Special',
                                                    cameraInfo: 'head_mount_kinect/qhd/camera_info',
                                                    clickable: true,
                                                    rotated: false,
                                                    width: 960, //1280,
                                                    height:540}//1024}
    }

    self.cameraModels = {};
    self.updateCameraModels = function () {
        for (camera in self.cameraData) {
            var infoTopic = self.cameraData[camera].cameraInfo;
            var type = typeof(self.cameraModels[infoTopic]);
            if (typeof(self.cameraModels[infoTopic]) === "undefined") {
                var rotated = self.cameraData[camera].rotated;
                self.cameraModels[infoTopic] = new CameraModel({ros:self.ros,
                                                                infoTopic: infoTopic,
                                                                rotated: rotated});
            } else {
                self.cameraModels[infoTopic].update();
            }
        }
    };
    self.updateCameraModels();

    self.activeParams = {'topic':self.cameraData['Head'].topic,
                                'width':self.width,
                                'height':self.height,
                                'quality':self.quality}

    self.server = "http://"+self.host+":"+self.port;
    self.imageId = self.divId + "Image";
    $("#"+self.divId).append("<img id="+self.imageId+"></img>");

    self.update = function () {
        var srcStr = self.server+ "/stream?"
        for (param in self.activeParams)
        {
            srcStr += param + '=' + self.activeParams[param] + '&'
        }
        $("#"+self.imageId).attr("src", srcStr)
                                  .width(self.activeParams['width'])
                                  .height(self.activeParams['height']);
    };

    // Set parameter value
    self.setParam = function (param, value) {
      self.activeParams[param] = value;
      self.update();
    };

    // Return parameter value
    self.getParam = function (param) {
      return self.activeParams[param];
    };

    // Convenience function for back compatability to set camera topic
    self.setCamera = function (cameraName) {
      $('#'+self.selectBoxId+" :selected").attr("selected", "");
      $('#'+self.selectBoxId+" option[value='"+cameraName+"']" ).attr('selected', 'selected').change();
    };

    self.createCameraMenu = function (divRef) {
      $(divRef).append("<select id='"+self.selectBoxId+"'></select>");
      for (camera in self.cameraData) {
        var optgroupLabel = self.cameraData[camera].optgroup;
        var optgroupID = "cameraGroup"+optgroupLabel;
        if ($('#'+optgroupID).length === 0) {
          $('#cameraSelect').append("<optgroup id='"+optgroupID+"' label='"+optgroupLabel+"'></optgroup>");
        }
        $('#'+optgroupID).append("<option value='"+camera+"'>"+camera+"</option>");
      };
    };

    self.onSelectChange = function () {
      var topic = self.cameraData[$('#'+self.selectBoxId+' :selected').text()].topic;
      self.setParam('topic', topic);
    };
};

var CameraModel = function (options) {
    var self = this;
    var options = options || {};
    self.ros = options.ros;
    self.infoTopic = options.infoTopic;
    self.width = options.default_width || 640;
    self.height = options.default_height || 480;
    self.rotated = options.rotated || false;

    self.D = [];
    self.K = [];
    self.R = [];
    self.P = [];
    self.KR = [];
    self.KR_inv = [];
    self.frame_id = '';
    self.ros.getMsgDetails('sensor_msgs/CameraInfo');

    self.makeMatrix = function (arr, rows, cols) {
        // max a (rosw)x(cols) matrix from a single array
        console.assert(rows*cols === arr.length, 
                       "Cannot make a %dx%d matrix with only %d numbers!",
                       rows, cols, arr.length);
        var matrix = [];
        for (var r = 0; r<rows; r += 1) {
            matrix[r] = arr.slice(r*cols, r*cols+cols);
        }
        return matrix
    }

    self.subscriber = new self.ros.Topic({
        name: self.infoTopic,
        messageType: 'sensor_msgs/CameraInfo'});

    self.msgCB = function (infoMsg) {
        console.log("Updated camera model from "+self.infoTopic)
        if (self.rotated) {
            self.frame_id = infoMsg.header.frame_id + '_rotated';
        } else {
            self.frame_id = infoMsg.header.frame_id;
        }
            
        self.height = infoMsg.height;
        self.width = infoMsg.width;
        self.distortion_model = infoMsg.distortion_model;
        self.D = infoMsg.D;
        self.K = self.makeMatrix(infoMsg.K, 3, 3);
        self.R = self.makeMatrix(infoMsg.R, 3, 3);
        self.P = self.makeMatrix(infoMsg.P, 3, 4);
        // Not collecting data on binning, ROI
        self.KR = numeric.dot(self.K, self.R);
        self.KR_inv = numeric.inv(self.KR);
        self.subscriber.unsubscribe(); // Close subscriber to save bandwidth
        }

    //Get up-to-date data on creation
    self.subscriber.subscribe(self.msgCB);

    self.update = function () {
        // Re-subscribe to get new parameters
        self.subscriber.subscribe(self.msgCB);
    }

    // back-project a pixel some distance into the real world
    // Returns a geoemtry_msgs/PointStamped msg
    self.projectPixel = function (px, py, dist) { 
       var pixel_hom = [[px],[py],[1]]; //Pixel value in homogeneous coordinates
       var vec = numeric.dot(self.KR_inv, pixel_hom);
       vec = numeric.transpose(vec)[0];
       var mag = numeric.norm2(vec);
       return numeric.mul(dist/mag, vec);
    }
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
    assistive_teleop.mjpeg = new MjpegClient({ros: assistive_teleop.ros,
                                    divId: 'mjpegDiv',
                                    host: assistive_teleop.ROBOT,
                                    port: 8080,
                                    selectBoxId: 'cameraSelect',
                                    width: 640,//1280,
                                    height: 512,//1024,//480,
                                    quality: 85});
    // Initialize the camera selection menu
    assistive_teleop.mjpeg.createCameraMenu('#cameraSelectCell');
    $('#cameraSelect').on('change', assistive_teleop.mjpeg.onSelectChange.bind(assistive_teleop.mjpeg));
    // Apply these initial settings
    assistive_teleop.mjpeg.update();    

    // Make the image resizeable
    var resizeStopCB = function (event, ui) {
      assistive_teleop.mjpeg.setParam('height', Math.round(ui.size.height));
      assistive_teleop.mjpeg.setParam('width', Math.round(ui.size.width));
      assistive_teleop.mjpeg.update()
    };
    $('#'+assistive_teleop.mjpeg.divId).resizable({aspectRatio:true,
                                         alsoResize:'#'+assistive_teleop.mjpeg.imageId,
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
