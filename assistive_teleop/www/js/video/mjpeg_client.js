
//Old camera parameters
//  'Right Arm': {topic: '/r_forearm_cam/image_color_rotated',
//                optgroup:'Default',
//                cameraInfo: '/r_forearm_cam/camera_info',
//                clickable:true,
//                rotated: true,
//                width:640,
//                height:480},
//  'Left Arm': {topic: '/l_forearm_cam/image_color_rotated',
//               optgroup:'Default',
//               cameraInfo: '/l_forearm_cam/camera_info',
//               clickable: false,
//               rotated: true,
//               width:640,
//               height:480},
//  'AR Tag': {topic:'/ar_servo/confirmation_rotated',
//             optgroup:'Special',
//             cameraInfo:'/r_forearm_cam/camera_info',
//             clickable:false,
//             rotated: true,
//             width:640,
//             height:480},
//  'Head Registration': {topic: '/head_registration/confirmation',
//                        optgroup:'Special',
//                        cameraInfo: 'head_mount_kinect/rgb/camera_info',
//                        clickable: true,
//                        rotated: false,
//                        width: 640, //1280,
//                        height:480}//1024}

RFH.MjpegClient = function (options) {
    var self = this;
    var options = options || {};
    self.imageTopic = options.imageTopic;
    self.divId = options.divId;
    self.server = "http://"+options.host+":"+options.port;
    self.activeParams = {'width': options.width,
                         'height': options.height,
                         'quality': options.quality || 80,
                         'topic': options.imageTopic}

    self.cameraModel = new RFH.ROSCameraModel({ros: options.ros,
                                                            infoTopic: options.infoTopic,
                                                            rotated: options.rotated || false});

    self.imageId = self.divId + "Image";
    $("#"+self.divId).append("<img id="+self.imageId+"></img>");

    self.update = function () {
        var srcStr = self.server+ "/stream"
        for (param in self.activeParams)
        {
            srcStr += "?" + param + '=' + self.activeParams[param]
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

    self.cameraModel.updateCameraInfo();
    self.update();

};

RFH.ROSCameraModel = function (options) {
    var self = this;
    var options = options || {};
    self.ros = options.ros;
    self.infoTopic = options.infoTopic;
    self.rotated = options.rotated || false;
    self.width = options.width || 640;
    self.height = options.height || 480;
    self.frame_id = '';
    self.distortion_model = '';
    self.D = [0,0,0,0,0];
    self.K = [[1,0,0],[0,1,0],[0,0,1]];
    self.R = [[1,0,0],[0,1,0],[0,0,1]];
    self.KR = [[1,0,0],[0,1,0],[0,0,1]];
    self.KR_inv = [[1,0,0],[0,1,0],[0,0,1]];
    self.P = [[1,0,0,0],[0,1,0,0],[0,0,1,0]];
    self.has_data = false;

    self.ros.getMsgDetails('sensor_msgs/CameraInfo');

    self.cameraInfoSubscriber = new ROSLIB.Topic({
        ros: self.ros,     
        name: self.infoTopic,
        messageType: 'sensor_msgs/CameraInfo'});

    self.msgCB = function (infoMsg) {
        console.log("Updating camera model from "+self.infoTopic)
        self.frame_id = infoMsg.header.frame_id;
        if (self.rotated) { self.frame_id += '_rotated'; }
        self.width = infoMsg.width;
        self.height = infoMsg.height;
        self.distortion_model = infoMsg.distortion_model;
        self.D = infoMsg.D;
        self.K = [infoMsg.K.slice(0,3),
                  infoMsg.K.slice(3,6), 
                  infoMsg.K.slice(6,9)];

        self.R = [infoMsg.R.slice(0,3),
                  infoMsg.R.slice(3,6), 
                  infoMsg.R.slice(6,9)];

        self.P = [infoMsg.P.slice(0,4),
                  infoMsg.P.slice(4,8), 
                  infoMsg.P.slice(8,12)];

        // Not collecting data on binning, ROI
        self.KR = numeric.dot(self.K, self.R);
        self.KR_inv = numeric.inv(self.KR);
        self.has_data = true;
        self.cameraInfoSubscriber.unsubscribe(); // Close subscriber to save bandwidth
        }
    
    self.updateCameraInfo = function () {
        // Re-subscribe to get new parameters
        self.cameraInfoSubscriber.subscribe(self.msgCB);
    }

    // back-project a pixel some distance into the real world
    // Returns a geoemtry_msgs/PointStamped msg
    self.projectPixel = function (px, py, dist) { 
       if (!self.has_data) { console.error("Camera Model has not received data from "+self.infoTopic); };
       var d = dist !== undefined ? dist : 2; 
       var pixel_hom = [[px],[py],[1]]; //Pixel value in homogeneous coordinates
       var vec = numeric.dot(self.KR_inv, pixel_hom);
       vec = numeric.transpose(vec)[0];
       var mag = numeric.norm2(vec);
       return numeric.mul(d/mag, vec);
    }
};

var initMjpegCanvas = function (divId) {
    // Initialize the mjpeg client
    $('#'+divId).off('click'); //Disable click detection so clickable_element catches it
    var width = 0.8 * window.innerWidth;
    var height = 0.95 * window.innerHeight;
    $('#'+divId).css({'height':height, 'width':width});
    RFH.mjpeg = new RFH.MjpegClient({ros: RFH.ros,
                                     imageTopic: '/head_mount_kinect/rgb/image_color',
                                     infoTopic: '/head_mount_kinect/rgb/camera_info',
                                     divId: 'mjpegDiv',
                                     host: RFH.ROBOT,
                                     port: 8080,
                                     width: width,
                                     height: height,
                                     quality: 85});
};
