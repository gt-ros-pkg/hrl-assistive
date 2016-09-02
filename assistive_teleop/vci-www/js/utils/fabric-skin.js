var RFH = (function (module) {
    module.FabricSkin = function (options) {
        'use strict';
        var self = this;
        var ros = options.ros;
        var tfClient = options.tfClient;
        self.side = options.side;
        self.part = options.part;
        var ns = 'pr2_fabric_'+self.side[0]+'_'+self.part+'_sensor';
        self.baseLink = null;
        self.taxelPoints = [];

        var linkServiceClient = new ROSLIB.Service({
            ros: ros,
            name: ns + '/taxels/srv/link_name',   
            serviceType: 'm3skin_ros/None_String'
        });
        var linkReq = new ROSLIB.ServiceRequest({});
        linkServiceClient.callService(linkReq, function (resp) {self.baseLink = response.data;});


        var frameServiceClient = new ROSLIB.Service({
            ros: ros,
            name: ns + '/taxels/srv/local_coord_frames',   
            serviceType: 'm3skin_ros/None_TransformArray'
        });

        var setTaxelPoints = function (resp) {
            var point;
            for (var i=0; i<resp.data.length; i += 1) {
                self.taxelPoints[i] = resp.data[i].translation;
            }
        };
        var framesReq = new ROSLIB.ServiceRequest({});
        frameServiceClient.callService(framesReq, function (resp) {self.baseLink = response.data;});

        /* Zero sensor readings in calibration node */
        var zeroPub = new ROSLIB.Topic({
            ros: ros,
            topic: ns+'/zero_sensor',
            messageType: 'std_msgs/Empty'
        });
        zeroPub.advertise();
        self.zeroSensor = function () {
            zeroPub.publish();
        };
     
        /* Enable Sensor data publishing */
        var enablePub = new ROSLIB.Topic({
            ros: ros,
            topic: ns+'/enable_sensor',
            messageType: 'std_msgs/Empty'
        });
        enablePub.advertise();
        self.enableSensor = function () {
            enablePub.publish();
        };

        /* Disable Sensor data publishing */
        var disablePub = new ROSLIB.Topic({
            ros: ros,
            topic: ns+'/disable_sensor',
            messageType: 'std_msgs/Empty'
        });
        disablePub.advertise();
        self.disableSensor = function () {
            disablePub.publish();
        };
        
        /* Process force data from sensor */
        var forceCBArray = [];
        var forcesCB = function (taxelArrayMsg) {
            for (var i=0; i<forceCBArray.length; i += 1) {
                forceCBArray[i](taxelArrayMsg);
            }
        };

        var forcesSub = new ROSLIB.Topic({
            ros: ros,
            topic: ns+'/taxels/forces',
            messageType: 'hrl_haptic_manipulation_in_clutter_msgs/TaxelArray'
        });       
        forcesSub.subscribe(forcesCB);
    };

    module.initSkin = function () {
        var skins = {};    
        skins.left = {};
        skins.left.upperarm = new module.FabricSkin({ros: RFH.ros,
                                                     tfClient: RFH.tfClient, 
                                                     side: "left",
                                                     part: "upperarm"});

        skins.left.forearm = new module.FabricSkin({ros: RFH.ros,
                                                     tfClient: RFH.tfClient, 
                                                     side: "left",
                                                     part: "forearm"});

        skins.right = {};
        skins.right.upperarm = new module.FabricSkin({ros: RFH.ros,
                                                      tfClient: RFH.tfClient, 
                                                      side: "right",
                                                      part: "upperarm"});

        skins.right.upperarm = new module.FabricSkin({ros: RFH.ros,
                                                      tfClient: RFH.tfClient, 
                                                      side: "right",
                                                      part: "forearm"});
    };
    return module;

})(RFH || {});

