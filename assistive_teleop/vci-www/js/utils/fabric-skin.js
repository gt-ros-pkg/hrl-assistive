var RFH = (function (module) {
    module.FabricSkin = function (options) {
        'use strict';
        var self = this;
        var ros = options.ros;
        var ns = options.ns;
        self.baseLink = null;
        self.taxelTransforms = [];

        var linkServiceClient = new ROSLIB.Service({
            ros: ros,
            name: ns + '/taxels/srv/link_name',   
            serviceType: 'm3skin_ros/None_String'
        });
        var linkReq = new ROSLIB.ServiceRequest({});
        linkServiceClient.callService(linkReq, function (resp) {
            self.baseLink = resp.data;});


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
        frameServiceClient.callService(framesReq, function (resp) {
                                                        self.taxelTransforms = resp.data;
                                                    });

        /* Zero sensor readings in calibration node */
        var zeroPub = new ROSLIB.Topic({
            ros: ros,
            name: ns+'/zero_sensor',
            messageType: 'std_msgs/Empty'
        });
        zeroPub.advertise();
        self.zeroSensor = function () {
            zeroPub.publish();
        };
     
        /* Enable Sensor data publishing */
        var enablePub = new ROSLIB.Topic({
            ros: ros,
            name: ns+'/enable_sensor',
            messageType: 'std_msgs/Empty'
        });
        enablePub.advertise();
        self.enableSensor = function () {
            enablePub.publish();
        };

        /* Disable Sensor data publishing */
        var disablePub = new ROSLIB.Topic({
            ros: ros,
            name: ns+'/disable_sensor',
            messageType: 'std_msgs/Empty'
        });
        disablePub.advertise();
        self.disableSensor = function () {
            disablePub.publish();
        };
        
        /* Process force data from sensor */
        self.forceCBArray = [];
        var forcesCB = function (taxelArrayMsg) {
            for (var i=0; i<self.forceCBArray.length; i+=1) {
                self.forceCBArray[i](taxelArrayMsg);
            }
        };

        var forcesSub = new ROSLIB.Topic({
            ros: ros,
            name: ns+'/taxels/forces',
            messageType: 'hrl_haptic_manipulation_in_clutter_msgs/TaxelArray',
            throttle_rate: 1000
        });       
        forcesSub.subscribe(forcesCB);
    };

    module.initSkin = function () {
        var skins = {};    
        skins.left = {};
        skins.left.upperarm = new module.FabricSkin({ros: module.ros,
                                                     ns: '/pr2_fabric_left_upperarm_sensor'
                                                     });

        skins.left.forearm = new module.FabricSkin({ros: module.ros,
                                                     ns: '/pr2_fabric_left_forearm_sensor'
                                                     });

        skins.right = {};
        skins.right.upperarm = new module.FabricSkin({ros: module.ros,
                                                     ns: '/pr2_fabric_right_upperarm_sensor'
                                                     });

        skins.right.forearm = new module.FabricSkin({ros: module.ros,
                                                     ns: '/pr2_fabric_right_forearm_sensor'
                                                     });
        skins.base = new module.FabricSkin({ros: module.ros,
                                             ns: '/pr2_fabric_base_sensor'
                                             });
        module.skins = skins;
    };
    return module;

})(RFH || {});

