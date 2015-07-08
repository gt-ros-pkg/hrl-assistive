RFH.EERotation = function (options) {
    'use strict';
    var self = this;
    self.div = options.div;
    self.arm = options.arm;
    self.tfClient = options.tfClient;
    self.eeFrame = options.eeFrame;

    self.rotSlider = $('#'+self.div+' .wrist-rot-slider').slider({min:-Math.PI,
                                                 max:Math.PI,
                                                 value:0.0,
                                                 step:Math.PI/36}).on('slidestop.rfh',
                                                    function(event, ui) {
                                                        self.sendRotation(); 
                                                    });;
    self.latSlider = $('#'+self.div+' .wrist-lat-slider').slider({min:-Math.PI/2,
                                                 max:Math.PI/2,
                                                 step:Math.PI/36,
                                                 value:0.0}).on('slidestop.rfh',
                                                    function(event, ui) {
                                                        self.sendRotation();
                                                    });
    self.lonSlider = $('#'+self.div+' .wrist-lon-slider').slider({min:0.0,
                                                 max:Math.PI/2,
                                                 step:Math.PI/18,
                                                 value: Math.PI/4, 
                                                 orientation: 'vertical'}).on('slidestop.rfh',
                                                    function(event, ui) {
                                                        self.sendRotation();
                                                    });
    //$('#'+self.div+' .wrist-rot-slider').CircularSlider({value:0.0});
    //$('#'+self.div+' .wrist-lat-slider').CircularSlider({min:0.0,
    //                                                    max:180,
    //                                                    value:90,
    //                                                    shape:'Half Circle Bottom'});
    //$('#'+self.div+' .wrist-lon-slider').CircularSlider({min:0.0,
    //                                                   max: 90,
    //                                                   value: 45,
    //                                                   shape:'Half Circle Right'});

    self.sendRotation = function (rx, ry, rz) {
        var rx = rx || 0;
        var ry = ry || 0;
        var rz = rz || 0;
        var rx = self.rotSlider.slider('option','value') + rx;
        var ry = self.lonSlider.slider('option','value') + ry;
        var rz = self.latSlider.slider('option','value') + rz;
        var euler = new THREE.Euler(rx, ry, rz, 'ZYX'); //Originally working
//        var euler = new THREE.Euler(rx, ry, rz, 'XYZ');
        var q = new THREE.Quaternion().setFromEuler(euler);
        var quat = new ROSLIB.Quaternion({x:q.x, y:q.y, z:q.z, w:q.w});
        console.log("Sending RPY: "+euler.x+", "+euler.y+", "+euler.z);
//        console.log("Sending Quat: "+q.x+", "+q.y+", "+q.z+", "+q.w);
        self.arm.sendGoal({orientation: quat});
    };

    self.updateState = function (eeTF) {
        var eeQuat = new THREE.Quaternion(eeTF.rotation.x, 
                                          eeTF.rotation.y, 
                                          eeTF.rotation.z,
                                          eeTF.rotation.w);
        var eeEuler = new THREE.Euler().setFromQuaternion(eeQuat, 'ZYX'); 
//        console.log("RPY: "+eeEuler.x+", "+eeEuler.y+", "+eeEuler.z);
        self.rotSlider.slider('option','value', eeEuler.x);
        self.lonSlider.slider('option','value', eeEuler.y);
        self.latSlider.slider('option','value', eeEuler.z);
    }
    self.tfClient.subscribe(self.eeFrame, self.updateState);

    self.setMode = function (mode) {
        switch (mode) {
            case "table":
                self.rotSlider.slider('option','min', -Math.PI);
                self.rotSlider.slider('option','max', Math.PI);
                self.latSlider.slider('option','min', -Math.PI);
                self.latSlider.slider('option','max', Math.PI);
                self.lonSlider.slider('option','min', -Math.PI/2);
                self.lonSlider.slider('option','max', 0.0);
                break;
            case "wall":
                self.rotSlider.slider('option','min', -Math.PI);
                self.rotSlider.slider('option','max', Math.PI);
                self.latSlider.slider('option','min', -Math.PI);
                self.latSlider.slider('option','max', Math.PI);
                self.lonSlider.slider('option','min', -Math.PI);
                self.lonSlider.slider('option','max', Math.PI);
                break;
            case "free":
                self.rotSlider.slider('option','min', -Math.PI);
                self.rotSlider.slider('option','max', Math.PI);
                self.latSlider.slider('option','min', -Math.PI);
                self.latSlider.slider('option','max', Math.PI);
                self.lonSlider.slider('option','min', -Math.PI);
                self.lonSlider.slider('option','max', Math.PI);
                break;
        };

    };

}
