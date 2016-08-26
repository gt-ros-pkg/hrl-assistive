var RFH = (function (module) {
    module.PointCloudView = function(options){
        'use strict';
        var self = this;
        var ros = options.ros;
        var topic = options.topic;
        var tfClient = options.tfClient;
        var maxPoints = options.maxPoints || 500;
        self.locked = false;

        var material = new THREE.PointsMaterial({size: 6.0,
            vertexColors: THREE.VertexColors,
            opacity:0.87,
            depthTest:1,
            transparent: true,
            sizeAttenuation:0
        });

        self.pcGeom = new THREE.BufferGeometry();

        var positions = new Float32Array( maxPoints * 3 );
        var colors = new Float32Array( maxPoints * 3 );
        var x = 0;
        var y = 0;
        var z = 0;
        var color = new THREE.Color();
        for (var i = 0; i<positions.length; i += 3) {
            positions[i] = x;
            positions[i+1] = y;
            positions[i+2] = z;
            color.setRGB(0.2,0.2,0.2);
            colors[i] = color.r;
            colors[i+1] = color.g;
            colors[i+2] = color.b;
        }

        self.pcGeom.addAttribute('position', new THREE.BufferAttribute(positions, 3));
        self.pcGeom.addAttribute('color', new THREE.BufferAttribute(colors, 3));

        self.pointCloud = new THREE.Points(self.pcGeom, material);
        self.pointCloud.userData.interactive = false;
        self.pointCloud.frameSubscribed = false;
        self.pointCloud.frameReceived = false;
        self.pointCloud.matrixAutoUpdate = false;
        self.pointCloud.visible = false;
        RFH.viewer.scene.add(self.pointCloud);

        self.setVisible = function (bool) {
            self.pointCloud.visible = bool;
            RFH.viewer.renderer.render(RFH.viewer.scene, RFH.viewer.camera);
        };

        var updatePCFrame = function(tf) {
            self.pointCloud.frameReceived = true;
            self.pointCloud.position.set(tf.translation.x, tf.translation.y, tf.translation.z);
            self.pointCloud.quaternion.set(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w);
            self.pointCloud.updateMatrix();
        };

        var pointCloudDisplayCB = function (message) {
            if (!self.pointCloud.frameSubscribed) {
                tfClient.subscribe(message.header.frame_id, updatePCFrame);
                self.pointCloud.frameSubscribed = true;
            }
            var n = message.height*message.width;
            var buffer;
            if(message.data.buffer){
                buffer = message.data.buffer.buffer;
            }else{
                buffer = Uint8Array.from(decode64(message.data)).buffer;
            }
            var dv = new DataView(buffer);
            var positions = self.pointCloud.geometry.getAttribute('position');
            var colors = self.pointCloud.geometry.getAttribute('color');
            var color = new THREE.Color();
            for(var i=0;i<maxPoints;i++){
                if (i<n){
                    var pt = read_point(message, i, dv);
                    positions.setXYZ(i, pt.x, pt.y, pt.y);
                    color.setHex( pt.rgb );
                    colors.setXYZ(i, color.r, color.g, color.b);
                } else {
                    positions.setXYZ(i, 0, 0, -10); // Hide non-current/unneeded points behind the camera
                }
            }
            positions.needsUpdate = true;
            colors.needsUpdate = true;
        };

        var pointCloudSub = new ROSLIB.Topic({
            ros: ros,
            name: topic,
            messageType: 'sensor_msgs/PointCloud2',
        });
        pointCloudSub.subscribe(pointCloudDisplayCB);
    };
    return module;

})(RFH || {});
