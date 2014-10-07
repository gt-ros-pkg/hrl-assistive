var initMarkerDisplay = function (divID) {
    // Create the main viewer.
    var width = 0.8 * window.innerWidth;
    var height = 0.95 * window.innerHeight;
    RFH.viewer = new ROS3D.Viewer({
        divID : divID,
        width : width,
        height : height,
        antialias : true
    });
    RFH.viewer.renderer.context.canvas.id = "clickable-canvas";

    // Setup a client to listen to TFs.
    RFH.tfClient = new ROSLIB.TFClient({
        ros : RFH.ros,
        angularThres : 0.01,
        transThres : 0.01,
        rate : 10.0,
        fixedFrame : '/base_link'
    });

    var updateCamera = function(transform) {
        RFH.viewer.camera.position.set(transform.translation.x,
                                                    transform.translation.y,
                                                    transform.translation.z);
        RFH.viewer.camera.quaternion = new THREE.Quaternion(transform.rotation.x,
                                                                         transform.rotation.y,
                                                                         transform.rotation.z,
                                                                         transform.rotation.w);
        RFH.viewer.camera.updateMatrix();
        RFH.viewer.camera.updateMatrixWorld();
        out = RFH.viewer.camera.localToWorld(new THREE.Vector3(1,0,0));
        RFH.viewer.camera.lookAt(out);
    }

    RFH.tfClient.subscribe('head_mount_kinect_rgb_link', updateCamera);

    // Setup the marker client.
    RFH.markerClient = new ROS3D.MarkerClient({
        ros : RFH.ros,
        tfClient : RFH.tfClient,
        topic : '/visualization_marker',
        rootObject : RFH.viewer.scene
    });
}
