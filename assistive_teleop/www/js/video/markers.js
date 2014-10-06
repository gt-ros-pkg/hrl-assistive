var initMarkerDisplay = function (divID) {
    // Create the main viewer.
    var width = 0.8 * window.innerWidth;
    var height = 0.95 * window.innerHeight;
    assistive_teleop.viewer = new ROS3D.Viewer({
        divID : divID,
        width : width,
        height : height,
        antialias : true
    });

    // Setup a client to listen to TFs.
    assistive_teleop.tfClient = new ROSLIB.TFClient({
        ros : assistive_teleop.ros,
        angularThres : 0.01,
        transThres : 0.01,
        rate : 10.0,
        fixedFrame : '/base_link'
    });

    var updateCamera = function(transform) {
        assistive_teleop.viewer.camera.position.set(transform.translation.x,
                                                    transform.translation.y,
                                                    transform.translation.z);
        assistive_teleop.viewer.camera.quaternion = new THREE.Quaternion(transform.rotation.x,
                                                                         transform.rotation.y,
                                                                         transform.rotation.z,
                                                                         transform.rotation.w);
        assistive_teleop.viewer.camera.updateMatrix();
        assistive_teleop.viewer.camera.updateMatrixWorld();
        out = assistive_teleop.viewer.camera.localToWorld(new THREE.Vector3(1,0,0));
        assistive_teleop.viewer.camera.lookAt(out);
    }

    assistive_teleop.tfClient.subscribe('head_mount_kinect_rgb_link', updateCamera);

    // Setup the marker client.
    assistive_teleop.markerClient = new ROS3D.MarkerClient({
        ros : assistive_teleop.ros,
        tfClient : assistive_teleop.tfClient,
        topic : '/visualization_marker',
        rootObject : assistive_teleop.viewer.scene
    });
}
