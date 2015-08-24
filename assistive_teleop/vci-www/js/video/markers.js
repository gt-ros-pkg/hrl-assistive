var initMarkerDisplay = function (divID) {
    "use strict";
    // Create the main viewer.
    var width = $('#'+divID).width();
    var height = $('#'+divID).height();
    RFH.viewer = new ROS3D.Viewer({
        divID : divID,
        width : width,
        height : height,
        antialias : true
    });
    RFH.viewer.renderer.context.canvas.id = "clickable-canvas";
    RFH.viewer.camera.projectionMatrix.elements = [1.1129, 0,0,0,0,2.009,0,0,0,0,-1.001, -1, 0,0,-0.0400266,0];

    var updateCamera = function(transform) {
        RFH.viewer.camera.position.set(transform.translation.x,
                                       transform.translation.y,
                                       transform.translation.z);
        RFH.viewer.camera.quaternion.set(transform.rotation.x,
                                         transform.rotation.y,
                                         transform.rotation.z,
                                         transform.rotation.w);
        RFH.viewer.camera.updateMatrix()
        RFH.viewer.camera.updateMatrixWorld();
        var out = RFH.viewer.camera.localToWorld(new THREE.Vector3(1,0,0));
        RFH.viewer.camera.lookAt(out);
    }

    RFH.tfClient.subscribe('head_mount_kinect_rgb_link', updateCamera);

    // Setup the marker client.
//    RFH.markerClient = new ROS3D.MarkerClient({
//        ros : RFH.ros,
//        tfClient : RFH.tfClient,
//        topic : '/visualization_marker',
//        rootObject : RFH.viewer.scene
//    });

//    RFH.rMPCMarkerClient = new ROS3D.InteractiveMarkerClient({
//        ros: RFH.ros,
//        tfClient: RFH.tfClient,
//        topic: '/haptic_mpc/interactive_markers',//TODO: FIXME (Get correct name)
//        camera: RFH.viewer.renderer.camera,
//        rootObject: RFH.viewer.scene
//
//    });
}
