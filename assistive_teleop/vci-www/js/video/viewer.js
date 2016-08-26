var RFH = (function (module) {
    module.initViewer = function (divID) {
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

        RFH.viewer.renderer.context.canvas.id = "viewer-canvas";
        RFH.viewer.renderer.setClearColor(0x000000,0); // Set transparent background
        RFH.viewer.camera = new THREE.PerspectiveCamera(53.8, 1.5632, 0.2, 30); // Kinect2 Params
        RFH.viewer.camera.projectionMatrix.elements = [1.1129, 0,0,0,0,2.009,0,0,0,0,-1.001, -1, 0,0,-0.0400266,0];
        RFH.viewer.camera.up = new THREE.Vector3(0,0,1);

        var updateCamera = function(transform) {
            if (RFH.kinectHeadPointCloud.locked) { return; } // Don't interrupt a 3d Peek view
            RFH.viewer.camera.position.set(transform.translation.x,
                                           transform.translation.y,
                                           transform.translation.z);
            RFH.viewer.camera.quaternion.set(transform.rotation.x,
                                             transform.rotation.y,
                                             transform.rotation.z,
                                             transform.rotation.w);
            RFH.viewer.camera.updateMatrix();
            RFH.viewer.camera.updateMatrixWorld();
            var out = RFH.viewer.camera.localToWorld(new THREE.Vector3(1,0,0));
            RFH.viewer.camera.lookAt(out);
        };

        var resizeRenderer = function (event) {
            var w = $(RFH.viewer.renderer.context.canvas.parentElement).width();
            var h = $(RFH.viewer.renderer.context.canvas.parentElement).height();
            RFH.viewer.renderer.setSize(w, h);
            RFH.viewer.renderer.render( RFH.viewer.scene, RFH.viewer.camera);
        };
        $(window).on('resize.rfh', resizeRenderer);
        resizeRenderer();

        RFH.tfClient.subscribe('head_mount_kinect_rgb_link', updateCamera);
    };
    return module;
})(RFH || {});
