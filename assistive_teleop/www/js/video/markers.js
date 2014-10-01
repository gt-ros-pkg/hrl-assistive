var initMarkerDisplay = function (divID) {
    // Create the main viewer.
    assistive_teleop.viewer = new ROS3D.Viewer({
        divID : divID,
        width : 1280,
        height : 1024,
        antialias : true
    });

    // Setup a client to listen to TFs.
    assistive_teleop.tfClient = new ROSLIB.TFClient({
        ros : assistive_teleop.ros,
        angularThres : 0.01,
        transThres : 0.01,
        rate : 10.0,
        fixedFrame : '/my_frame'
    });

    // Setup the marker client.
    assistive_teleop.markerClient = new ROS3D.MarkerClient({
        ros : assistive_teleop.ros,
        tfClient : assistive_teleop.tfClient,
        topic : '/visualization_marker',
        rootObject : assistive_teleop.viewer.scene
    });
}
