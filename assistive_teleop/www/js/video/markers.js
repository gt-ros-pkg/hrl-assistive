// Create the main viewer.
assistive_teleop.viewer = new ROS3D.Viewer({
    divID : '3d-viewer',
    width : 1280,
    height : 1050,
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

// Setup the marker client.
assistive_teleop.markerClient = new ROS3D.MarkerClient({
    ros : assistive_teleop.ros,
    tfClient : assistive_teleop.tfClient,
    topic : '/visualization_marker',
    rootObject : viewer.scene
});
