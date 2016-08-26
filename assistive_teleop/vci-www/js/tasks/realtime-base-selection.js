var RFH = (function (module) {
    module.RealtimeBaseSelection = function (options){
        'use strict';
        var self = this;
        var ros = options.ros;
        var arm = options.arm;
        var camera = options.camera;
        var $image = $('#mjpeg-image');

        var pixel23d = new RFH.Pixel23DClient({
            ros: ros,
            cameraInfoTopic: camera.infoTopic
        });

        var poseCB = function(pose_msg) {
            console.log("Original Pose msg: ", pose_msg);
            $image.removeClass('cursor-wait');
        };

        var clickCB = function(event, ui) {
            var pt = RFH.positionInElement(event);
            var px = (pt[0]/event.target.clientWidth);
            var py = (pt[1]/event.target.clientHeight);
            try {
                pixel23d.callRelativeScale(px, py, poseCB);
                $image.addClass('cursor-wait');
            } catch(err) {
                log(err);
            }
        };

        // Load msg details
        ros.getMsgDetails('trajectory_msgs/JointTrajectory');
        ros.getMsgDetails('trajectory_msgs/JointTrajectoryPoint');
        ros.getMsgDetails('assistive_teleop/HeadSweepGoal');

        // Create head sweep action client
        var sweepActionClient = new ROSLIB.ActionClient({
            ros: ros,
            serverName:'/head_sweep_action',
            actionName:'assistive_teleop/HeadSweepAction'
        });

        // Defin sweep trajectory, build at call time so the msg details have time to import
        var getSweepTrajectory = function () {
            var sweep_trajectory = ros.composeMsg('trajectory_msgs/JointTrajectory');
            var start_point = ros.composeMsg('trajectory_msgs/JointTrajectoryPoint');
            var end_point = ros.composeMsg('trajectory_msgs/JointTrajectoryPoint');
            start_point.positions = [1.1, 0.73];
            end_point.positions = [-1.1, 0.73];
            end_point.time_from_start.secs = 5;
            end_point.time_from_start.nsecs = 0;
            sweep_trajectory.points = [start_point, end_point];
            sweep_trajectory.joint_names = ['head_pan_joint', 'head_tilt_joint'];
            return sweep_trajectory;
        };

        var scanResultCB = function (result) {
            console.log(result);
        };

        var approachButtonCB = function (event) {
            var traj = getSweepTrajectory();
            var goalMsg = ros.composeMsg('assistive_teleop/HeadSweepGoal');
            goalMsg.sweep_trajectory = traj;
            var goal = new ROSLIB.Goal({
                actionClient: sweepActionClient,
                goalMessage:goalMsg
            });
            goal.on('result', scanResultCB);
            goal.send();
            console.log("Call head sweep for " + arm + " arm");
        };

        // Set up button with callback
        $('#controls > div.rtbs.'+arm).button().on('click.rfh', approachButtonCB);
    };
    return module;

})(RFH || {});
