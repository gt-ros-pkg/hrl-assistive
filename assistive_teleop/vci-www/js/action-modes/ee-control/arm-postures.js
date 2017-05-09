var RFH = (function (module) {
    'use strict';
    module.ArmPostureUtility = function (options) {
        var ros = options.ros;
        var arm = options.arm;
        var eeDisplay = options.eeDisplay;

        var asidePosture = arm.side[0] == 'r' ? [-1.8, 1.25, -1.9, -2.0, 3.5, -1.5, 0] : [1.8,  1.25, 1.9, -2.0, 2.8, -1.5, 0];
        var manipPosture = arm.side[0] == 'r' ? [-0.8, 0.0, -1.57, -1.9, 3.0, -1.0, -1.57] : [0.8, 0.0, 1.5, -1.9, 3.0, -1.0, 1.57];
        
        var postures = {'aside': asidePosture,
                        'manip': manipPosture};

        var goToPosture = function (postureName) {
            var angles = postures[postureName];
            eeDisplay.hide();
            eeDisplay.disableGoal();
            arm.sendJointAngleGoal(angles);
        };

        $('#controls div.posture.aside.'+arm.side[0]+'-arm-ctrl').button().on('click.rfh', function () {goToPosture('aside');}).prop('title', "Move this hand down and to the side of the robot.");
        $('#controls div.posture.manip.'+arm.side[0]+'-arm-ctrl').button().on('click.rfh', function () {goToPosture('manip');}).prop('title', "Move this hand up in front of the robot with the elbow raised.");

    };
    return module;
})(RFH || {});
