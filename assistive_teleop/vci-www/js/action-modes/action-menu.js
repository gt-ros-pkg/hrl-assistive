var RFH = (function (module) {
    module.ActionMenu = function (options) {
        "use strict";
        var self = this;
        var $div = $('#'+ options.divId);
        var ros = options.ros;
        self.actions = {};
        self.activeAction = null;
        self.defaultActionName = null;

        var statePublisher = new ROSLIB.Topic({
            ros: ros,
            name: '/web_teleop/current_mode',
            messageType: 'std_msgs/String',
            latch: true
        });
        statePublisher.advertise();

        self.addAction = function (actionObject) {
            self.actions[actionObject.name] = actionObject;
            if (actionObject.showButton) {
                actionObject.$button = $('#'+actionObject.buttonID).button();
                $('label[for="'+actionObject.buttonID+'"]').prop('title', actionObject.toolTipText);
                actionObject.$button.on('click.rfh', function(event){self.buttonCB(actionObject); });
            }
        };

        self.buttonCB = function (actionObject) {
            var newAction;
            if (actionObject === self.activeAction) {
                newAction = self.defaultActionName;
            } else {
                newAction = actionObject.name;
            }
            self.stopActiveAction();
            self.startAction(newAction);
        };

        self.startAction = function (actionName) {
            if (self.activeAction !== null) {
                self.stopActiveAction();
            }
            var actionObject = self.actions[actionName] || self.actions[self.defaultActionName];
            actionObject.start();
            if (actionObject.showButton) {
                actionObject.$button.prop('checked', true).button('refresh');
            }
            self.activeAction = actionObject;
            statePublisher.publish({'data':actionObject.name});
        };

        self.stopActiveAction = function () {
            if (self.activeAction === null) {
                return;
            }
            var actionObject = self.activeAction;
            // Stop currently running action
            actionObject.stop();
            if (actionObject.showButton) {
                actionObject.$button.prop('checked', false).button('refresh');
            }
            self.activeAction = null;
        };

        self.removeAction = function (actionObject) {
            self.stopTast(actionObject);
            actionObject.$button.off('click.rfh');
            $div.removeChild(actionObject.$button);
            self.actions.pop(self.actions.indexOf(actionObject));
        };
    };

    module.initActionMenu = function (divId) {
        RFH.actionMenu = new RFH.ActionMenu({divId: divId,
            ros: RFH.ros});
        RFH.actionMenu.addAction(new RFH.Look({ros: RFH.ros, 
            div: 'video-main',
            head: RFH.pr2.head,
            camera: RFH.mjpeg.cameraModel}));
        RFH.actionMenu.defaultActionName = 'lookingAction';

        RFH.actionMenu.addAction(new RFH.Torso({containerDiv: 'video-main',
            sliderDiv: 'torso-slider',
            torso: RFH.pr2.torso}));

        RFH.actionMenu.addAction(new RFH.CartesianEEControl({arm: RFH.pr2.l_arm_cart,
            ros: RFH.ros,
            div: 'video-main',
            gripper: RFH.pr2.l_gripper,
            tfClient: RFH.tfClient,
            eeDisplay: RFH.leftEEDisplay,
            skinDisplay: RFH.skinContactDisplay,
            camera: RFH.mjpeg.cameraModel}));

        RFH.actionMenu.addAction(new RFH.CartesianEEControl({arm: RFH.pr2.r_arm_cart,
            ros: RFH.ros,
            div: 'video-main',
            gripper: RFH.pr2.r_gripper,
            tfClient: RFH.tfClient,
            eeDisplay: RFH.rightEEDisplay,
            skinDisplay: RFH.skinContactDisplay,
            camera: RFH.mjpeg.cameraModel}));

        RFH.actionMenu.addAction(new RFH.Drive({ros: RFH.ros, 
            tfClient: RFH.tfClient,
            camera: RFH.mjpeg.cameraModel,
            head: RFH.pr2.head,
            left_arm: RFH.pr2.l_arm_cart,
            right_arm: RFH.pr2.r_arm_cart,
            base: RFH.pr2.base,
            forwardOnly: false}));
        RFH.actionMenu.addAction(new RFH.GetClickedPose({ros:RFH.ros,
            camera: RFH.mjpeg.cameraModel}));
        // Start looking action by default
        RFH.actionMenu.actions.lookingAction.$button.click();
    };
    return module;
})(RFH || {});
