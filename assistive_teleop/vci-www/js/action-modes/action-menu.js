var RFH = (function (module) {
    module.ActionMenu = function (options) {
        "use strict";
        var self = this;
        self.div = $('#'+ options.divId);
        var ros = options.ros;
        self.tasks = {};
        self.activeAction = null;
        self.defaultActionName = null;

        var statePublisher = new ROSLIB.Topic({
            ros: ros,
            name: '/web_teleop/current_mode',
            messageType: 'std_msgs/String',
            latch: true
        });
        statePublisher.advertise();

        self.addAction = function (taskObject) {
            self.tasks[taskObject.name] = taskObject;
            if (taskObject.showButton) {
                // if (taskObject.buttonText) {
                var checkbox = document.createElement('input');
                checkbox.type = "checkbox";
                checkbox.id = taskObject.buttonText;
                var label = document.createElement('label');
                label.htmlFor = taskObject.buttonText;
                self.div.append(checkbox, label);
                $('#'+taskObject.buttonText).button({label:taskObject.buttonText.replace('_',' ')});
                $('label[for="'+taskObject.buttonText+'"]').addClass(taskObject.name + ' menu-item').prop('title', taskObject.toolTipText);
                $('#'+taskObject.buttonText).on('click.rfh', 
                    function(event){
                        self.buttonCB(taskObject);
                    });
            }
        };

        self.buttonCB = function (taskObject) {
            var newAction;
            if (taskObject === self.activeAction) {
                newAction = self.defaultActionName;
            } else {
                newAction = taskObject.name;
            }
            self.stopActiveAction();
            self.startAction(newAction);
        };

        self.startAction = function (taskName) {
            if (self.activeAction !== null) {
                self.stopActiveAction();
            }
            var taskObject = self.tasks[taskName] || self.tasks[self.defaultActionName];
            taskObject.start();
            if (taskObject.buttonText) {
                $('#'+taskObject.buttonText).prop('checked', true).button('refresh');
            }
            self.activeAction = taskObject;
            statePublisher.publish({'data':taskObject.name});
        };

        self.stopActiveAction = function () {
            if (self.activeAction === null) {
                return;
            }
            var taskObject = self.activeAction;
            // Stop currently running task
            taskObject.stop();
            if (taskObject.buttonText) {
                $('#'+taskObject.buttonText).prop('checked', false).button('refresh');
            }
            self.activeAction = null;
        };

        self.removeAction = function (taskObject) {
            self.stopTast(taskObject);
            $('#'+taskObject.buttonText).off('click.rfh');
            self.div.removeChild('#'+taskObject.buttonText);
            self.tasks.pop(self.tasks.indexOf(taskObject));
        };
    };

    module.initActionMenu = function (divId) {
        RFH.taskMenu = new RFH.ActionMenu({divId: divId,
            ros: RFH.ros});
        RFH.taskMenu.addAction(new RFH.Look({ros: RFH.ros, 
            div: 'video-main',
            head: RFH.pr2.head,
            camera: RFH.mjpeg.cameraModel}));
        RFH.taskMenu.defaultActionName = 'lookingAction';

        RFH.taskMenu.addAction(new RFH.Torso({containerDiv: 'video-main',
            sliderDiv: 'torsoSlider',
            torso: RFH.pr2.torso}));

        RFH.taskMenu.addAction(new RFH.CartesianEEControl({arm: RFH.pr2.l_arm_cart,
            ros: RFH.ros,
            div: 'video-main',
            gripper: RFH.pr2.l_gripper,
            tfClient: RFH.tfClient,
            eeDisplay: RFH.leftEEDisplay,
            camera: RFH.mjpeg.cameraModel}));

        RFH.taskMenu.addAction(new RFH.CartesianEEControl({arm: RFH.pr2.r_arm_cart,
            ros: RFH.ros,
            div: 'video-main',
            gripper: RFH.pr2.r_gripper,
            tfClient: RFH.tfClient,
            eeDisplay: RFH.rightEEDisplay,
            camera: RFH.mjpeg.cameraModel}));

        RFH.taskMenu.addAction(new RFH.Drive({ros: RFH.ros, 
            tfClient: RFH.tfClient,
            camera: RFH.mjpeg.cameraModel,
            head: RFH.pr2.head,
            left_arm: RFH.pr2.l_arm_cart,
            right_arm: RFH.pr2.r_arm_cart,
            base: RFH.pr2.base,
            forwardOnly: false}));
        RFH.taskMenu.addAction(new RFH.GetClickedPose({ros:RFH.ros,
            camera: RFH.mjpeg.cameraModel}));
        // Start looking task by default
        $('#'+RFH.taskMenu.tasks.lookingAction.buttonText).click();
    };
    return module;
})(RFH || {});
