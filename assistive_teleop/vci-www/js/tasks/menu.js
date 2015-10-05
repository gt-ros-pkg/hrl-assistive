RFH.TaskMenu = function (divId) {
    "use strict";
    var self = this;
    self.div = $('#'+divId);
    self.tasks = {};
    self.activeTask = null;

    self.addTask = function (taskObject) {
        self.tasks[taskObject.name] = taskObject;
        if (taskObject.buttonText) {
            var checkbox = document.createElement('input');
            checkbox.type = "checkbox";
            checkbox.id = taskObject.buttonText;
            var label = document.createElement('label');
            label.htmlFor = taskObject.buttonText;
            self.div.append(checkbox, label);
            $('#'+taskObject.buttonText).button({label:taskObject.buttonText.replace('_',' ')});
            $('label[for="'+taskObject.buttonText+'"]').addClass(taskObject.buttonClass + ' menu-item');
            $('#'+taskObject.buttonText).on('click.rfh', 
                                            function(event){
                                                self.buttonCB(taskObject);
                                            });
        }
    };

    self.buttonCB = function (taskObject) {
        var newTask;
        if (taskObject === self.activeTask) {
            newTask = self.defaultTask;
        } else {
            newTask = taskObject.name;
        }
        self.stopActiveTask();
        self.startTask(newTask);
    };

    self.startTask = function (taskName) {
        if (self.activeTask !== null) {
            self.stopActiveTask();
        }
        var taskObject = self.tasks[taskName];
        taskObject.start();
        if (taskObject.buttonText) {
            $('#'+taskObject.buttonText).prop('checked', true).button('refresh');
        }
        self.activeTask = taskObject;
    };
    
    self.stopActiveTask = function () {
        var taskObject = self.activeTask;
        // Stop currently running task
        taskObject.stop();
        if (taskObject.buttonText) {
            $('#'+taskObject.buttonText).prop('checked', false).button('refresh');
        }
        self.activeTask = null;
    };

    self.removeTask = function (taskObject) {
        self.stopTast(taskObject);
        $('#'+taskObject.buttonText).off('click.rfh');
        self.div.removeChild('#'+taskObject.buttonText);
        self.tasks.pop(self.tasks.indexOf(taskObject));
    };
};

RFH.initTaskMenu = function (divId) {
    RFH.taskMenu = new RFH.TaskMenu( divId );
    RFH.taskMenu.addTask(new RFH.Look({ros: RFH.ros, 
                                       div: 'video-main',
                                       head: RFH.pr2.head,
                                       camera: RFH.mjpeg.cameraModel}));
    RFH.taskMenu.defaultTask = 'lookingTask';

    RFH.taskMenu.addTask(new RFH.CartesianEEControl({arm: RFH.pr2.l_arm_cart,
                                                     div: 'video-main',
                                                     gripper: RFH.pr2.l_gripper,
                                                     tfClient: RFH.tfClient,
                                                     camera: RFH.mjpeg.cameraModel}));

    RFH.taskMenu.addTask(new RFH.CartesianEEControl({arm: RFH.pr2.r_arm_cart,
                                                     div: 'video-main',
                                                     gripper: RFH.pr2.r_gripper,
                                                     tfClient: RFH.tfClient,
                                                     camera: RFH.mjpeg.cameraModel}));

    RFH.taskMenu.addTask(new RFH.Torso({containerDiv: 'video-main',
                                        sliderDiv: 'torsoSlider',
                                        torso: RFH.pr2.torso}));

    RFH.taskMenu.addTask(new RFH.Drive({ros: RFH.ros, 
                                       targetDiv: 'mjpeg-image',
                                       camera: RFH.mjpeg.cameraModel,
                                       head: RFH.pr2.head,
                                       base: RFH.pr2.base}));
    RFH.taskMenu.addTask(new RFH.MoveObject({ros:RFH.ros}));
    RFH.taskMenu.addTask(new RFH.IdLocation({ros:RFH.ros}));
                                    
    // Start looking task by default
    RFH.taskMenu.tasks.lookingTask.start();
    RFH.taskMenu.activeTask = RFH.taskMenu.tasks.lookingTask;
};
