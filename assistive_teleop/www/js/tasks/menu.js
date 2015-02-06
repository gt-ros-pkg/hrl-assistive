RFH.TaskMenu = function (divId) {
    "use strict";
    var self = this;
    self.divId = divId;
    self.tasks = [];
    self.activeTask = null;
    self.waitTimer = null;

    self.addTask = function (taskObject, position) {
        var position = position !== undefined ? position : self.tasks.length;
        self.tasks.splice(position, 0, taskObject);
        var checkbox = document.createElement('input');
        checkbox.type = "checkbox";
        checkbox.id = taskObject.buttonText;
        var label = document.createElement('label');
        label.htmlFor = taskObject.buttonText;
        $('#'+divId).append(checkbox, label);
        $('#'+taskObject.buttonText).button({label:taskObject.buttonText.replace('_',' ')});
        $('label[for="'+taskObject.buttonText+'"]').addClass(taskObject.buttonClass + ' menu-item');
        $('#'+taskObject.buttonText).on('click.rfh', function(event){ self.startTask(taskObject) });
    };

    self.startTask = function (taskObject) {
        if (self.activeTask){
            $("#"+self.activeTask.buttonText).click();
            $('*').addClass('no-cursor');//TODO: find a better way to do this?
            self.waitForTaskStop();
        }
        $('#'+taskObject.buttonText).off('click.rfh').on('click.rfh', function(){self.stopTask(taskObject)});
        taskObject.start();
        self.activeTask = taskObject;
        $('*').removeClass('no-cursor');
    };
    
    self.stopTask = function (taskObject) {
        taskObject.stop();
        self.activeTask = null;
        $('#'+taskObject.buttonText).off('click.rfh').on('click.rfh', function(){self.startTask(taskObject)});
    };

    self.waitForTaskStop = function (task) {
        task =  (task === null) ? self.activeTask : task;
        if (self.activeTask) {
            self.waitTimer = setTimeout(function(){ self.waitForTaskStop(task) }, 100);
        } else {
            return true;
        }
    };

    self.removeTask = function (taskObject) {
        self.stopTast(taskObject);
        $('#'+taskObject.buttonText).off('click.rfh');
        $('#'+self.divId).removeChild('#'+taskObject.buttonText);
        self.tasks.pop(self.tasks.indexOf(taskObject));
    };
}

RFH.initTaskMenu = function (divId) {
    RFH.lookCtrl = new RFH.Look({ros: RFH.ros, 
                                div: 'mjpeg',
                                head: RFH.pr2.head,
                                camera: RFH.mjpeg.cameraModel});
    RFH.taskMenu = new RFH.TaskMenu( divId );
    RFH.taskMenu.addTask(new RFH.CartesianEEControl({arm: RFH.pr2.l_arm_cart,
                                                     div: 'mjpeg',
                                                     gripper: RFH.pr2.l_gripper,
                                                     tfClient: RFH.tfClient,
                                                     camera: RFH.mjpeg.cameraModel}));

    RFH.taskMenu.addTask(new RFH.CartesianEEControl({arm: RFH.pr2.r_arm_cart,
                                                     div: 'mjpeg',
                                                     gripper: RFH.pr2.r_gripper,
                                                     tfClient: RFH.tfClient,
                                                     camera: RFH.mjpeg.cameraModel}));
    RFH.taskMenu.addTask(new RFH.Torso({containerDiv: 'mjpeg',
                                        sliderDiv: 'torsoSlider',
                                        torso: RFH.pr2.torso}));
    RFH.taskMenu.addTask(new RFH.Drive({ros: RFH.ros, 
                                       targetDiv: 'mjpeg-image',
                                       camera: RFH.mjpeg.cameraModel,
                                       tfClient: RFH.tfClient,
                                       base: RFH.pr2.base}));
}
