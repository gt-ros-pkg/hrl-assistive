RFH.TaskMenu = function (divId) {
    "use strict";
    var self = this;
    self.divId = divId;
    self.tasks = [];

    self.addTask = function (taskObject, position) {
        var position = position !== undefined ? position : self.tasks.length;
        self.tasks.splice(position, 0, taskObject);
        var checkbox = document.createElement('input');
        checkbox.type = "checkbox";
        checkbox.id = taskObject.buttonText;
        var label = document.createElement('label');
        label.htmlFor = taskObject.buttonText;
        label.appendChild(document.createTextNode(taskObject.buttonText));
        $('#'+divId).append(checkbox, label);
        $('#'+taskObject.buttonText).button({label:taskObject.buttonText});
        $('#'+taskObject.buttonText).addClass(taskObject.buttonClass);
        $('#'+taskObject.buttonText).on('click.rfh', function(event){ self.startTask(taskObject) });
    };

    self.startTask = function (taskObject) {
        $('#'+taskObject.buttonText).off('click.rfh').on('click.rfh', function(){self.stopTask(taskObject)});
        taskObject.start();
    };
    
    self.stopTask = function (taskObject) {
        taskObject.stop();
        $('#'+taskObject.buttonText).off('click.rfh').on('click.rfh', function(){self.startTask(taskObject)});
    };

    self.removeTask = function (taskObject) {
        //Stop task, remove button, etc.
    };
}

RFH.initTaskMenu = function (divId) {
    RFH.taskMenu = new RFH.TaskMenu( divId );
    RFH.taskMenu.addTask(new RFH.Look({ros: RFH.ros, 
                                       div: 'markers',
                                       head: RFH.pr2.head,
                                       camera: RFH.mjpeg.cameraModel}));
    RFH.taskMenu.addTask(new RFH.Drive({ros: RFH.ros, 
                                       div: 'markers',
                                       camera: RFH.mjpeg.cameraModel,
                                       head: RFH.pr2.head,
                                       torso: RFH.pr2.torso}));
}
