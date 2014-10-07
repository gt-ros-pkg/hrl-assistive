R4H.TaskMenu = function (divId) {
    "use strict";
    var self = this;
    self.divId = divId;
    self.tasks = [];

    self.addTask = function (taskObject, position) {
        var position = position !== undefined ? position : a.length;
        self.tasks.splice(position, 0, taskObject);
        $('#'+divId).append('<button id="'+taskObject.buttonText+'" class="menu-item">'+taskObject.buttonText+'</button>');
        $('#'+taskObject.buttonText).addClass(taskObject.buttonClass);
        $('#'+taskObject.buttonText).on('click.rfh', function(){self.startTask(taskObject)});
        //Add callbacks, etc.
    }

    self.startTask = function (taskObject) {
        $('#'+taskObject.buttonText).off('click.rfh').on('click.rfh', function(){self.stopTask(taskObject)});
        taskObject.start();
    }
    
    self.startTask = function (taskObject) {
        $('#'+taskObject.buttonText).off('click.rfh').on('click.rfh', function(){self.startTask(taskObject)});
        taskObject.stop();
    }

    self.removeTask = function (taskObject) {
        //Stop task, remove button, etc.
    }
}

R4H.initTaskMenu = function (divId) {
    R4H.taskMenu = new R4H.TaskMenu( divId );
    R4H.taskMenu.addTask(new assistive_teleop.Look({ros: assistive_teleop.ros, 
                                                    div: 'markers',
                                                    camera: assistive_teleop.mjpeg.cameraModel}));
}
