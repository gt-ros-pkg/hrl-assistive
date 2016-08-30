var RFH = (function (module) {
    module.TaskMenu = function (options) {
        "use strict";
        var self = this;
        self.div = $('#'+ options.divId);
        var ros = options.ros;
        self.domains = {};

        var startButtonCB = function (event) {
            // Call Domain Start
        };

        self.updateOptions = function () {
            
        };

    };

    module.initTaskMenu = function () {
        RFH.taskMenu = new RFH.TaskMenu({divId: 'task-menu',
                                         ros: RFH.ros});
        $('#task-select').selectmenu({collapsible:true});
        $('.task-option-select').selectmenu();

        new RFH.Domains.Pick({ros:RFH.ros,
            r_arm: RFH.pr2.r_arm_cart,
            r_gripper: RFH.pr2.r_gripper,
            l_arm: RFH.pr2.l_arm_cart,
            l_gripper: RFH.pr2.l_gripper});
        new RFH.Domains.Place({ros:RFH.ros,
            r_arm: RFH.pr2.r_arm_cart,
            r_gripper: RFH.pr2.r_gripper,
            l_arm: RFH.pr2.l_arm_cart,
            l_gripper: RFH.pr2.l_gripper});
        new RFH.Domains.PickAndPlace({ros:RFH.ros});
        new RFH.Domains.ADL({ros:RFH.ros});
        new RFH.Domains.RealtimeBaseSelection({ros:RFH.ros});
    };
    return module;
})(RFH || {});
