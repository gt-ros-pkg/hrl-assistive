var RFH = (function (module) {
    module.ShaverToggle = function (options) {
        var self = this;
        var ros = options.ros;
        var divId = options.divId || 'toggle-shaver-button';
        var $div = $('#'+divId);
        $div.button();

        var toggleShaverPub = new ROSLIB.Topic({
            ros: ros,
            name: "/toggle_shaver",
            messageType: 'std_msgs/Bool'
        });
        toggleShaverPub.advertise();

        var clickCB = function (event) {
            toggleShaverPub.publish({data:true});
        };
        $div.on('click.rfh', clickCB);
    };
    return module;
})(RFH || {});
