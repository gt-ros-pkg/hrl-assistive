var RFH = (function (module) {
    module.extendROSJS = function (ros) {
        ros.msgs = {};
        ros.getMsgDetailsClient = new ROSLIB.Service({
            ros: ros,
            name: '/rosapi/message_details',
            serviceType: 'rosapi/MessageDetails'});

        ros.getMsgDetails = function (msgType) {
            if (ros.msgs[msgType] === undefined) {
                var req = new ROSLIB.ServiceRequest({type: msgType});
                ros.getMsgDetailsClient.callService(req, function(res) {
                    ros.msgs = ros.msgs || {};
                    for (var item in res.typedefs){
                        if (ros.msgs[res.typedefs[item].type] === undefined) {
                            console.log('Imported '+
                                res.typedefs[item].type.toString()+' Msg');
                            ros.msgs[res.typedefs[item].type] = res.typedefs[item] ;
                        }
                    }
                });
            }
        };

        ros.composeMsg = function (type) {
            if (ros.msgs[type] === undefined) {
                console.error('Cannot compose '+ type + ' message: '+
                              'Message details not imported');
                return;
            }
            var msg = {};
            for (var field in ros.msgs[type].fieldnames){
                var example = ros.msgs[type].examples[field];
                if (example === "{}"){
                    msg[ros.msgs[type].fieldnames[field]] =
                            ros.composeMsg(ros.msgs[type].fieldtypes[field]);
                } else if (example === "[]"){
                    msg[ros.msgs[type].fieldnames[field]] = [];
                } else if (example === ""){
                    msg[ros.msgs[type].fieldnames[field]] = "";
                } else if (example === "False"){
                    msg[ros.msgs[type].fieldnames[field]] = false;
                } else if (parseInt(example) === 0){
                    msg[ros.msgs[type].fieldnames[field]] = 0;
                } else if (example === undefined) {
                    msg[ros.msgs[type].fieldnames[field]] = undefined;
                }
            }
            return msg;
        };
    };
    return module;
})(RFH || {});
