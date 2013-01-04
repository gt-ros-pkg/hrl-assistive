var extendROSJS = function () {
    window.ros.msgs = {};
    window.ros.getMsgDetailsClient = new window.ros.Service({
        name: '/rosapi/message_details',
        serviceType: 'rosapi/MessageDetails'});

    window.ros.getMsgDetails = function (msgType) {
        var req = new window.ros.ServiceRequest({type: msgType});
        window.ros.getMsgDetailsClient.callService(req, function(res) {
            window.ros.msgs = window.ros.msgs || {};
            for (item in res.typedefs){
                if (window.ros.msgs[res.typedefs[item].type] === undefined) {
                    console.log('Imported '+
                        res.typedefs[item].type.toString()+'Msg')
                    window.ros.msgs[res.typedefs[item].type] = res.typedefs[item] 
                }
            }
        });
    };

    window.ros.composeMsg = function (type) {
        if (window.ros.msgs[type] === undefined) {
            console.error('Cannot compose '+ type + 'message:'+
                          'Message details not imported');
            return
        }
        var msg = {}
        for (field in window.ros.msgs[type].fieldnames){
            var example = window.ros.msgs[type].examples[field];
            if (example === "{}"){
                msg[window.ros.msgs[type].fieldnames[field]] =
                        window.ros.composeMsg(window.ros.msgs[type].fieldtypes[field]);
            } else if (example === "[]"){
                msg[window.ros.msgs[type].fieldnames[field]] = [];
            } else if (example === ""){
                msg[window.ros.msgs[type].fieldnames[field]] = "";
            } else if (example === "False"){
                msg[window.ros.msgs[type].fieldnames[field]] = false;
            } else if (parseInt(example) === 0){
                msg[window.ros.msgs[type].fieldnames[field]] = 0;
            } else if (example === undefined) {
                msg[window.ros.msgs[type].fieldnames[field]] = undefined;
            }
        }
        return msg
    };
}

