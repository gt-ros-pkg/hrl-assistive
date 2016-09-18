var RFH = (function (module) {
    module.BumperDisplay = function (options) {
        'use strict';
        var self = this;
        var tfClient = options.tfClient;
        var head = options.head;
        var camera = options.camera;
        var skins = options.skins;
        var visible = true;
        var $displayDiv = $('#bumper-contact-display');
        var $contactMarkers = $('.bumper');
        var contacts = [];

        var contactDetectors = [];
        for (var i=0; i<skins.length; i+=1) {
            contactDetectors.push(new module.SkinContactDetector({skinUtil: skins[i], 
                                                                  tfClient: tfClient}));
            contactDetectors[i].updateCBList.push(updateContacts);
        }

        self.show = function () {
            visible = true;
            $displayDiv.show();
        };

        self.hide = function () {
            visible = false;
            $displayDiv.hide();
        };

        var displayInViewContact = function (imgPt, marker) {
            // Show indicator of in-view contact at imgPt
            var w = $displayDiv.width(); 
            var h = $displayDiv.height(); 
            marker.css({left:imgPt[0]*w, top:imgPt[1]*h});
        };

        var updateContacts = function () {
            contacts = []; 
            for (var i=0; i < contactDetectors.length; i+=1) {
                Array.prototype.push.apply(contacts, contactDetectors[i].getContacts());
            }
            updateDisplay();
        };

        var inView = function (pt) {
            return (pt[0] >= 0 && pt[0] <=1 && pt[1] >= 0 && pt[1] <= 1);
        };

        var notInView = function (pt) {
            return !inView(pt);
        };

        var updateDisplay = function () {
            if (!contacts.length || !visible) {
                $contactMarkers.hide();
                return;
             }
            var imgPts = camera.projectPoints(contacts, 'base_link');    
            if (imgPts === undefined) { return; }
            // Find and display contact points in the camera view
            var ptsInView = imgPts.filter(inView);
            var marker;
            for (var i=0; i<ptsInView.length && i<$contactMarkers.length; i+=1) {
                if (i<ptsInView.length) {
                    marker = $($contactMarkers[i]).show();
                    displayInViewContact(ptsInView[i], marker);
                } else {
                    $contactMarkers[i].hide();
                }
            }
        };
    };

    module.SkinContactDetector = function (options) {
        'use strit';
        var self = this;
        var skin = options.skinUtil;
        var contactForceThreshold = options.contactForceThreshold || 0.0;
        var tfClient = options.tfClient;
        var tfLinkToBase = new THREE.Matrix4();
        var contactPoints = [];

        var updateTF = function (tf) {
            var tfPos = new THREE.Vector3(tf.translation.x,
                                          tf.translation.y,
                                          tf.translation.z);
            var tfQuat = new THREE.Quaternion(tf.rotation.x,
                                              tf.rotation.y,
                                              tf.rotation.z, 
                                              tf.rotation.w);
            tfLinkToBase.makeRotationFromQuaternion(tfQuat);
            tfLinkToBase.setPosition(tfPos);
        };
        var tfSub = function () {
            if (skin.baseLink !== null) {
                tfClient.subscribe(skin.baseLink, updateTF);
                console.log("Skin Frame identified. Subscribing to tf for ", skin.baseLink);
            } else {
                console.log("Waiting for skin base frame");
                setTimeout(tfSub, 1000);
            }
        };
        tfSub();

        var magnitude = function (x, y, z) {
            return Math.sqrt(x*x + y*y + z*z);
        };
            
        var updateContacts = function (contactPts) {
            contactPoints = [];
            for (var i=0; i < contactPts.length; i += 3) {
                contactPoints.push(contactPts.splice(0,3));
            }
        };

        self.updateCBList = [];
        processCallbacks = function () {
            for (var i = 0; i<self.updateCBList.length; i+=1) {
                self.updateCBList[i](contactPoints);
            }
        };

        var taxelCB = function (taMsg) {
            var contactPts = [];
            var mag;
            for (var idx = 0; idx < taMsg.values_x.length; idx += 1) {
                mag = magnitude(taMsg.values_x[idx], taMsg.values_y[idx], taMsg.values_z[idx]);
                if (mag > contactForceThreshold) {
//                    console.log("Contact Detected!");
                    contactPts.push(taMsg.centers_x[idx]);
                    contactPts.push(taMsg.centers_y[idx]);
                    contactPts.push(taMsg.centers_z[idx]);
                }
            }
            tfLinkToBase.applyToVector3Array(contactPts);
            updateContacts(contactPts);
            processCallbacks();
        };
        skin.forceCBArray.push(taxelCB);

        self.getContacts = function () {
            return contactPoints;
        };
    };
    return module;

})(RFH || {});
