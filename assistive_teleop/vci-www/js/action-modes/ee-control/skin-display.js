var RFH = (function (module) {
    module.SkinDisplay = function (options) {
        'use strict';
        var self = this;
        var tfClient = options.tfClient;
        var head = options.head;
        var camera = options.camera;
        var skins = options.skins;
        var visible = true;
        var $displayDiv = $('#skin-contact-display');
        var $contactMarkers = $('.contact-marker');
        var $edgeMarkers = $('.edge-contact');
        var contacts = [];
        var contactEdgesActive = {'n': false, 'ne': false, 'e': false, 'se': false,
                                  's': false, 'sw': false, 'w': false, 'nw': false};

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
            console.log("All: ",  contacts.length);
            updateDisplay();
        };

        var contactDetectors = [];
        for (var i=0; i<skins.length; i+=1) {
            contactDetectors.push(new module.SkinContactDetector({skinUtil: skins[i], 
                                                                  tfClient: tfClient}));
            contactDetectors[i].updateCBList.push(updateContacts);
        }

        var inView = function (pt) {
            return (pt[0] >= 0 && pt[0] <=1 && pt[1] >= 0 && pt[1] <= 1);
        };

        var notInView = function (pt) {
            return !inView(pt);
        };

        var updateContactEdges = function (pts) {
            var ang;
            contactEdgesActive.n = false;
            contactEdgesActive.ne = false;
            contactEdgesActive.e = false;
            contactEdgesActive.se = false;
            contactEdgesActive.s = false;
            contactEdgesActive.sw = false;
            contactEdgesActive.w = false;
            contactEdgesActive.nw = false;
            for (var i=0; i<pts.length; i+=1) {
                ang = Math.atan2(pts[0], pts[1]);
                if (ang > -Math.PI/6 && ang <= Math.PI) {
                    contactEdgesActive.n = true;
                } else if (ang > Math.PI/6 && ang <= Math.PI/3) {
                    contactEdgesActive.nw = true;
                } else if (ang > Math.PI/3 && ang <= 2*Math.PI/3) {
                    contactEdgesActive.w = true;
                } else if (ang > 2*Math.PI/3 && ang <= 5*Math.PI/6) {
                    contactEdgesActive.sw = true;
                } else if (ang > -Math.PI/3 && ang <= -Math.PI/6) {
                    contactEdgesActive.ne = true;
                } else if (ang > -2*Math.PI/3 && ang <= -Math.PI/3) {
                    contactEdgesActive.e = true;
                } else if (ang > -5*Math.PI/6 && ang <= -2*Math.PI/3) {
                    contactEdgesActive.se = true;
                } else if (ang < -5*Math.PI/6 || ang > 5*Math.PI/6) {
                    contactEdgesActive.s = true;
                }
            }
        };

        var displayEdgeContacts = function () {
            for (var dir in contactEdgesActive) {
                if (contactEdgesActive.hasOwnProperty(dir)) {
                    if (contactEdgesActive[dir]) {
                        $edgeMarkers.filter('.'+dir).show();
                    } else {
                        $edgeMarkers.filter('.'+dir).hide();
                    }
                }
            }
        };

        var updateDisplay = function () {
            if (!contacts.length || !visible) {
                $contactMarkers.hide();
                $edgeMarkers.hide();
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
            // Find and display out-of-view contacts along the edges
            var outOfView = imgPts.filter(notInView);
            updateContactEdges(outOfView); 
            displayEdgeContacts();
        };

        var lookOutOfView = function (event) {
            RFH.actionMenu.startAction('lookingAction');
            var dPan = 0;
            var dTilt = 0;
            var curTar = $(event.currentTarget);
            if (curTar.hasClass('n')) {
                    dTilt = -0.3;
            } else if (curTar.hasClass('ne')) {
                    dPan = -0.3;
                    dTilt = -0.3;
            } else if (curTar.hasClass('e')) {
                    dPan = -0.3;
            } else if (curTar.hasClass('se')) {
                    dPan = -0.3;
                    dTilt = 0.3;
            } else if (curTar.hasClass('s')) {
                    dTilt = 0.3;
            } else if (curTar.hasClass('sw')) {
                    dPan = 0.3;
                    dTilt = 0.3;
            } else if (curTar.hasClass('w')) {
                    dPan = 0.3;
            } else if (curTar.hasClass('nw')) {
                    dPan = 0.3;
                    dTilt = -0.3;
            } 
            head.delPosition(dPan, dTilt);
        };

        $edgeMarkers.on('click.rfh', lookOutOfView);
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
//            tfLinkToBase.getInverse(tfLinkToBase);
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
//            console.log(skin.side, skin.part, contactPoints.length);
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
                    console.log("Contact Detected!");
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
