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

        var updateContacts = function () {
            contacts = []; 
            for (var i=0; i < contactDetectors.length; i+=1) {
                Array.prototype.push.apply(contacts, contactDetectors[i].getContacts());
            }
            updateDisplay();
        };

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
    return module;
})(RFH || {});
