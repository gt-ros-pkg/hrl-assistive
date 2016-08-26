RFH.regions = [];
RFH.RegionView = function (options) {
    'use strict';
    var self = this;
    var ros = options.ros;
    self.name = options.name;
    var viewer = options.viewer;
    var center = options.center || {x:0, y:0, z:0};
    var radius = options.radius || 0.1;
    
   self.getRadius = function () { return radius; };
   self.setRadius = function (rad) {
         radius = rad;
         var regionGeom = new THREE.SphereGeometry(radius, 16, 16);
         regionMesh.geometry = regionGeom;
    };

    self.getCenter = function () { return center; };
    self.setCenter = function (cen) {
         center = cen;
         regionMesh.position.set(cen.x, cen.y, cen.z); 
         };

    self.show = function () {
        regionMesh.visible = true;
    };

    self.hide = function () {
        regionMesh.visible = false;
    };

    self.remove = function (){
        viewer.scene.remove(regionMesh);
    };
   
    var regionGeom = new THREE.SphereGeometry(radius, 16, 16);
    var regionMaterial = new THREE.MeshBasicMaterial({
        color: 0x333333,
        opacity: 0.3, 
        transparent: true
    });
    var regionMesh = new THREE.Mesh(regionGeom, regionMaterial);

    regionMesh.position.set(center.x, center.y, center.z);
    regionMesh.userData.interactive = false;

    viewer.scene.add(regionMesh);
};
