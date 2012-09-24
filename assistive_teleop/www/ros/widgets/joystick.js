//this script depends on jquery
var Joystick = function(targetID) {
/*  var target = $('#' + targetID);
  var w = target.width();
  var h = target.height();

  var chars = 'abcdefghijklmnopqrstuvwxyz';
  var id = '';
  for (var i = 0; i  < 16; i++) {
    id += chars[Math.floor(Math.random()*chars.length)];
  }
  target.html('<canvas id="' + id + '" width="' + w + '" height="' + h + '"><\/canvas>');
  var cnvs = $('#' + id);*/
  var f = document.getElementById("joystick")
  var g = f.getContext("2d");
  var w = target.width();
  var h = target.height();

  //setup drawing
  function circle(x,y) {
    g.clearRect(0,0,w,h);
    g.save();
      g.fillStyle = "rgba(255,255,255,1)";
      g.fillRect(0,0,w,h);
    g.restore();
    g.save();
      g.fillStyle = "rgba(0,0,0,1)";
      g.translate(w/2,h/2);
      g.beginPath();
      g.arc(x,y,w/12,0,Math.PI*2);
      g.closePath();
      g.fill();
    g.restore();
  }

  //setup interactivity
  var ths = this;
  this.x = 0;
  this.z = 0;
  var active = false;
  var offset = cnvs.offset();

  function activate(val) {
    active = val;
    if (!val) {
      ths.x = 0;
      ths.z = 0;
      circle(0,0);
    }
  }
  cnvs.mousedown(function() {
    activate(true);
    offset = cnvs.offset();
  });
  cnvs.mouseup(function() {
    activate(false);
  });
  cnvs.hover(function(e) {}, function(e) {
    activate(false);
  });
  cnvs.mousemove(function(e) {
    if (!active) return;
    var jx = e.pageX - w/2 - offset.left;
    var jy = e.pageY - h/2 - offset.top;
    circle(jx,jy);
    var tz = (-jx/(w/2))*1.25;
    var tx = (-jy/(h/2))*1.25;
    if (tz < -1) tz = -1;
    if (tz > 1) tz = 1;
    if (tx < -1) tx = -1;
    if (tx > 1) tx = 1;
    ths.z = tz;
    ths.x = tx;
  });

  //draw the initial stick
  circle(0,0);
};

