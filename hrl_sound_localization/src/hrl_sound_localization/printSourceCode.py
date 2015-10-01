import numpy
import pylab
import exceptions
import harkpython.harkbasenode as harkbasenode
## import harkbasenode

import hrl_common_code_darpa_m3.visualization.draw_scene as ds

def cartesian2polar(xyz):
    """cartesian2polar(x, y, z)
converts coordinate from xyz to r\theta\phi"""
    x, y, z = xyz
    ## if z == 0:
    ##     raise exceptions.ValueError("cartesian " + str(xyz) + " z must be nonzero.")

    r = pylab.sqrt(sum(map(lambda x: x**2, [x, y, z])))
    theta = pylab.arccos(z/r)
    phi = pylab.arccos(x / pylab.sqrt(sum(map(lambda x: x**2, [x, y]))))
    if y < 0: phi *= -1

    return (r, theta, phi)

def r2d(r):
    return 180.0 / numpy.pi * r

class HarkNode(harkbasenode.HarkBaseNode):
    def __init__(self):
        self.outputNames = ("OUTPUT", )
        self.outputTypes = ("prim_int",)

        self.plot_frame   = {}
        self.plot_azimuth = {}
        self.plot_r       = {}
        self.plot_theta       = {}
        self.winlen = 300

    def calculate(self):
        for src in self.SOURCES:
            if src.has_key("x"):
                self.plot_frame.setdefault(src["id"], [])
                self.plot_r.setdefault(src["id"], [])
                self.plot_theta.setdefault(src["id"], [])
                self.plot_azimuth.setdefault(src["id"], [])
                
                self.plot_frame[src["id"]].append(self.count)
                (r, theta, phi) = cartesian2polar(src["x"])
                self.plot_r[src["id"]].append(r)
                self.plot_theta[src["id"]].append(r2d(theta))
                self.plot_azimuth[src["id"]].append(r2d(phi))
                
        if self.count % 10 == 0:
            for i, srcid in enumerate(self.plot_frame.keys()):
                
                if i == 0: pylab.hold(False)
                else: pylab.hold(True)
                pylab.plot(self.plotx[srcid], self.ploty[srcid], "." + "rgb"[srcid%3])
            pylab.xlim([self.count-self.winlen, self.count])
            pylab.ylim([-180, 180])
            pylab.xlabel("Time [frame]")
            pylab.ylabel("Azimuth [deg]")
            pylab.draw()

        self.outputValues["OUTPUT"] = 0


    def __del__(self):
        for i, srcid in enumerate(self.plotx.keys()):
            if i == 0: pylab.hold(False)
            else: pylab.hold(True)
            pylab.plot(self.plotx[srcid], self.ploty[srcid], "." + "rgb"[srcid%3])
        pylab.xlim([0, self.count])
        pylab.ylim([-180, 180])
        pylab.xlabel("Time [frame]")
        pylab.ylabel("Azimuth [deg]")
        pylab.savefig("aaa.png")
        pylab.close()
