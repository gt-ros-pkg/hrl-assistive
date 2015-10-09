
import numpy
import pylab
import exceptions
import harkpython.harkbasenode as harkbasenode


class HarkNode(harkbasenode.HarkBaseNode):
    def __init__(self):
        self.outputNames = ("OUTPUT", )
        self.outputTypes = ("prim_int",)


    def calculate(self):

        self.outputValues["OUTPUT"] = 0
        
        if len(self.SOURCES) > 0:
            print len(self.SOURCES)
            print self.SOURCES[0].keys()

        ## for src in self.SOURCES:

            ## if src.has_key("x"):
            ##     self.plot_frame.setdefault(src["id"], [])
            ##     self.plot_r.setdefault(src["id"], [])
            ##     self.plot_theta.setdefault(src["id"], [])
            ##     self.plot_azimuth.setdefault(src["id"], [])
                
            ##     self.plot_frame[src["id"]].append(self.count)
            ##     (r, theta, phi) = cartesian2polar(src["x"])
            ##     print r
            ##     self.plot_r[src["id"]].append(r)
            ##     self.plot_theta[src["id"]].append(r2d(theta))
            ##     self.plot_azimuth[src["id"]].append(r2d(phi))
                
