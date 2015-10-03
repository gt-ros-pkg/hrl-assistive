import pylab
import numpy
import numpy as np
import harkpython.harkbasenode as harkbasenode

class HarkNode(harkbasenode.HarkBaseNode):
    def __init__ (self):
        self.outputNames = ("OUTPUT",)
        self.outputTypes = ("prim_int",)

        self.max_sources = 3 #self.MAX_N_SOURCES

        self.showwave = {}
        ## [ [] for x in xrange(self.max_sources) ]
        self.wavelen = 22050
        pylab.ion()
        pylab.hold(False)

    def calculate(self):
        self.outputValues["OUTPUT"] = 0

        for key in self.WAVE.keys():
            if key not in self.showwave.keys():
                self.showwave[key] = np.array(self.WAVE[key])
            else:
                self.showwave[key] = np.concatenate( (self.showwave[key], self.WAVE[key]), axis=1 )
                    
        if self.count % 10 == 0:
            for i, key in enumerate(self.WAVE.keys()):
                print type(key)
                if len(self.showwave[key]) > self.wavelen:
                    self.showwave[key] = self.showwave[key][len(self.showwave)-self.wavelen:]

                pylab.subplot(self.max_sources, 1, i+1)
                pylab.plot(self.showwave[key])
                pylab.xlim([0, self.wavelen])
                pylab.xticks(range(0, self.wavelen, 5000),
                             range(self.count, self.count + self.wavelen, 5000))
                pylab.ylabel("Magnitude of ID "+str(key))
                mx = max(abs(self.showwave[key]))
                pylab.ylim([-mx, mx])

            if self.count == 0:
                pylab.xlabel("Time [frame]")
            pylab.draw()
        

                    
                
if __name__ == "__main__":
    import cProfile
    for i in range(5):
        calculate(i, numpy.zeros((3, 512)))
