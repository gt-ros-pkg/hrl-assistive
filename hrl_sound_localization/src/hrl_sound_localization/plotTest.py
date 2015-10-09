import numpy
import numpy as np
import pylab
import harkpython.harkbasenode as harkbasenode

class HarkNode(harkbasenode.HarkBaseNode):
    def __init__(self):
        self.outputNames = ("OUTPUT",)
        self.outputTypes = ("prim_int",)

        self.max_sources = 3 #self.MAX_N_SOURCES
        self.showpower = {}
        self.powerlen  = 22050
        self.global_mx = 0

        pylab.ion()
        pylab.hold(False)

    def calculate(self):


        for key in self.FFT.keys():
            if key not in self.showpower.keys():
                self.showpower[key] = 20.*np.log10(np.array(self.FFT[key]))
            else:
                self.showpower[key] = np.concatenate( (self.showpower[key], 20.*np.log10(np.array(self.FFT[key]))), axis=1 )

            # remove keys not in 
            for i, key in enumerate(self.showpower.keys()):
                if key not in self.FFT.keys():
                    del self.showpower[key]

        if self.count % 10 == 0:

            for i, key in enumerate(self.FFT.keys()):
                if len(self.showpower[key]) > self.powerlen:
                    self.showpower[key] = self.showpower[key][-self.powerlen:]

                ## pylab.subplot(self.max_sources, 1, i+1)
                pylab.plot(self.showpower[key], label=key)
                pylab.xlim([0, self.powerlen])
                pylab.xticks(range(0, self.powerlen, 5000),
                             range(self.count, self.count + self.powerlen, 5000))
                pylab.ylabel("Magnitude of ID")
                
                mx = np.amax(self.showpower[key])

                if self.global_mx < mx:
                    self.global_mx = mx
            pylab.ylim([0, self.global_mx])
            pylab.legend()


            if self.count == 0:
                pylab.xlabel("Time [frame]")
            pylab.draw()

            
        self.outputValues["OUTPUT"] = 0
