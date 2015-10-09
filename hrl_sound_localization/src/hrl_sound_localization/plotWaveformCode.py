import pylab
import numpy
import harkpython.harkbasenode as harkbasenode

class HarkNode(harkbasenode.HarkBaseNode):
    def __init__ (self):
        self.outputNames = ("OUTPUT",)
        self.outputTypes = ("prim_int",)
        
        self.showwave = None
        self.wavelen = 22050
        pylab.ion()
        pylab.hold(False)

    def calculate(self):
        self.outputValues["OUTPUT"] = 0

        if self.showwave == None:
            self.showwave = numpy.array(self.WAVE)
        else:
            self.showwave = numpy.concatenate((self.showwave, self.WAVE), axis = 1)

        if self.count % 10 == 0:
            if self.showwave.shape[1] > self.wavelen:
                self.showwave = self.showwave[:, len(self.showwave)-self.wavelen:]
                
            for ch in range(self.showwave.shape[0]):
                pylab.subplot(self.showwave.shape[0], 1, ch+1)
                pylab.plot(self.showwave[ch, :])
                pylab.xlim([0, self.wavelen])
                pylab.xticks(range(0, self.wavelen, 5000),
                             range(self.count, self.count + self.wavelen, 5000))
                pylab.ylabel("Magnitude")
                mx = max(abs(self.showwave[ch, :]))
                pylab.ylim([-mx, mx])

            if self.count == 0:
                pylab.xlabel("Time [frame]")
            pylab.draw()
        
if __name__ == "__main__":
    import cProfile
    for i in range(5):
        calculate(i, numpy.zeros((3, 512)))
