import numpy
import pylab
import harkpython.harkbasenode as harkbasenode

class HarkNode(harkbasenode.HarkBaseNode):
    def __init__(self):
        self.outputNames = ("OUTPUT",)
        self.outputTypes = ("prim_int",)

        self.im = None
        self.cb = None
        self.specgram = []
        self.winlen = 300
        pylab.ion()
        pylab.hold(False)

    def calculate(self):

        if len(self.FFT.keys()) >0:
            key = self.FFT.keys()[0]
        else:
            return
        
        tmp = numpy.array(abs(self.FFT[key][:]))
        if(numpy.sum(tmp) == 0):
            return;
        self.specgram.append(20 * numpy.log10(tmp))

        if self.count % 50 == 0:
            if len(self.specgram) > self.winlen:
                self.specgram = self.specgram[len(self.specgram)-self.winlen:]

            arr = numpy.array(self.specgram).transpose()

            if self.im is None:
                self.im = pylab.imshow(arr)
                self.cb = pylab.colorbar()
                pylab.xlim([0, self.winlen])
                pylab.xlabel("Time [frame]")
                pylab.ylabel("Frequency bin")
                pylab.gca().invert_yaxis()
                self.cb.set_label("Power [dB]")
            else:
                self.im.set_array(arr)
                ext = self.im.get_extent()
                self.im.set_extent((ext[0], arr.shape[1], ext[2], ext[3]))
                self.cb.set_clim(vmin=arr.min(), vmax=arr.max())
                self.cb.draw_all()

            pylab.xticks(range(0, self.winlen, 50),
                         range(self.count - self.winlen, self.count, 50))
            pylab.draw()
        
        self.outputValues["OUTPUT"] = 0
