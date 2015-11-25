
import harkpython.harkbasenode as harkbasenode

class HarkNode(harkbasenode.HarkBaseNode):
    def __init__(self):
        self.outputNames=("output",)  # one output terminal named "output"
        self.outputTypes=("prim_float",)  # the type is primitive float.

    def calculate(self):
        self.outputValues["output"] = self.input1 ** 3 + self.input2 ** 5
        # set output value
        # from two inputs: input1 and input2.
