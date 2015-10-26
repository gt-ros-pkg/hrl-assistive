


class learning_hmm(learning_base):
    def __init__(self, aXData, nState, nMaxStep, nFutureStep=5, nCurrentStep=10, step_size_list=None, trans_type="left_right"):

        learning_base.__init__(self, aXData, trans_type)
