import numpy as np
import yaafelib as yaafe

__author__ = 'zerickson'

# Initialization
fp = yaafe.FeaturePlan(sample_rate=16000)
fp.addFeature('mfcc: MFCC blockSize=512 stepSize=256')
fp.addFeature('sr: SpectralRolloff blockSize=512 stepSize=256')
fp.addFeature('sf: SpectralFlux blockSize=512 stepSize=256')
engine = yaafe.Engine()
engine.load(fp.getDataFlow())
# Get input metadata
print engine.getInputs()
# Get output metadata
print engine.getOutputs()

# Extract features from a random numpy array
audio = np.random.randn(1, 1000000)
feats = engine.processAudio(audio)
print feats['mfcc'].shape
print feats['sf'].shape
print feats['sr'].shape

# Extracting features block per block
engine.reset()
# Iterate over 10 random blocks of audio data
for i in range(1, 10):
    # Generate random data
    audio = np.random.rand(1,32000)
    engine.writeInput('audio', audio)
    engine.process()
    # Read available feature data
    feats = engine.readAllOutputs()
    print [key for key in feats.keys()]
    print [value.shape for value in feats.values()]
# Flush out remaining data
engine.flush()
# Read last data
feats = engine.readAllOutputs()
print [key for key in feats.keys()]
print [value.shape for value in feats.values()]
