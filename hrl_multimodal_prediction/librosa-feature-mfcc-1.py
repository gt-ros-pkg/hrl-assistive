# Generate mfccs from a time series
import matplotlib.pyplot as plt
import librosa
import librosa.display

y, sr = librosa.load('./sounds/original/data13.wav')

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)

# Visualize the MFCC series

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()