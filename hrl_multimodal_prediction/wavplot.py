import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import librosa

audio1, sr1 = librosa.load('./sounds/cropped/data13crop4.wav', mono=True)
audio2, sr2 = librosa.load('./sounds/predicted/combined_predict_testdata13FromMFCC3.wav', mono=True)

pred_image = np.loadtxt('./csv/predicted/combined_predict_testdata20.txt')

print pred_image
print len(pred_image[0])

pred_image = pred_image[:, 10:]
print pred_image.shape

# l1 = len(audio1)
# l2 = len(audio2)
# print l1
# print l2
# print l1-l2
# print 512*5

# spf = wave.open('./sounds/predicted/combined_predict_testdata13FromMFCC3.wav','r')

# #Extract Raw Audio from Wav File
# signal = spf.readframes(-1)
# signal = np.fromstring(signal, 'Int16')
# fs = spf.getframerate()

# #If Stereo
# if spf.getnchannels() == 2:
#     print 'Just mono files'
#     sys.exit(0)


# Time=np.linspace(0, len(signal)/fs, num=len(signal))

# plt.figure(1)
# plt.title('Signal Wave...')
# plt.plot(Time,signal)
# plt.show()
