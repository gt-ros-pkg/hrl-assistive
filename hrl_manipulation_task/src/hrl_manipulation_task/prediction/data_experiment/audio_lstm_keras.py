# All public gists https://gist.github.com/rwldrn
# Copyright 2017, Nao Tokui
# MIT License, https://gist.github.com/naotokui/12df40fa0ea315de53391ddc3e9dc0b9

import seaborn
import librosa
import numpy as np

audio_filename = '369148__flying-deer-fx__music-box-the-flea-waltz.wav'
sr = 8000
y, _ = librosa.load(audio_filename, sr=sr, mono=True)

print y.shape #same as len(y), which is a vector or audio samples

#librosa.output.write_wav('waltz.wav', y, sr) #--Works!

min_y = np.min(y)
max_y = np.max(y)

# normalize
y = (y - min_y) / (max_y - min_y)
print y.dtype, min_y, max_y

#librosa.output.write_wav('waltz.wav', y, sr) #--Works!

# import matplotlib.pyplot as plt
# plt.figure(figsize=(30,5))
# plt.plot(y[0:157696].transpose())
# plt.show()


#############################################################
#Perhaps use a phase vocoder or sth to reduce the training time

# Build a model
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.optimizers import RMSprop
import tensorflow as tf

# so try to estimate next sample after given (maxlen) samples
maxlen     = 128 # 128 / sr = 0.016 sec #1024/44100 = 0.003s
nb_output = 256  # resolution - 8bit encoding
latent_dim = 128 

inputs = Input(shape=(maxlen, nb_output))
x = LSTM(latent_dim, return_sequences=True)(inputs)
x = Dropout(0.4)(x)
x = LSTM(latent_dim)(x)
x = Dropout(0.4)(x)
output = Dense(nb_output, activation='softmax')(x)
model = Model(inputs, output)

# Printed
# inputs = Tensor("input_1:0", shape=(?, 1024, 2048), dtype=float32)
# x = Tensor("cond_1/Merge:0", shape=(?, 1024), dtype=float32)
# output = Tensor("Softmax:0", shape=(?, 2048), dtype=float32)

#optimizer = Adam(lr=0.005)
optimizer = RMSprop(lr=0.01) 
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

from tqdm import tqdm
# try to estimate next_sample (0 -255) based on 256 previous samples 
step = 5
next_sample = []
samples = []
for j in tqdm(range(0, y.shape[0] - maxlen, step)):
    seq = y[j: j + maxlen + 1]  
    seq_matrix = np.zeros((maxlen, nb_output), dtype=bool) #initializing false matrix of the size 128x256
    for i,s in enumerate(seq):
        sample_ = int(s * (nb_output - 1)) # 0-255
        if i < maxlen:
            seq_matrix[i, sample_] = True
        else:
            seq_vec = np.zeros(nb_output, dtype=bool)
            seq_vec[sample_] = True
            next_sample.append(seq_vec)
    samples.append(seq_matrix)
samples = np.array(samples, dtype=bool)
next_sample = np.array(next_sample, dtype=bool)
print samples.shape, next_sample.shape


from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
csv_logger = CSVLogger('training_audio.log')
escb = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
checkpoint = ModelCheckpoint("models/audio-{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=1, period=2)

model.fit(samples, next_sample, shuffle=True, batch_size=250, verbose=1, validation_split=0.1, nb_epoch=500, callbacks=[csv_logger, escb, checkpoint])



import matplotlib.pyplot as plt
# #%matplotlib inline

print "Training history"
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(1, 2, 1)
plt.plot(model.history.history['loss'])
ax1.set_title('loss')
ax2 = fig.add_subplot(1, 2, 2)
plt.plot(model.history.history['val_loss'])
ax2.set_title('validation loss')
#load array to audio buffer and play!!

from IPython.display import Audio, display
    
def sample(preds, temperature=1.0, min_value=0, max_value=1):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    v = np.argmax(probas)/float(probas.shape[1])
    return v * (max_value - min_value) + min_value
    

seqA = []
#model.load_weights("/home/mpark/bagfiles/data_experiment/models/audio-91-1.06.hdf5") 
for start in range(5000,220000,10000):
    seq = y[start: maxlen]  
    seq_matrix = np.zeros((maxlen, nb_output), dtype=bool) 
    for i,s in enumerate(seq):
        sample_ = int(s * (nb_output - 1)) # 0-255
        seq_matrix[i, sample_] = True

    for i in tqdm(range(5000)):
        z = model.predict(seq_matrix.reshape((1,maxlen,nb_output)))
        s = sample(z[0], 1.0)
        seq = np.append(seq, s)

        sample_ = int(s * (nb_output - 1))    
        seq_vec = np.zeros(nb_output, dtype=bool)
        seq_vec[sample_] = True

        seq_matrix = np.vstack((seq_matrix, seq_vec))  # added generated note info 
        seq_matrix = seq_matrix[1:]
        
    # scale back 
    seq = seq * (max_y - min_y) + min_y

    plot
    plt.figure(figsize=(30,5))
    plt.plot(seq.transpose())
    plt.show()
    
    display(Audio(seq, rate=sr))
    print seq
    seqA.append(seq)
    #join seq data

seqA2 = np.hstack(seqA)
librosa.output.write_wav('data1_seq.wav', seqA2, sr)


