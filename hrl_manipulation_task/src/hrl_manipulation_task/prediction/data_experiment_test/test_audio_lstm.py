# All public gists https://gist.github.com/rwldrn
# Copyright 2017, Nao Tokui
# MIT License, https://gist.github.com/naotokui/12df40fa0ea315de53391ddc3e9dc0b9

import seaborn
import librosa
import numpy as np

from IPython.display import Audio
import matplotlib.pyplot as plt
import os
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.optimizers import RMSprop
import tensorflow as tf
from tqdm import tqdm
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from IPython.display import display


# load array to audio buffer and play!!
def sample(preds, temperature=1.0, min_value=0, max_value=1):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    v = np.argmax(probas)/float(probas.shape[1])
    return v * (max_value - min_value) + min_value

def main():
    #audio_filename = '369148__flying-deer-fx__music-box-the-flea-waltz.wav'
    #audio_filename = 'jsbach.wav'
    audio_filename = 'rosbag_microwave_train.wav'

    sr = 8000
    y, _ = librosa.load(audio_filename, sr=sr, mono=True)
    print y.shape
    print y
    print len(y)

    min_y = np.min(y)
    max_y = np.max(y)

    # normalize
    y = (y - min_y) / (max_y - min_y)
    print y.dtype, min_y, max_y

    Audio(y, rate=sr)

    #matplotlib inline
    plt.figure(figsize=(30,5))
    plt.plot(y[20000:20128].transpose())
    plt.show()

    # Build a model
    os.environ["KERAS_BACKEND"] = "tensorflow"

    # so try to estimate next sample afte given (maxlen) samples
    maxlen     = 128 # 128 / sr = 0.016 sec
    nb_output = 256  # resolution - 8bit encoding
    latent_dim = 128 

    inputs = Input(shape=(maxlen, nb_output))
    x = LSTM(latent_dim, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(latent_dim)(x)
    x = Dropout(0.2)(x)
    output = Dense(nb_output, activation='softmax')(x)
    model = Model(inputs, output)

    model.load_weights('/home/mpark/bagfiles/data_experiment_test/models/rosbag_microwave_train.hdf5')
    #optimizer = Adam(lr=0.005)
    optimizer = RMSprop(lr=0.01) 
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # try to estimate next_sample (0 -255) based on 256 previous samples 
    # step = 5
    # next_sample = []
    # samples = []
    # for j in tqdm(range(0, y.shape[0] - maxlen, step)):
    #     seq = y[j: j + maxlen + 1]  
    #     seq_matrix = np.zeros((maxlen, nb_output), dtype=bool) 
    #     for i,s in enumerate(seq):
    #         sample_ = int(s * (nb_output - 1)) # 0-255
    #         if i < maxlen:
    #             seq_matrix[i, sample_] = True
    #         else:
    #             seq_vec = np.zeros(nb_output, dtype=bool)
    #             seq_vec[sample_] = True
    #             next_sample.append(seq_vec)
    #     samples.append(seq_matrix)
    # #print type(samples), len(samples)
    # #print type(next_sample), len(next_sample)
    # samples = np.array(samples, dtype=bool)
    # next_sample = np.array(next_sample, dtype=bool)
    # print samples.shape, next_sample.shape

    # csv_logger = CSVLogger('training_audio.log')
    # escb = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    # checkpoint = ModelCheckpoint("models/audio-{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=1, period=2)

    # model.fit(samples, next_sample, shuffle=True, batch_size=256, verbose=1, #initial_epoch=50,
    #           validation_split=0.1, nb_epoch=500, callbacks=[csv_logger, escb, checkpoint])

    # #matplotlib inline
    # print "Training history"
    # fig = plt.figure(figsize=(10,4))
    # ax1 = fig.add_subplot(1, 2, 1)
    # plt.plot(model.history.history['loss'])
    # ax1.set_title('loss')
    # ax2 = fig.add_subplot(1, 2, 2)
    # plt.plot(model.history.history['val_loss'])
    # ax2.set_title('validation loss')

    seqA = []
    print y.shape
#    for start in range(5000,220000,10000):
    for start in tqdm(range(0, y.shape[0]- maxlen)):#, 5)):
        seq = y[start: start + maxlen]  
        #print len(seq) = 128
        seq_matrix = np.zeros((maxlen, nb_output), dtype=bool) 
        for i,s in enumerate(seq):
            sample_ = int(s * (nb_output - 1)) # 0-255
            seq_matrix[i, sample_] = True

        #for i in tqdm(range(5000)):
        z = model.predict(seq_matrix.reshape((1,maxlen,nb_output)))
        s = sample(z[0], 1.0)
        #print s, s.shape = 1 float value
        #seq = np.append(seq, s)
            # sample_ = int(s * (nb_output - 1))    
            # seq_vec = np.zeros(nb_output, dtype=bool)
            # seq_vec[sample_] = True

            # seq_matrix = np.vstack((seq_matrix, seq_vec))  # added generated note info 
            # seq_matrix = seq_matrix[1:]
            
        # scale back 
        #seq = seq * (max_y - min_y) + min_y
        s = s * (max_y - min_y) + min_y
        
        # plot
        # plt.figure(figsize=(30,5))
        # plt.plot(seq.transpose())
        # plt.show()
        
        #display(Audio(seq, rate=sr))
        #seqA.append(seq)
        seqA.append(s)
        #join seq data    
    #print seqA
    #print type(seqA), len(seqA)
    seqA2 = np.hstack(seqA)
    #print seqA2
    #print type(seqA2), len(seqA2)
    librosa.output.write_wav('rosbag_microwave_predict2.wav', seqA2, sr)


if __name__ == '__main__':
    main()


