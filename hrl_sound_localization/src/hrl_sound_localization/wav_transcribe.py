#!/usr/bin/env python3

import speech_recognition as sr

# obtain path to "test.wav" in the same folder as this script
from os import path
WAV_FILE = path.join(path.dirname(path.realpath(__file__)), "/home/dpark/hark/hark-designer/userdata/networks/sep_files/sep_69.wav")

# use "test.wav" as the audio source
r = sr.Recognizer()
with sr.WavFile(WAV_FILE) as source:
    audio = r.record(source) # read the entire WAV file

# recognize speech using Google Speech Recognition
try:
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    print("Google Speech Recognition thinks you said " + r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

