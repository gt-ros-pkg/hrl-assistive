import librosa
import numpy as np

def main():
	#audio_filename1 = './sounds/data1.wav'
	#audio_filename2 = './sounds/data2.wav'
	#audio_filename3 = './sounds/data5.wav'
	#book = './sounds/AudioBook.wav'
	# audio_filename4 = './sounds/data20.wav'
	audio_filename5 = './sounds/data13.wav'

	sr = 44100
	#b1, _ = librosa.load(book, sr=sr, mono=True)
	#y1, _ = librosa.load(audio_filename1, sr=sr, mono=True)
	#y2, _ = librosa.load(audio_filename2, sr=sr, mono=True)
	#y3, _ = librosa.load(audio_filename3, sr=sr, mono=True)
	# y4, _ = librosa.load(audio_filename4, sr=sr, mono=True)
	y5, _ = librosa.load(audio_filename5, sr=sr, mono=True)

	#print y1.shape, y2.shape, y3.shape

	#Amplify
	#y1 = y1
	#y2 = y2*3
	#y3 = y3*5
	#y4 = y4*6
	y5 = y5*3

	# Manually Crop
	#y1 = y1[55536:147696]
	#y2 = y2[0:92160]
	#y3 = y3[0:92160]
	#b1 = b1[100000:300000]
	#y4 = y4[100000:192160]
	y5 = y5[0:92160]

	#librosa.output.write_wav('./sounds/cropped/data1crop4.wav', y1, sr)
	#librosa.output.write_wav('./sounds/cropped/data2crop4.wav', y2, sr)
	#librosa.output.write_wav('./sounds/cropped/data5crop4.wav', y3, sr)
	#librosa.output.write_wav('./sounds/cropped/AudioBookCrop.wav', b1, sr)
	# librosa.output.write_wav('./sounds/cropped/data20crop4Crop.wav', y4, sr)
	librosa.output.write_wav('./sounds/cropped/data13crop4.wav', y5, sr)	

if __name__ == '__main__':
	main()
