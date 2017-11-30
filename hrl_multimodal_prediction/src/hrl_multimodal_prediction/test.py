# from predictor import predictor
import config
import librosa
import pyaudio
import wave
import sys
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation
import librosa
import scipy.io.wavfile as wav
import scipy as sp
import scipy.interpolate

import rospy
from std_msgs.msg import String

def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass


# Sent for figure
# font = {'size'   : 9}
# # Setup figure and subplots
# f0 = figure(num = 0, figsize = (12, 8))#, dpi = 100)
# f0.suptitle("ARtag & Audio combined Prediction", fontsize=12)
# ax01 = subplot2grid((2, 2), (0, 0))


# fig, ax = plt.subplots()
# x = np.arange(0, 2*np.pi, 0.01)
# line, = ax.plot(x, np.sin(x))

# def animate(i):
# 	print i
# 	line.set_ydata(np.sin(x + i/10.0))  # update the data
# 	# line.set_ydata()
# 	return line,

# def main():
# 	# pred = predictor()
# 	# pred.test()
# 	# print pred.TESTA

# 	ani = animation.FuncAnimation(fig, animate, np.arange(1, 20), interval=25, repeat=False)
# 	plt.show()


# 	y, sr = librosa.load('DO2.wav', mono=True)	
# 	librosa.output.write_wav('aa.wav', y, sr)
# 	print y.shape, y.max(), y.min(), type(y[0])
# 	pya = pyaudio.PyAudio()
# 	stream = pya.open(format=pyaudio.paFloat32, channels=1, rate=44100/2, output=True)
# 	stream.write(y)
# 	stream.stop_stream()
# 	stream.close()
# 	pya.terminate()
	# print("* Preview completed!")

# if __name__ == '__main__':
# 	main()    
