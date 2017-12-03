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

# def talker():
#     pub = rospy.Publisher('chatter', String, queue_size=10)
#     rospy.init_node('talker', anonymous=True)
#     rate = rospy.Rate(10) # 10hz
#     while not rospy.is_shutdown():
#         hello_str = "hello world %s" % rospy.get_time()
#         rospy.loginfo(hello_str)
#         pub.publish(hello_str)
#         rate.sleep()

# if __name__ == '__main__':
#     try:
#         talker()
#     except rospy.ROSInterruptException:
#         pass


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


y, sr = librosa.load('./bagfiles/unpacked/data1.wav', mono=True)	
print y.shape, y.max(), y.min(), type(y[0])
p = pyaudio.PyAudio()
# deviceIndex = self.find_input_device()
device_index = None
for i in range(p.get_device_count()):
    devinfo = p.get_device_info_by_index(i)
    print('Device %d: %s'%(i, devinfo['name']))

    for keyword in ['HDA', 'Intel', 'PCH', 'ALC662']:
        if keyword in devinfo['name'].lower():
            print('Found an output: device %d - %s'%(i, devinfo['name']))
            device_index = i

if device_index is None:
    print('No preferred input found; using default input device.')


deviceIndex = 0
devInfo = p.get_device_info_by_index(deviceIndex)
print 'Audio device:', deviceIndex
print 'Sample rate:', devInfo['defaultSampleRate']
print 'Max input channels:',  devInfo['maxInputChannels']

stream = p.open(format=p.get_format_from_width(wave.getsampwidth()), channels=1, rate=44100/2, output=True, input_device_index=deviceIndex)
stream.write(y)
stream.close()
p.terminate()
print("* Preview completed!")

## self.stream.start_stream()



# stream = pya.open(format=pyaudio.paFloat32, channels=1, rate=44100/2, output=True)
# stream.write(y)
# stream.stop_stream()
# stream.close()
# pya.terminate()
# print("* Preview completed!")

# if __name__ == '__main__':
# 	main()    
