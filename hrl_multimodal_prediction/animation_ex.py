import numpy as np
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from matplotlib import style

#style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

# image = np.loadtxt('./csv/data1.txt')	
# image = np.rollaxis(image, 1, 0)
# print image.shape
# x=image[0]
# y=image[1]
# z=image[2]

# l = image.shape[1]
# print l

# t = np.linspace(0,2,l)
# print t

def animate(i):
	image = np.loadtxt('./csv/data1.txt')	
	image = np.rollaxis(image, 1, 0)
	print image.shape
	x=image[0]
	y=image[1]
	z=image[2]

	l = image.shape[1]
	print l

	t = np.linspace(0,2,l)
	print t

	xs = []
	ys = []
	for i in range(l):
		xs.append(t[i])
		ys.append(x[i])
	ax1.clear()
	ax1.plot(xs, ys)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()

