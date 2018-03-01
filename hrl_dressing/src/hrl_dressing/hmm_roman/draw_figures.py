__author__ = 'yuwenhao'

import sys
import matplotlib.pyplot as pp
import numpy as np
from matplotlib.ticker import FuncFormatter

import matplotlib.patches as patches

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

def show_confusion(cmat):
    total = float(cmat.sum())
    true = float(np.trace(cmat))
    percent_acc = "{0:.2f}".format((true/total)*100)

    # Plot Confusion Matrix
    Nlabels = np.size(categories)
    fig = pp.figure(0)
    ax = fig.add_subplot(111)
    figplot = ax.matshow(cmat, interpolation = 'nearest', origin = 'upper', extent=[0, Nlabels, 0, Nlabels], cmap=pp.cm.gray_r)
    pp.xlabel("Actual", fontsize=18)
    ax.xaxis.set_label_position('top')
    pp.ylabel("Predictions", fontsize=18)
    ax.set_title('Lowest Maximum Force : Accuracy = ' + str(percent_acc), fontsize=20, y=1.15)
    #ax.set_title('Generalization Across Subject : Accuracy = ' + str(percent_acc), fontsize=20, y=1.15)
    #ax.set_title('Generalization Across Subject : Accuracy = ' + str(percent_acc) + '\nThreshold by Lowest Maximum Force', fontsize=20, y=1.15)
    if len(categories) == 3:
        ax.set_xticks([0.5,1.5,2.5])
    else:
        ax.set_xticks([0.5,1.5,2.5,3.5])
    ax.set_xticklabels(categories, fontsize=15)
    if len(categories) == 3:
        ax.set_yticks([2.5,1.5,0.5])
    else:
        ax.set_yticks([3.5,2.5,1.5,0.5])
    ax.set_yticks([2.5,1.5,0.5])
    ax.set_yticklabels(categories, fontsize=15)
    #figbar = fig.colorbar(figplot)
    #figbar.ax.tick_params(labelsize=20)

    max_val = np.max(cmat)
    i = 0
    while (i < len(categories)):
        j = 0
        while (j < len(categories)):
            if cmat[i, j] < 0.3 * max_val:
                pp.text(j+0.45,-0.625 + len(categories)-i, str(int(cmat[i,j])), fontsize=35, color='black')
            elif cmat[i, j] < 100:
                pp.text(j+0.37,-0.625 + len(categories)-i, str(int(cmat[i,j])), fontsize=35, color='white')
            else:
                pp.text(j+0.27,-0.625 + len(categories)-i, str(int(cmat[i,j])), fontsize=35, color='white')
            j = j+1
        i = i+1

if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print 'Usage:\n python draw_figures [data_filename]\n'
        quit()
    categories = ['missed', 'good', 'caught']

    confusion_matrix = []
    percentages = []
    accuracies = []
    xonly_accuracies = []
    zonly_accuracies = []

    additional_percentages = []
    additional_accuracies = []

    datafile = open(sys.argv[1])
    for i in range(len(categories)):
        row = []
        row_tex = datafile.readline().strip().split()
        for ele in row_tex:
            row.append(float(ele.replace(',', '').replace('[', '').replace(']', '')))
        confusion_matrix.append(row)

    percentage_len = int(datafile.readline().strip())
    for i in range(percentage_len):
        percentages.append(float(datafile.readline().strip()))
    for i in range(percentage_len):
        accuracies.append(float(datafile.readline().strip()))

    hasxz = datafile.readline().strip()
    if hasxz == 'hasxz':
        for i in range(percentage_len):
            xonly_accuracies.append(float(datafile.readline().strip()))
        for i in range(percentage_len):
            zonly_accuracies.append(float(datafile.readline().strip()))

    additional_len = int(datafile.readline().strip())
    for i in range(additional_len):
        additional_percentages.append(float(datafile.readline().strip()))
    for i in range(additional_len):
        additional_accuracies.append(float(datafile.readline().strip()))

    datafile.close()

    confusion_matrix = np.matrix(confusion_matrix)
    show_confusion(confusion_matrix)
    '''
    for i in range(len(accuracies)):
        accuracies[i] *= 100
        if hasxz == 'hasxz':
            xonly_accuracies[i] *= 100
            zonly_accuracies[i] *= 100
    for i in range(additional_len):
        additional_accuracies[i] *= 100
    '''
    fig = pp.figure(1)
    ax = fig.add_subplot(1,1,1)

    #################################
    #ax.set_autoscale_on(False)

    pp.title('Accuracy w.r.t position threshold', fontsize=20)
    #pp.title('Accuracy w.r.t force threshold', fontsize=20)
    maincurve, = ax.plot(percentages, accuracies, 'g-', linewidth=4.0, label='Bivariate model')

    if hasxz == 'hasxz':
        xonly, = ax.plot(percentages, xonly_accuracies, 'b-', linewidth = 4.0, label='Univariate Model (Direction of Gravity)')
        zonly, = ax.plot(percentages, zonly_accuracies, 'y-', linewidth = 4.0, label='Univariate Model (Direction of Movement)')

    if not additional_len == 0:
        if not additional_len == 1:
            evenlysampled, = ax.plot(additional_percentages[0:-1], additional_accuracies[0:-1], 'go', markersize=9, label='Sampled points')
        minmax, = ax.plot(additional_percentages[-1], additional_accuracies[-1], 'ro', markersize=9, label='Minimal maximal force threshold')
    pp.ylabel("Accuracy", fontsize=18)
    #pp.xlabel("Force Threshold (N)", fontsize=18)
    pp.xlabel("Position (m)", fontsize=18)
    #pp.xlim([0, 0.87])


    avg_initial = 0.12125
    avg_elbow = 0.44
    avg_miss_end = 0.850016647285
    avg_good_end = 0.74096625823
    avg_caught_end = 0.505228792688
    std_miss_end = 0.000217967165412
    std_good_end = 0.0181755961634
    std_caught_end = 0.0702616306328

    #ax.plot([avg_initial, avg_initial], [0.0, 1], 'b--')
    #ax.plot([avg_elbow, avg_elbow], [0.0, 1], 'b--')

    #ax.plot([avg_miss_end, avg_miss_end], [0.0, 1], 'b--')
    ax.plot([avg_good_end, avg_good_end], [0.0, 1], 'b--')
    ax.add_patch(patches.Rectangle((avg_good_end-std_good_end, 0.0),   # (x,y)
                                    std_good_end*2,          # width
                                    1.0,          # height
                                   alpha=0.15,
                                    )
                  )
    ax.plot([avg_caught_end, avg_caught_end], [0.0, 1], 'b--')
    ax.add_patch(patches.Rectangle((avg_caught_end-std_caught_end, 0.0),   # (x,y)
                                   std_caught_end*2,          # width
                                   1.0,          # height
                                   alpha=0.15,
                                   )
                 )
    ax.plot([0, 0.9], [0.333, 0.333], 'g--', linewidth=1.0)

    ytick = np.array(np.arange(0, 1.01, 0.2))
    ytick=np.append(ytick, [0.3333])
    ytick = np.sort(ytick)
    pp.yticks(ytick)

    #pp.legend([maincurve, xonly, zonly, evenlysampled, minmax], ['2 dimensional model', 'X axis force only', 'Z axis force only', 'Sampled points', 'Minimal Maximal Force'], fontsize=20)
    pp.legend(bbox_to_anchor=(0.96, 0.84),
           bbox_transform=pp.gcf().transFigure, numpoints=1)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(17)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(17)

    formatter = FuncFormatter(to_percent)

    # Set the formatter
    pp.gca().yaxis.set_major_formatter(formatter)

    pp.show()