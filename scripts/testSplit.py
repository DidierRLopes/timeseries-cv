#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

#from TimeSeriesCrossValidation.splitTrain import split_train
from TimeSeriesCrossValidation.splitTrainVal import *

timeSeries = np.arange(31)
print(timeSeries)


X, y, Xcv, ycv = split_train_val_forwardChaining(timeSeries, n_steps_input=4, n_steps_forecast=3, n_steps_jump=3)
for j in np.arange(5):
    print("--------- SET %d ---------" % (j+1))
    print("X[%d] ="% (j+1), X[j])
    print("y[%d] ="% (j+1), y[j])
   
# Plot original time series in usage
fig, ax = plt.subplots()
ax.plot(timeSeries, len(timeSeries) * [len(X)-0.75], lw=3, c='k')
k = 1./max([len(x)+2 for x in X.values()])
numSets = len(X)
# Loop through train/val sets
for i in np.arange(numSets):
    sorted_arr = np.sort(np.array([x[0] for x in np.concatenate((X[i], Xcv[i]), axis=0)]))
    # Loop through training sets
    for j in np.arange(len(X[i])):
        level = sum(sorted_arr<X[i][j][0])
        plt.plot(X[i][j], len(X[i][j]) * [numSets-i-1-k*level], lw=3, c='blue')
        plt.plot([X[i][j][-1], y[i][j][0]], 2*[numSets-i-1-k*level], lw=1, c='blue')
        plt.plot(y[i][j], len(y[i][j]) * [numSets-i-1-k*level], lw=3, c='lightblue')
    # Loop through cross-validation sets
    for j in np.arange(len(Xcv[i])):
        level = sum(sorted_arr<Xcv[i][j][0])
        plt.plot(Xcv[i][j], len(Xcv[i][j]) * [numSets-i-1-k*level], lw=3, c='red')
        plt.plot([Xcv[i][j][-1], ycv[i][j][0]], 2*[numSets-i-1-k*level], lw=1, c='red')
        plt.plot(ycv[i][j], len(ycv[i][j]) * [numSets-i-1-k*level], lw=3, c='tomato')
    rect = patches.Rectangle((0, len(X)-i-1), len(timeSeries)-1, k*(-len(sorted_arr)+1), 
                             ls='-.', linewidth=1, edgecolor='k',facecolor='lavender')
    ax.add_patch(rect)
ax.grid(which='minor')
ax.set_yticks(np.arange(len(X), 0, -1)-1)
ax.set_yticklabels(np.arange(1, len(X)+1, 1))
ax.set_ylabel("Train/val set", size=16)
custom_lines = [Line2D([0], [0], color='k', lw=4),    patches.Patch(facecolor='lavender', edgecolor='k'),
                Line2D([0], [0], color='blue', lw=4), Line2D([0], [0], color='lightblue', lw=4),
                Line2D([0], [0], color='red', lw=4),  Line2D([0], [0], color='tomato', lw=4)]
ax.legend(custom_lines, ['Original TimeSeries', 'Train/val set', 
                         'X (Train input)', 'y (Train output)', 
                         'Xcv (Cross-validation input)', 'ycv (Cross-validation output)'], 
          loc='upper center', ncol=3, handleheight=2.4, labelspacing=0.05)
ax.set_ylim([-1.5, len(X)])
ax.set_title('Depict Cross-Validation Algorithm', size=18);
plt.show()
