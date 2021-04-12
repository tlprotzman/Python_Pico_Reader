# \author Skipper Kagamaster
# \date 03/20/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

# Not all are in use at present.
import sys
import os
import typing
import logging
import numpy as np
import awkward as ak
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pico_reader as pr
from scipy.stats import skew, kurtosis

# Directory where your picos live.
dataDirect = r"E:\2019Picos\14p5GeV\Runs"
os.chdir(dataDirect)
r = len(os.listdir())
r_ = 1300  # For loop cutoff (to test on later picos).
count = 1

# Just a simple histogram for testing (using pg vs dedx).
a = 500
counts = np.zeros((2, a, a))
bins = np.zeros((2, 2, a+1))

# Loop over the picos.
for file in sorted(os.listdir()):
    run_num = file[:-5]
    # This cuts off the loop for testing.
    """
    if count < r_:
        continue
    """
    r = 20
    if count > r:
        break

    # This is just to show how far along the script is.
    if count % 5 == 0:
        print("Working on " + str(count) + " of " + str(r) + ".")
    
    # Import data from the pico.
    try:
        pico = pr.PicoDST()
        pico.import_data(file)
        # Pre event cut histogram of pq vs dedx
        counts1, binsX1, binsY1 = np.histogram2d(ak.to_numpy(ak.flatten(pico.dedx)), 
                                                 ak.to_numpy(ak.flatten(pico.p_g)) * ak.to_numpy(ak.flatten(pico.charge)),
                                                 bins=a, range=((0, 20), (-5, 5)))
        # Event cuts and the same histogram after.
        pico.event_cuts()
        counts2, binsX2, binsY2 = np.histogram2d(ak.to_numpy(ak.flatten(pico.dedx)), 
                                                 ak.to_numpy(ak.flatten(pico.p_g)) * ak.to_numpy(ak.flatten(pico.charge)),
                                                 bins=a, range=((0, 20), (-5, 5)))
        counts += (counts1, counts2)
        bins = ((binsX1, binsY1), (binsX2, binsY2))
    except Exception as e:  # For any issues that might pop up.
        print(e.__class__, "occurred in", run_num)
        count += 1
        continue
    count += 1

fig, ax = plt.subplots(2, figsize=(16, 9), constrained_layout=True)
titles = ["Before Event Cuts", "After Event Cuts"]
for i in range(2):
    X, Y = np.meshgrid(bins[i][1], bins[1][0])
    imDp = ax[i].pcolormesh(X, Y, counts[i], cmap="jet", norm=colors.LogNorm())
    ax[i].set_xlabel(r"p*q $(\frac{GeV}{c})$", fontsize=10)
    ax[i].set_ylabel(r"$\frac{dE}{dX} (\frac{KeV}{cm})$", fontsize=10)
    ax[i].set_title(titles[i], fontsize=20)
    fig.colorbar(imDp, ax=ax[i])
plt.show()
plt.close()
