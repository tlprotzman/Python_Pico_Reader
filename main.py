# \author Skipper Kagamaster
# \date 03/20/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

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


time_start = time.perf_counter()

dataDirect = r"E:\2019Picos\14p5GeV\Runs"
saveDirect = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\Protons"
finalDirect = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\Protons\YuCuts"

os.chdir(saveDirect)
bad_runs = np.loadtxt("badlist.txt").astype(int)

# Yu's cuts for RefMult3:
RefCuts = np.asarray((0, 10, 21, 41, 72, 118, 182, 270, 392, 472))

# This is to save our protons.
protons = []
antiprotons = []
refmult3 = []

# To display the proton selection.
a = 500
counts = np.zeros((2, a, a))
bins = np.zeros((2, 2, a+1))

# The following are all for the average value for an event.
# The second row is for the error.
AveRefMult3 = [[], []]
AveVz = [[], []]
AvePt = [[], []]
AveEta = [[], []]
AveVr = [[], []]
AveZdcX = [[], []]
AvePhi = [[], []]
AveDca = [[], []]

# Arrays to hold our histogram data for before and after cut analysis.
a = 1000
b = 161
c = 86
d = 101
v_z = np.zeros((2, a))  # bins included in this one
v_r = np.zeros((a, a))
v_r_bins = np.zeros((2, a))
RefMult_TOFMult = np.zeros((1700, a))
RefMult_TOFMult_bins = np.zeros((2, a))
RefMult_TOFMatch = np.zeros((a, a))
RefMult_TOFMatch_bins = np.zeros((2, a))
RefMult_BetaEta = np.zeros((400, a))
RefMult_BetaEta_bins = np.zeros((2, a))
p_t = np.zeros((2, a))  # bins included in this one
phi = np.zeros((2, a))  # bins included in this one
dca = np.zeros((2, a))  # bins included in this one
eta = np.zeros((2, a))  # bins included in this one
nHitsFit_charge = np.zeros((2, b))  # bins included in this one
nHits_dEdX = np.zeros((2, c))  # bins included in this one
m_pq = np.zeros((a, a))
m_pq_bins = np.zeros((2, a))
beta_p = np.zeros((a, a))
beta_p_bins = np.zeros((2, a))
dEdX_pq = np.zeros((a, a))
dEdX_pq_bins = np.zeros((2, a))
v_z_cuts = np.zeros((2, a))  # bins included in this one
v_r_cuts = np.zeros((a, a))
v_r_bins_cuts = np.zeros((2, a))
RefMult_TOFMult_cuts = np.zeros((1700, a))
RefMult_TOFMult_bins_cuts = np.zeros((2, a))
RefMult_TOFMatch_cuts = np.zeros((a, a))
RefMult_TOFMatch_bins_cuts = np.zeros((2, a))
RefMult_BetaEta_cuts = np.zeros((400, a))
RefMult_BetaEta_bins_cuts = np.zeros((2, a))
p_t_cuts = np.zeros((2, a))  # bins included in this one
phi_cuts = np.zeros((2, a))  # bins included in this one
dca_cuts = np.zeros((2, a))  # bins included in this one
eta_cuts = np.zeros((2, a))  # bins included in this one
nHitsFit_charge_cuts = np.zeros((2, b))  # bins included in this one
nHits_dEdX_cuts = np.zeros((2, c))  # bins included in this one
m_pq_cuts = np.zeros((a, a))
m_pq_bins_cuts = np.zeros((2, a))
beta_p_cuts = np.zeros((a, a))
beta_p_bins_cuts = np.zeros((2, a))
dEdX_pq_cuts = np.zeros((a, a))
dEdX_pq_bins_cuts = np.zeros((2, a))

os.chdir(dataDirect)
r = len(os.listdir())
r_ = 1300  # For loop cutoff (to test on smaller batches).
count = 1

for file in sorted(os.listdir()):
    # This is to omit all runs marked "bad."
    run_num = file[:-5]
    # This cuts off the loop for testing.
    """
    if count < r_:
        continue
    """
    r = 200
    if count > r:
        break

    # This is just to show how far along the script is.
    if count % 2 == 0:
        print("Working on " + str(count) + " of " + str(r) + ".")
    # From Yu's list of bad runs.
    if int(run_num) in bad_runs:
        print("Run", run_num, "skipped for being marked bad.")
        print("Bad, I tell you. Bad!!")
        r -= 1
        continue
    # Yu's cutoff for nSigmaProton and dE/dx calibration issues.
    if int(run_num) > 20118040:
        print("Over the threshold Yu had set for display (20118040).")
        r -= 1
        break
    # Import data from the pico.
    try:
        pico = pr.PicoDST()
        pico.import_data(file)
        pico.event_cuts()
        pico.calibrate_nsigmaproton()
        pico.track_cuts_test()

        # Histogramming
        dedx = ak.to_numpy(ak.flatten(pico.dedx))
        p_g = ak.to_numpy(ak.flatten(pico.p_g))
        charge = ak.to_numpy(ak.flatten(pico.charge))
        counts1, binsX1, binsY1 = np.histogram2d(dedx, p_g * charge,
                                                 bins=a, range=((0, 20), (-5, 5)))
        counts2, binsX2, binsY2 = np.histogram2d(pico.dedx_histo, pico.p_g_histo * pico.charge_histo,
                                                 bins=a, range=((0, 20), (-5, 5)))
        counts += (counts1, counts2)
        bins = ((binsX1, binsY1), (binsX2, binsY2))

        # Protons
        protons.append(ak.to_numpy(pico.protons))
        antiprotons.append(ak.to_numpy(pico.antiprotons))
        refmult3.append(ak.to_numpy(pico.refmult3))
    except Exception as e:  # For any issues that might pop up.
        print(e.__class__, "occurred in", run_num)
        count += 1
        continue
    count += 1

fig, ax = plt.subplots(2, figsize=(16, 9), constrained_layout=True)
titles = ["After QA Cuts", "After Proton Selection"]
for i in range(2):
    X, Y = np.meshgrid(bins[i][1], bins[1][0])
    imDp = ax[i].pcolormesh(X, Y, counts[i], cmap="jet", norm=colors.LogNorm())
    ax[i].set_xlabel(r"p*q $(\frac{GeV}{c})$", fontsize=10)
    ax[i].set_ylabel(r"$\frac{dE}{dX} (\frac{KeV}{cm})$", fontsize=10)
    ax[i].set_title(titles[i], fontsize=20)
    fig.colorbar(imDp, ax=ax[i])
plt.show()
plt.close()

protons = np.asarray(np.hstack(protons).flatten())
antiprotons = np.asarray(np.hstack(antiprotons).flatten())
refmult3 = np.asarray(np.hstack(refmult3).flatten())
print("Events: " + str(len(protons)))
cumulants_cbwc, cumulants_no_cbwc, cumulants_all, ref_set = pr.cbwc(protons, antiprotons, refmult3)
cumulant_names = (r"$\mu$", r"$\sigma^2$", "Skew", "Kurtosis")
RefCuts_labels = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                  "50-60%", "60-70%", "70-80%", "80-100%"]
# Yu's cumulant values (CBWC, uncorrected for efficiencies).
C = [[0.5, 0.95, 1.58, 2.88, 4.61, 7.09, 10.59, 14.0, 16.78],
     [0.17, 0.78, 1.65, 2.99, 4.85, 7.53, 11.26, 14.89, 17.79],
     [0.17, 0.63, 1.30, 2.39, 3.87, 6.10, 9.10, 11.96, 13.79],
     [0.51, 1.10, 1.78, 2.54, 4.24, 6.53, 9.75, 13.05, 15.34]]

# Plot of my cumulants, cbwc cumulants, and Yu's cbwc cumulants.
fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for i in range(2):
    for j in range(2):
        x = i * 2 + j
        #ax[i, j].plot(RefCuts, cumulants_no_cbwc[x], c='b', marker="o", mfc='red', lw=0, mew=2,
        #              label="No CBWC", alpha=0.5)
        ax[i, j].plot(RefCuts, cumulants_cbwc[x], c='black', marker="o", mfc='red', lw=0, mew=2,
                      label="CBWC", alpha=0.5)
        ax[i, j].plot(RefCuts[1:], C[x], c='r', marker="o", mfc='green', lw=0, mew=2,
                      label="Yu's Values", alpha=0.5)
        ax[i, j].set_title(cumulant_names[x])
        #ax[i, j].set_xticks(RefCuts[::-1])
        #ax[i, j].set_xticklabels(RefCuts_labels, rotation=45)
        ax[i, j].set_xlabel("RefMult3")
        ax[i, j].grid(True)
        ax[i, j].legend()
plt.show()
plt.close()

# Raw cumulants plot by refmult3.
fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for i in range(2):
    for j in range(2):
        x = i * 2 + j
        ax[i, j].plot(ref_set, np.divide(cumulants_all[x+1], cumulants_all[0]), c='b', label="Raw Cumulants")
        ax[i, j].set_title(cumulant_names[x])
        ax[i, j].set_xlabel("RefMult3")
        ax[i, j].grid(True)
plt.suptitle("Raw Cumulants")
plt.show()
plt.close()

# Save the protons! Those poor protons.
os.chdir(finalDirect)
np.save("protons_yu.npy", protons)
np.save("antiprotons_yu.npy", antiprotons)
np.save("refmult3_yu.npy", refmult3)
