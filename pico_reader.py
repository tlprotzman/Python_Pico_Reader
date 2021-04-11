#
# \PicoDst reader for use in finding proton kurtosis
#
# \author Skipper KAgamaster
# \date 03/19/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema as arex
from scipy.signal import savgol_filter as sgf
from scipy.stats import skew, kurtosis
import uproot as up
import awkward as ak
import time
import logging

# Speed of light, in m/s
c = 299792458
# Proton mass, in GeV
mp = 0.9382720813


def index_cut(a, *args):
    for arg in args:
        arg = arg[a]
        yield arg


# TODO Check to see that these results are reasonable. Graph against p_t.
def rapidity(p_z):
    e_p = np.power(np.add(mp**2, np.power(p_z, 2)), 1/2)
    e_m = np.subtract(mp**2, np.power(p_z, 2))
    e_m = ak.where(e_m < 0.0, 0.0, e_m)  # to avoid imaginary numbers
    e_m = np.power(e_m, 1/2)
    e_m = ak.where(e_m == 0.0, 1e-10, e_m)  # to avoid infinities
    y = np.multiply(np.log(np.divide(e_p, e_m)), 1/2)
    return y


def cbwc(proton, antiproton, refmult3, RefMult=np.asarray((0, 10, 21, 41, 72, 118, 182, 270, 392, 472))):
    protons = [[], [], [], []]
    protons[0].append(ak.to_numpy(proton))
    protons[1].append(ak.to_numpy(antiproton))
    protons[2].append(ak.to_numpy(refmult3))
    protons = np.asarray(protons)
    protons[0] = np.hstack(protons[0]).flatten()
    protons[1] = np.hstack(protons[1]).flatten()
    protons[2] = np.hstack(protons[2]).flatten()
    protons[3] = np.subtract(protons[0], protons[1])
    protons = np.asarray(protons)
    cumulants_no_cbwc = [[], [], [], []]
    cumulants_cbwc = [[], [], [], []]
    ref_set = np.unique(protons[2])
    # ref_set = np.linspace(0, np.max(protons[2]), np.max(protons[2])+1).astype(int)
    cumulants_all = np.zeros((5, len(ref_set)))

    prodists = []
    cumulants = [[], [], [], [], []]
    for i in range(len(ref_set)):
        prodists.append([])
        index = np.where(protons[2] == ref_set[i])
        prodists[i].append(protons[3][index])
        cumulants[0].append(len(protons[3][index]))
        cumulants[1].append(cumulants[0][i] * np.mean(protons[3][index]))
        cumulants[2].append(cumulants[0][i] * np.var(protons[3][index]))
        cumulants[3].append(cumulants[0][i] * skew(protons[3][index]) * np.power(np.sqrt(np.var(protons[3][index])), 3))
        cumulants[4].append(cumulants[0][i] * kurtosis(protons[3][index]) * np.power(np.var(protons[3][index]), 2))
    cumulants = np.asarray(cumulants)
    # Now to apply the cbwc (and get the uncorrected values for comparison).
    for i in range(len(RefMult) - 1):
        for j in range(4):
            index = np.where((ref_set >= RefMult[i]) & (ref_set < RefMult[i+1]))
            cumulants_cbwc[j].append(np.divide(np.sum(cumulants[j + 1][index]),
                                               np.sum(cumulants[0][index])))
            no_cbwc = np.divide(cumulants[j+1][index], cumulants[0][index])
            if j == 0:
                cumulants_no_cbwc[j].append(np.mean(no_cbwc))
            elif j == 1:
                cumulants_no_cbwc[j].append(np.var(no_cbwc))
            elif j == 2:
                cumulants_no_cbwc[j].append(skew(no_cbwc) * np.power(np.sqrt(np.var(no_cbwc)), 3))
            elif j == 3:
                cumulants_no_cbwc[j].append(kurtosis(no_cbwc) * np.power(np.var(no_cbwc), 2))
    for j in range(4):
        index = np.where(ref_set >= RefMult[len(RefMult) - 1])
        cumulants_cbwc[j].append(np.divide(np.sum(cumulants[j + 1][index]),
                                           np.sum(cumulants[0][index])))
        no_cbwc = np.divide(cumulants[j + 1][index], cumulants[0][index])
        if j == 0:
            cumulants_no_cbwc[j].append(np.mean(no_cbwc))
        elif j == 1:
            cumulants_no_cbwc[j].append(np.var(no_cbwc))
        elif j == 2:
            cumulants_no_cbwc[j].append(skew(no_cbwc) * np.power(np.sqrt(np.var(no_cbwc)), 3))
        elif j == 3:
            cumulants_no_cbwc[j].append(kurtosis(no_cbwc) * np.power(np.var(no_cbwc), 2))
    """
    for i in ref_set:
        index = (protons[2] == i)
        if True not in index:
            continue
        else:
            pl = len(protons[3][index])
            cumulants_all[0][i] = pl
            cumulants_all[1][i] = np.mean(protons[3][index])
            cumulants_all[2][i] = np.var(protons[3][index])
            cumulants_all[3][i] = skew(protons[3][index]) * np.power(np.sqrt(np.var(protons[3][index])), 3)
            cumulants_all[4][i] = kurtosis(protons[3][index]) * np.power(np.var(protons[3][index]), 2)
    for i in range(len(RefMult) - 1):
        for j in range(4):
            index = ((ref_set >= RefMult[i]) & (ref_set < RefMult[i+1]))
            cumulants_cbwc[j].append(np.divide(np.sum(cumulants_all[j + 1][RefMult[i]:RefMult[i+1]]),
                                               np.sum(cumulants_all[0][RefMult[i]:RefMult[i+1]])))
            index = ((protons[2] >= RefMult[i]) & (protons[2] < RefMult[i + 1]))
            cumulants_no_cbwc[j].append(np.mean(protons[3][RefMult[i]:RefMult[i+1]]))
    for j in range(4):
        # cumulants_cbwc[j].append(np.divide(np.sum(np.multiply(cumulants_all[j+1][RefCuts[len(RefCuts)-1]:]),
        #                                          cumulants_all[0][RefCuts[len(RefCuts)-1]:]),
        #                                          np.sum(cumulants_all[0][RefCuts[len(RefCuts)-1]:])))
        index = (ref_set >= RefMult[len(RefMult) - 1])
        cumulants_cbwc[j].append(np.divide(np.sum(cumulants_all[j + 1][RefMult[len(RefMult)-1]:]),
                                           np.sum(cumulants_all[0][RefMult[len(RefMult)-1]:])))
        index = (protons[2] >= RefMult[-1])
        cumulants_no_cbwc[j].append(np.mean(protons[3][RefMult[len(RefMult)-1]:]))
    """
    return cumulants_cbwc, cumulants_no_cbwc, cumulants, ref_set


class PicoDST:
    """This class makes the PicoDST from the root file, along with
    all of the observables I use for proton kurtosis analysis."""

    def __init__(self):
        """This defines the variables we'll be using
        in the class."""
        self.data: bool
        self.v_x = None
        self.v_y = None
        self.v_z = None
        self.v_r = None
        self.refmult3 = None
        self.tofmult = None
        self.tofmatch = None
        self.bete_eta_1 = None
        self.p_t = None
        self.p_g = None
        self.phi = None
        self.dca = None
        self.eta = None
        self.nhitsfit = None
        self.nhitsdedx = None
        self.m_2 = None
        self.charge = None
        self.beta = None
        self.dedx = None
        self.zdcx = None
        self.rapidity = None
        self.nhitsmax = None
        self.nsigma_proton = None
        self.tofpid = None
        self.protons = None
        self.antiprotons = None
        self.dedx_histo = None
        self.p_g_histo = None
        self.charge_histo = None

    def import_data(self, data_in):
        """This imports the data. You must have the latest versions
        of uproot and awkward installed on your machine (uproot4 and
        awkward 1.0 as of the time of this writing).
        Use pip install uproot awkward.
        Args:
            data_in (str): The path to the picoDst ROOT file"""
        try:
            data = up.open(data_in)["PicoDst"]
            # Make vertices
            self.v_x = ak.to_numpy(ak.flatten(data["Event"]["Event.mPrimaryVertexX"].array()))
            self.v_y = ak.to_numpy(ak.flatten(data["Event"]["Event.mPrimaryVertexY"].array()))
            self.v_z = ak.to_numpy(ak.flatten(data["Event"]["Event.mPrimaryVertexZ"].array()))
            self.v_r = np.sqrt(np.power(np.subtract(np.mean(self.v_x), self.v_x), 2) +
                               np.power(np.subtract(np.mean(self.v_y), self.v_y), 2))
            self.zdcx = ak.to_numpy(ak.flatten(data["Event"]["Event.mZDCx"].array()))
            self.refmult3 = ak.to_numpy(ak.flatten(data["Event"]["Event.mRefMult3PosEast"].array() +
                                                   data["Event"]["Event.mRefMult3PosWest"].array() +
                                                   data["Event"]["Event.mRefMult3NegEast"].array() +
                                                   data["Event"]["Event.mRefMult3NegWest"].array()))
            self.tofmult = ak.to_numpy(ak.flatten(data["Event"]["Event.mbTofTrayMultiplicity"].array()))
            self.tofmatch = ak.to_numpy(ak.flatten(data["Event"]["Event.mNBTOFMatch"].array()))
            # Make p_g and p_t
            p_x = data["Track"]["Track.mGMomentumX"].array()
            p_y = data["Track"]["Track.mGMomentumY"].array()
            p_y = ak.where(p_y == 0.0, 1e-10, p_y)  # to avoid infinities
            p_z = data["Track"]["Track.mGMomentumZ"].array()
            self.p_t = np.sqrt(np.power(p_x, 2) + np.power(p_y, 2))
            self.p_g = np.sqrt((np.power(p_x, 2) + np.power(p_y, 2) + np.power(p_z, 2)))
            self.eta = np.arcsinh(np.divide(p_z, self.p_t))
            self.rapidity = rapidity(p_z)
            # Make dca
            dca_x = data["Track"]["Track.mOriginX"].array() - self.v_x
            dca_y = data["Track"]["Track.mOriginY"].array() - self.v_y
            dca_z = data["Track"]["Track.mOriginZ"].array() - self.v_z
            self.dca = np.sqrt((np.power(dca_x, 2) + np.power(dca_y, 2) + np.power(dca_z, 2)))
            self.nhitsdedx = data["Track"]["Track.mNHitsDedx"].array()
            self.nhitsfit = data["Track"]["Track.mNHitsFit"].array()
            self.nhitsmax = data["Track"]["Track.mNHitsMax"].array()
            self.nhitsmax = ak.where(self.nhitsmax == 0, 1e-10, self.nhitsmax)  # to avoid infinities
            self.dedx = data["Track"]["Track.mDedx"].array()
            self.nsigma_proton = data["Track"]["Track.mNSigmaProton"].array()
            self.charge = ak.where(self.nhitsfit >= 0, 1, -1)
            self.beta = data["BTofPidTraits"]["BTofPidTraits.mBTofBeta"].array()/20000.0
            self.tofpid = data["BTofPidTraits"]["BTofPidTraits.mTrackIndex"].array()
            # Make B_n_1
            be1_1 = ak.sum(ak.where(self.beta > 0.1, 1, 0), axis=-1)
            be1_2 = ak.sum(ak.where(np.absolute(self.eta) < 1.0, 1, 0), axis=-1)
            be1_3 = ak.sum(ak.where(self.dca < 3.0, 1, 0), axis=-1)
            be1_4 = ak.sum(ak.where(np.absolute(self.nhitsfit) > 10, 1, 0), axis=-1)
            self.bete_eta_1 = be1_1 + be1_2 + be1_3 + be1_4
            # Make m^2
            p_squared = np.power(self.p_g[self.tofpid], 2)
            b_squared = np.power(self.beta, 2)
            b_squared = ak.where(b_squared == 0.0, 1e-10, b_squared)  # to avoid infinities
            g_squared = np.subtract(1, b_squared)
            self.m_2 = np.divide(np.multiply(p_squared, g_squared), b_squared)
            # Make phi.
            o_x = data["Track"]["Track.mOriginX"].array()
            o_y = data["Track"]["Track.mOriginY"].array()
            self.phi = np.arctan2(o_y, o_x)

            # print("PicoDst " + data_in[-13:-5] + " loaded.")

        except ValueError:  # Skip empty picos.
            print("ValueError at: " + data_in)  # Identifies the misbehaving file.
        except KeyError:  # Skip non empty picos that have no data.
            print("KeyError at: " + data_in)  # Identifies the misbehaving file.

    def event_cuts(self, v_r_cut=2.0, v_z_cut=30.0, tofmult_refmult=np.array([[2.536, 200], [1.352, -54.08]]),
                   tofmatch_refmult=np.array([0.239, -14.34]), beta_refmult=np.array([0.447, -17.88])):
        """This is used to make event cuts.
        """
        index = ((np.absolute(self.v_z) <= v_z_cut) & (self.v_r <= v_r_cut) &
                 (self.tofmult <= (np.multiply(tofmult_refmult[0][0], self.refmult3) + tofmult_refmult[0][1])) &
                 (self.tofmult >= (np.multiply(tofmult_refmult[1][0], self.refmult3) + tofmult_refmult[1][1])) &
                 (self.tofmatch >= (np.multiply(tofmatch_refmult[0], self.refmult3) + tofmatch_refmult[1])) &
                 (self.bete_eta_1 >= (np.multiply(beta_refmult[0], self.refmult3) + beta_refmult[1])))

        self.v_x, self.v_y, self.v_z, self.v_r, self.refmult3, self.tofmult, self.tofmatch, self.bete_eta_1, \
            self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx, self.m_2, \
            self.charge, self.beta, self.dedx, self.zdcx, self.rapidity, self.nhitsmax, self.nsigma_proton, \
            self.tofpid = \
            index_cut(index, self.v_x, self.v_y, self.v_z, self.v_r, self.refmult3, self.tofmult, self.tofmatch,
                      self.bete_eta_1, self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit,
                      self.nhitsdedx, self.m_2, self.charge, self.beta, self.dedx, self.zdcx, self.rapidity,
                      self.nhitsmax, self.nsigma_proton, self.tofpid)

        # print("Event cuts complete.")

    def track_cuts_low(self, nhitsdedx_cut=5, nhitsfit_cut=20, ratio_cut=0.52, dca_cut=1.0,
                       p_t_cut=np.array((0.2, 10.0)), rapid_cut=0.5):
        """This is used to make track cuts for low p_t tracks.
        """
        index = ((self.nhitsdedx > nhitsdedx_cut) & (np.absolute(self.nhitsfit) > nhitsfit_cut) &
                 (np.divide(np.absolute(self.nhitsfit), self.nhitsmax) > ratio_cut) &
                 (self.dca < dca_cut) & (self.p_t > p_t_cut[0]) &
                 (self.p_t < p_t_cut[1]) & (np.absolute(self.rapidity) <= rapid_cut) &
                 (np.absolute(self.nsigma_proton) <= 2000.0))
        charge, p_t, p_g, dedx = \
            index_cut(index, self.charge, self.p_t, self.p_g, self.dedx)
        index1 = ((p_t < 0.8) & (p_g <= 1.0))
        prot1 = ak.sum(ak.where(charge[index1] >= 0, 1, 0), axis=-1)
        aprot1 = ak.sum(ak.where(charge[index1] < 0, 1, 0), axis=-1)
        dedx_histo = [dedx[index1]]
        p_g_histo = [p_g[index1]]
        charge_histo = [charge[index1]]
        charge, p_t, p_g, dedx, nhitsdedx, nhitsfit, nhitsmax, dca, rapidit, nsigma_proton = \
            index_cut(self.tofpid, self.charge, self.p_t, self.p_g, self.dedx, self.nhitsdedx,
                      self.nhitsfit, self.nhitsmax, self.dca, self.rapidity, self.nsigma_proton)
        index = ((nhitsdedx > nhitsdedx_cut) & (np.absolute(nhitsfit) > nhitsfit_cut) &
                 (np.divide(np.absolute(nhitsfit), nhitsmax) > ratio_cut) &
                 (dca < dca_cut) & (p_t > p_t_cut[0]) &
                 (p_t < p_t_cut[1]) & (np.absolute(rapidit) <= rapid_cut) &
                 (np.absolute(nsigma_proton) <= 2000.0))
        charge, p_t, p_g, dedx, m_2 = \
            index_cut(index, charge, p_t, p_g, dedx, self.m_2)
        index2 = ((p_t >= 0.8) & (p_g <= 3.0) &
                  (m_2 >= 0.6) & (m_2 <= 1.2))
        dedx_histo.append(dedx[index2])
        self.dedx_histo = ak.to_numpy(ak.flatten(ak.Array(dedx_histo), axis=None))
        p_g_histo.append(p_g[index2])
        self.p_g_histo = ak.to_numpy(ak.flatten(ak.Array(p_g_histo), axis=None))
        charge_histo.append(charge[index2])
        self.charge_histo = ak.to_numpy(ak.flatten(ak.Array(charge_histo), axis=None))
        prot2 = ak.sum(ak.where(charge[index2] >= 0, 1, 0), axis=-1)
        aprot2 = ak.sum(ak.where(charge[index2] < 0, 1, 0), axis=-1)
        self.protons = prot1+prot2
        self.antiprotons = aprot1+aprot2
        # print("Track cuts low complete.")

    def track_cuts_test(self, nhitsdedx_cut=5, nhitsfit_cut=20, ratio_cut=0.52, dca_cut=1.0,
                        p_t_cut=np.array((0.2, 10.0)), rapid_cut=0.5):
        """This is used to make track cuts for low p_t tracks.
        """
        charge, p_t, p_g, dedx, dca, rapidit, nsigma_proton, nhitsdedx, nhitsfit, nhitsmax = \
            index_cut(self.tofpid, self.charge, self.p_t, self.p_g, self.dedx,
                      self.dca, self.rapidity, self.nsigma_proton, self.nhitsdedx,
                      self.nhitsfit, self.nhitsmax)
        index = ((nhitsdedx > nhitsdedx_cut) & (np.absolute(nhitsfit) > nhitsfit_cut) &
                 (np.divide(np.absolute(nhitsfit), nhitsmax) > ratio_cut) &
                 (dca < dca_cut) & (p_t > p_t_cut[0]) & (p_t < p_t_cut[1]) &
                 (np.absolute(rapidit) <= rapid_cut) & (np.absolute(nsigma_proton) <= 2000.0))
        charge, p_t, p_g, dedx, m_2 = \
            index_cut(index, charge, p_t, p_g, dedx, self.m_2)
        index1 = ((p_t < 0.8) & (p_g <= 1.0))
        prot1 = ak.sum(ak.where(charge[index1] >= 0, 1, 0), axis=-1)
        aprot1 = ak.sum(ak.where(charge[index1] < 0, 1, 0), axis=-1)
        dedx_histo = [dedx[index1]]
        p_g_histo = [p_g[index1]]
        charge_histo = [charge[index1]]
        index2 = ((p_t >= 0.8) & (p_g <= 3.0) &
                  (m_2 >= 0.6) & (m_2 <= 1.2))
        prot2 = ak.sum(ak.where(charge[index2] >= 0, 1, 0), axis=-1)
        aprot2 = ak.sum(ak.where(charge[index2] < 0, 1, 0), axis=-1)
        dedx_histo.append(dedx[index2])
        self.dedx_histo = ak.to_numpy(ak.flatten(ak.Array(dedx_histo), axis=None))
        p_g_histo.append(p_g[index2])
        self.p_g_histo = ak.to_numpy(ak.flatten(ak.Array(p_g_histo), axis=None))
        charge_histo.append(charge[index2])
        self.charge_histo = ak.to_numpy(ak.flatten(ak.Array(charge_histo), axis=None))
        self.protons = prot1+prot2
        self.antiprotons = aprot1+aprot2
        # print("Track cuts low complete.")

    def track_cuts_high(self, nhitsdedx_cut=5, nhitsfit_cut=20, ratio_cut=0.52, dca_cut=1.0,
                        p_t_cut=np.array((0.2, 10.0)), rapid_cut=0.5):
        self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx, \
            self.charge, self.dedx, self.rapidity, self.nhitsmax, self.nsigma_proton, = \
            index_cut(self.tofpid, self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit,
                      self.nhitsdedx, self.charge, self.dedx, self.rapidity, self.nhitsmax,
                      self.nsigma_proton)
        index = ((self.nhitsdedx > nhitsdedx_cut) & (self.nhitsfit > nhitsfit_cut) &
                 (np.divide(self.nhitsfit, self.nhitsmax) > ratio_cut) &
                 (self.dca < dca_cut) & (self.p_t >= p_t_cut[0]) &
                 (self.p_t <= p_t_cut[1]) & (self.rapidity <= rapid_cut))
        self.nsigma_proton, self.m_2 = self.nsigma_proton[index], self.m_2[index]
        print("Track cuts high complete.")

    def calibrate_nsigmaproton(self):
        # Calibration of nSigmaProton for 0.0 < p_t < 0.8 (assumed 0 otherwise)
        # First, we'll separate it into discrete groupings of p_t.
        sig_length = 19
        nsigmaproton_p = []
        p_t_n = ak.to_numpy(ak.flatten(self.p_t))
        ns_n = ak.to_numpy(ak.flatten(self.nsigma_proton))
        nsigmaproton_p.append(ns_n[(p_t_n <= 0.2)])
        for k in range(2, sig_length+1):
            nsigmaproton_p.append(ns_n[((p_t_n > 0.1*k) & (p_t_n <= 0.1*(k+1)))])
        nsigmaproton_p = np.asarray(nsigmaproton_p)

        # Now to find the peak of the proton distribution. I'm going to try smoothing the
        # distributions, then finding the inflection points via a second order derivative.
        sig_means = []
        p_count = 0
        for dist in nsigmaproton_p:
            counter, bins = np.histogram(dist, range=(-10000, 10000), bins=200)
            sgf_proton_3 = sgf(counter, 45, 2)
            sgf_proton_3_2 = sgf(sgf_proton_3, 45, 2, deriv=2)
            infls = bins[:-1][np.where(np.diff(np.sign(sgf_proton_3_2)))[0]]
            sig_mean = 0
            if infls.size >= 2:
                infls_bounds = np.sort(np.absolute(infls))
                first = infls[np.where(np.absolute(infls) == infls_bounds[0])[0][0]]
                second = infls[np.where(np.absolute(infls) == infls_bounds[1])[0][0]]
                if first > second:
                    sig_mean = first-(first-second)/2
                else:
                    sig_mean = second-(second-first)/2
            if p_count >= 10:
                sig_mean = 0
            sig_means.append(sig_mean)
            # The below is to check things; turned off if running over lots of files.
            """
            plt.plot(bins[:-1], counter, c="blue", lw=2, label="Raw")
            plt.plot(bins[:-1], sgf_proton_3, c="red", lw=1, label="Smoothed")
            plt.plot(bins[:-1], sgf_proton_3_2, c="green", label="2nd derivative")
            for k, infl in enumerate(infls, 1):
                plt.axvline(x=infl, color='k', label=f'Inflection Point {k}')
            plt.axvline(x=sig_mean, c="pink", label="nSigmaMean")
            p_title = r'$p_T$ <= ' + str(0.1*(p_count+2))
            plt.title(p_title)
            plt.legend()
            plt.show()
            """
            p_count += 1
        sig_means = np.asarray(sig_means)

        # Now to modify nSigmaProton to be the difference between the values and
        # the found means.
        self.nsigma_proton = ak.where(self.p_t <= 0.2, self.nsigma_proton-sig_means[0], self.nsigma_proton)
        for k in range(1, len(sig_means)):
            self.nsigma_proton = ak.where((self.p_t > 0.1*(k+1)) & (self.p_t <= 0.1*(k+2)),
                                           self.nsigma_proton-sig_means[k], self.nsigma_proton)
