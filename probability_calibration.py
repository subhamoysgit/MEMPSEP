"""
Created on Tue May 11, 2022 
last updated on Sat Apt 22,2023 12:39 PM MDT

@author: Subhamoy Chatterjee
"""


import numpy as np
import scipy.special as sc
import pandas as pd
import matplotlib.pyplot as plt


class probability_calibration():
    '''
    The purpose of this class is to convert a neural network outcome for
    binary classification to probability in non-parameteric manner

    Reference: https://www.dbmi.pitt.edu/wp-content/uploads/2022/10/Obtaining-well-calibrated-probabilities-using-Bayesian-binning.pdf

    '''
    def __init__(self, pred: float, gt: int, infer: float):

        '''
        Initialize an object of the probability_calibration class
        Parameters:
            pred (float):
                list of NN outcomes to fit the calibrator on

            gt (int):
                list of ground truths corresponding to pred data

            infer (float):
                list of NN outcomes to convert to calibrated probability

        '''

        self.pred = pred
        self.gt = gt
        self.infer = infer

    def histModel(self, nbins: int):

        '''
        Method to create a histogram binning model
        Parameters:
            nbins (int): number of bins

        returns:
            bin_edges (list): defining equal frequency histogram
            freq (list): defining transformed list of frequencies over the bins
            score (float): log score based on Beta distribution


        '''

        score = 0

        # bin edges for equal frequency histogram
        _, b_edges = pd.qcut(self.pred, nbins, duplicates='drop', retbins=True)
        nbins = len(list(b_edges))-1
        b_centers = 0.5*(b_edges[:nbins]+b_edges[1:])
        b_edges[nbins] = 1.01  # to include 1 in the bin
        hist, bin_edges = np.histogram(self.pred, bins=b_edges)
        freq, bc = [], []

        for i in range(nbins):

            # beta distribution parameters
            N1byB = 2/nbins
            Nb = hist[i]
            pb = b_centers[i]
            alpha_b = N1byB*pb
            beta_b = N1byB*(1-pb)
            m_b = np.sum(self.gt*(self.pred >= b_edges[i])*(self.pred
                                                            < b_edges[i+1])
                         )
            n_b = Nb - m_b
            p_sum = np.sum(self.pred*(self.pred >= b_edges[i])*(self.pred
                                                                < b_edges[i+1])
                           )
            p0 = (p_sum+pb)/(m_b+1)

            # binning model log score with gamma function
            t1 = sc.gammaln(N1byB) - sc.gammaln(Nb+N1byB) 
            t2 = sc.gammaln(m_b+alpha_b) - sc.gammaln(alpha_b)
            t3 = sc.gammaln(n_b+beta_b) - sc.gammaln(beta_b)
            score = score + t1 + t2 + t3

            if Nb > 0:
                freq.append((m_b+p0)/(Nb+1))
                bc.append(pb)

        return score, bin_edges, freq

    def selectBins(self):

        '''
        Method to select histogram binning models

        returns: 
            bins_sel (list): list of selected bins for histogram models

        '''

        # find search range for nbins
        ll = len(self.gt)
        maxBinNo = int(np.min([np.round(ll/5),np.round(10*ll**(1/3))]))
        minBinNo = int(np.max([1, np.round(ll**(1/3)/10)]))

        # generate scores for the binning models
        ln_score, bins_sel = [], []
        for b in range(minBinNo, maxBinNo+1):
            s, _, _ = self.histModel(b)
            ln_score.append(s)
            bins_sel.append(b)

        # log(score) -> score
        ln_score = np.array(ln_score)
        score = np.exp(ln_score-np.min(ln_score)) 

        # select bins
        bins_sel = np.array(bins_sel)
        ss = np.argsort(score)[::-1]
        bins_sel = bins_sel[ss]  # restrict to top s scores the reduce run time

        return bins_sel

    def calibrateProbability(self, n_sel=-1):

        '''
        Method to calibate probability on test data

        Parameters:
            n_sel (int): select top n_sel bins based on beta distribution derived score

        returns: 
            p_out (list): list of calibrated probabilities from test data

        '''

        bins_sel = self.selectBins()[:n_sel]
        p_out = []

        bin_dict = {}

        for b in bins_sel:
            s, b_edges, freq = self.histModel(b)
            bin_dict[b, 's'] = s
            bin_dict[b, 'b_edges'] = b_edges
            bin_dict[b, 'freq'] = freq

        for i in range(len(self.infer)):

            score, po = [], []

            for b in bins_sel:
                s = bin_dict[b, 's']
                b_edges = bin_dict[b, 'b_edges']
                freq = bin_dict[b, 'freq']

                # convert NN inference to frequency based on bin location

                if len(freq) >= 2:
                    for kk in range(len(freq)):
                        if (self.infer[i] >= b_edges[kk]) and (self.infer[i] <
                                                               b_edges[kk+1]):
                            pp = freq[kk]
                            break

                else:
                    pp = freq[0]
                score.append(s)
                po.append(pp)

            # weighted average of all histogram model outcomes
            po = np.array(po)
            score = np.array(score)
            score = np.exp(score-np.min(score))
            p_out.append(np.sum(po*score)/np.sum(score))

        return p_out


def reliability_diagram(pred, gt):
    '''
    Parameters:
        pred (float):
            list of NN outcomes to fit the calibrator on

        gt (int):
            list of ground truths corresponding to pred data

    returns:
        probability (xx) and frequency list (mm1) for all the bins
        ece (float): expected calibration error


    '''
    n = 10
    be = np.linspace(0, n, n+1)*(1/n)
    num = n+1
    be[n] = 1.01
    xx = 0.5*(be[:num-1]+be[1:])

    m10 = np.array(pred)
    hist, _ = np.histogram(m10, bins=be)
    mm1, p_avg = [], []
    for i in range(num-1):
        s = np.sum(gt*(m10 >= be[i]) * (m10 < be[i+1]))
        s1 = np.sum((m10 >= be[i]) * (m10 < be[i+1]) * (m10))/hist[i]
        mm1.append(s)
        p_avg.append(s1)

    ind = np.where(hist > 0)
    mm1 = mm1/hist
    mm1 = np.asarray(mm1)
    p_avg = np.asarray(p_avg)
    ece = np.sum(np.abs(mm1[ind]-p_avg[ind])*hist[ind]/len(gt))

    return xx[ind], mm1[ind], ece


def main():
    ngen = 1000
    s = np.random.normal(0.5, 0.2, ngen)

    s[s < 0] = 0
    s[s > 1] = 1

    gt_p = list(s > 0.5)[:ngen//2]
    gt_i = list(s > 0.5)[ngen//2:]
    pred = list(s)[:ngen//2]
    infer = list(s)[ngen//2:]

    p = probability_calibration(pred, gt_p, infer)
    cal = p.calibrateProbability(n_sel=10)

    idx = np.argsort(infer)

    p1, freq1, ece1 = reliability_diagram(infer, gt_i)
    p2, freq2, ece2 = reliability_diagram(cal, gt_i)
    plt.subplot(1, 3, 1)
    plt.plot(p1, freq1)
    plt.plot([0, 1], [0, 1], '-r')
    plt.subplot(1, 3, 2)
    plt.plot(np.array(infer)[idx], np.array(cal)[idx])
    plt.subplot(1, 3, 3)
    plt.plot(p2, freq2)
    plt.plot([0, 1], [0, 1], '-r')
    plt.show()
    print(f'ECE before calibration {np.round(ece1, 4)*100} %')
    print(f'ECE after calibration {np.round(ece2, 4)*100} %')


if __name__ == '__main__':
    main()
