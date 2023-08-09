import numpy as np
import pandas as pd
import random


def histedges_equal(data, nbin):
    '''
    Function to return edges of equal frequency histrograms bins
    Parameters:
        data (list): list of datapoints e.g. flare peak
        nbin (int): number of bins

    returns:
        bin_edges (numpy array): defining equal frequency histogram
    '''
    npt = len(data)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(data))


def opt_rand_subset(list_date, length, ens_num):
    '''
    Function to create random subsets with minimal intersection
    @author: Subhamoy Chatterjee
    Parameters:
        list_date (list): list of dates
        length (int): size of the subset
        ens_num (int): number of subsets to be drawn randomly from list_date

    returns:
        l (list): list of subsets
        membership (list): number of selection of each date from list_date
                           over the random subsets
    '''
    random.seed(23)
    membership_bound = np.ceil(ens_num * length / len(list_date))
    for _ in range(100):
        num = []
        l = [[0]*length]
        membership = [0]*len(list_date)
        lst = list_date.copy()
        for i in range(ens_num):
            random.shuffle(lst)
            lst_m = lst[:length]
            l.append(lst_m)
            num.append(len(lst_m))
            for n in list_date:
                if n in lst_m:
                    ind = list_date.index(n)
                    membership[ind] = membership[ind] + 1
                    if membership[ind] >= membership_bound:
                        # remove a date from list when its ocurrence across
                        # subsets exceeds membership_bound
                        lst = list(set(lst)-{list_date[ind]})

        if np.min(num) == length:
            break

    return l[1:], membership


def membership(iden_lst, yrs):
    '''
    Function to return occurrence of different years
    in a list of datetime identifiers
    @author: Subhamoy Chatterjee
    Parameters:
        iden_list (list): list of flare onset date/time identifiers
        ens_num (list): list of year

    returns:
        l (list): redundant list of years from iden_list
        mem (list): number of occurrence of each year from yrs in l
    '''
    l = [iden[:4] for iden in iden_lst]
    mem = [l.count(yr) for yr in yrs]
    return l, mem


def test_rms(file, seed):
    '''
    Function to tune test set to ensure least modulation over solar cycle
    @author: Subhamoy Chatterjee
    Parameters:
        file (str): file location and name
        seed (int): seed value for random selection datapoints
                    in different flare bins

    returns:
        RMS deviation of datapoints per year over an expected membership value
    '''
    random.seed(seed)
    sep_data = pd.read_csv(file, header=1)
    data_p = list(sep_data['Peak Flux'][sep_data['event_type'] == 1])
    data_n = list(sep_data['Peak Flux'][sep_data['event_type'] == 0])
    iden_p = list(sep_data['Start Time'][sep_data['event_type'] == 1]
                  .astype(str))
    iden_n = list(sep_data['Start Time'][sep_data['event_type'] == 0]
                  .astype(str))
    error_msg = []
    # test data selection for events
    n_p = 5  # number of points per flare bin for events
    nb = 10  # number of flare bins
    nbs = histedges_equal(data_p, nb)  # flare bin edges
    nYears = 16
    startYear = 1998
    ba = startYear-0.5+np.linspace(0, nYears, nYears + 1)  # year bin edges
    y_list = [str(int(startYear+i))
              for i in range(nYears)]  # list of years sampled

    yrs_sp = []
    ml_p = n_p*nb/nYears  # threshold membership value
    hdr_p = {'flare start': [], 'peak': []}
    for i in range(nb):
        yrs = []
        pdt = (data_p >= nbs[i])*(data_p < nbs[i+1])
        idx = np.where(pdt == 1)[0].astype(int)
        if np.sum(pdt) < n_p:
            error_msg.append('cannot proceed')
            break
        ind = random.sample(range(np.sum(pdt)),n_p)
        for j in range(np.sum(pdt)):
            yrs.append(iden_p[idx[j]][:4])
        for k in range(n_p):
            yrs_sp.append(np.array(iden_p)[idx[ind[k]]][:4])
            hdr_p['flare start'].append(iden_p[idx[ind[k]]])
            hdr_p['peak'].append(data_p[idx[ind[k]]])
        _, mm = membership(yrs_sp, y_list)
        ind_l = []
        # remove years that cross the membership threshold
        for nn in range(len(mm)):
            if mm[nn] >= ml_p:
                for iid in range(len(iden_p)):
                    if iden_p[iid][:4] == y_list[nn]:
                        ind_l.append(iid)

        iden_pc = iden_p.copy()
        data_pc = data_p.copy()
        if len(ind_l) > 0:
            for indd in ind_l:
                iden_pc.remove(iden_p[indd])
                data_pc.remove(data_p[indd])
        data_p = data_pc
        iden_p = iden_pc

    # test data selection for non-events
    n_n = 2*n_p
    yrs_sn = []
    ml_n = n_n*nb/nYears  # threshold membership value
    hdr_n = {'flare start': [], 'peak': []}

    for i in range(nb):
        yrs = []
        pdt = (data_n >= nbs[i])*(data_n < nbs[i+1])
        idx = np.where(pdt == 1)[0].astype(int)
        if np.sum(pdt) < n_n:
            error_msg.append('cannot proceed')
            break
        ind = random.sample(range(np.sum(pdt)),n_n)
        for j in range(np.sum(pdt)):
            yrs.append(iden_n[idx[j]][:4])
        for k in range(n_n):
            yrs_sn.append(np.array(iden_n)[idx[ind[k]]][:4])
            hdr_n['flare start'].append(iden_n[idx[ind[k]]])
            hdr_n['peak'].append(data_n[idx[ind[k]]])
        _, mm = membership(yrs_sn, y_list)
        ind_l = []
        # remove years that cross membership limit
        for nn in range(len(mm)):
            if mm[nn] >= ml_n:
                for iid in range(len(iden_n)):
                    if iden_n[iid][:4] == y_list[nn]:
                        ind_l.append(iid)

        iden_nc = iden_n.copy()
        data_nc = data_n.copy()
        if len(ind_l) > 0:
            for indd in ind_l:
                iden_nc.remove(iden_n[indd])
                data_nc.remove(data_n[indd])
        data_n = data_nc
        iden_n = iden_nc

    if 'cannot proceed' in error_msg:
        return np.NaN
    else:
        yh_p, _ = np.histogram(np.asarray(yrs_sp).astype('float'), bins=ba)
        yh_n, _ = np.histogram(np.asarray(yrs_sn).astype('float'), bins=ba)
        sqSum = np.sum((yh_p - ml_p)**2)+np.sum((yh_n - ml_n)**2)
        rms = np.sqrt(sqSum/(2*nYears))
        return rms


def main():
    # create a dataset of seed vs test set RMS deviation 
    DATA_DIR = '/d1/sep_data/'
    N_ITER = 100
    FILENAME = 'all_ML_parameters_int_flux.csv'
    seed_vs_rms = {'seed':[], 'rms': []}
    for seed in range(N_ITER):
        seed_vs_rms['seed'].append(seed)
        rms = test_rms(DATA_DIR + FILENAME, seed)
        seed_vs_rms['rms'].append(rms)
        print([seed, rms])
    sr = pd.DataFrame(seed_vs_rms)
    sr.to_csv(DATA_DIR + 'seed_vs_rms.csv')


if __name__ == '__main__':
    main()
