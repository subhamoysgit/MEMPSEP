import numpy as np
import random
random.seed(23)


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
