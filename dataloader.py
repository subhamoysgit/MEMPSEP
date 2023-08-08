import random
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

seed = 23
DATA_DIR = '/d1/sep_data/'


def trn_val_split(ensemble):
    '''
    Returns time identifiers for training  and validation sets
    to train each ensemble member
    @author: Subhamoy Chatterjee
    Parameters:
        ensemble: index of ensemble member
    '''
    trn_p = pd.read_csv(DATA_DIR+'sep_trn_p.csv')
    iden_tp = trn_p['flare start']
    tag_tp = ['1']*len(iden_tp)
    trn_n = pd.read_csv(DATA_DIR+'sep_trn_n_ensemble.csv')
    iden_tn = trn_n['flare start'][trn_n['ensemble number'] == ensemble]
    tag_tn = ['0']*len(iden_tn)
    val_p = pd.read_csv(DATA_DIR+'sep_val_p.csv')
    iden_vp = val_p['flare start']
    tag_vp = ['1']*len(iden_vp)
    val_n = pd.read_csv(DATA_DIR+'sep_val_n_ensemble.csv')
    iden_vn = val_n['flare start'][val_n['ensemble number'] == ensemble]
    tag_vn = ['0']*len(iden_vn)
    iden_v = list(iden_vp) + list(iden_vn)
    iden_t = list(iden_tp) + list(iden_tn)
    tag_v = list(tag_vp) + list(tag_vn)
    tag_t = list(tag_tp) + list(tag_tn)
    trn_list = []
    val_list = []

    for _, _, files in os.walk(DATA_DIR+'magnetograms/'):
        for file in files:
            if file.startswith('mag'):
                fid = int(file[-14:-2])
                if fid in iden_t:
                    ind = iden_t.index(fid)
                    trn_list.append(str(iden_t[ind]) + '_' + tag_t[ind])
                if fid in iden_v:
                    ind = iden_v.index(fid)
                    val_list.append(str(iden_v[ind]) + '_' + tag_v[ind])

    random.Random(seed).shuffle(trn_list)
    random.Random(seed).shuffle(val_list)
    return trn_list, val_list


def dataLoader(file_list, batch_size, input_dict):
    '''
    dataLoader for MEMPSEP
    @author: Subhamoy Chatterjee
    Parameters:
        file_list (list): list of flare onset dates from training/validation
        batch_size (int): size of training batches
        input_dict (dict): dictionary of model inputs vs. binary mask

    returns:
        tuple of model inputs and output
    '''
    L = len(file_list)
    k = 0
    num = np.int(L/batch_size)
    # peparing insitu parameter list
    df = pd.read_csv(DATA_DIR+'all_ML_parameters_int_flux.csv', header=1)
    insitu_cols = ['SW Temp', 'SW Velocity', 'SW Density', 'IMF B', 'IMF Bx',
                   'IMF By', 'IMF Bz', 'Fe/O Lo', 'Fe/O Hi', 'H Lo', 'H Hi',
                   'O Lo', 'O Hi', 'Fe Lo', 'Fe Hi']
    etype0 = list(df['event_type'] == 0)
    etype1 = list(df['event_type'] == 1)

    for col in insitu_cols:
        ind = np.logical_and(list(df[col] == -9999), etype0)
        ind_c = np.logical_and(list(df[col] != -9999), etype0)
        df.loc[ind, col] = df[col][ind_c].sample(np.sum(ind), replace=True,
                                                 random_state=239).values

    for col in insitu_cols:
        ind = np.logical_and(list(df[col] == -9999), etype1)
        ind_c = np.logical_and(list(df[col] != -9999), etype1)
        df.loc[ind, col] = df[col][ind_c].sample(np.sum(ind), replace=True,
                                                 random_state=239).values

    scale = MinMaxScaler()
    for col in insitu_cols:
        if col in ['IMF Bx', 'IMF By', 'IMF Bz']:
            df[col] = df[col]/np.max(np.abs(df[col]))
        else:
            df[col] = pd.DataFrame(scale.fit_transform(df[[col]].values),
                                   columns=[col], index=df.index)

    df_INSITU = df[insitu_cols]

    pn = list(df['Start Time'])
    dw = []
    for _, _, files in os.walk(DATA_DIR+'wind_waves/'):
        for name in files:
            dw.append(name[3:-len('.p')])

    ds = []
    xray = [[0]*1441]
    for _, _, files in os.walk(DATA_DIR+'x_ray/'):
        for name in files:
            ds.append(name[:12])

    de = []
    for _, _, files in os.walk(DATA_DIR+'electrons_L1/'):
        for name in files:
            de.append(name[:12])

    while True:
        k = k % num
        f_batch = file_list[batch_size*k:batch_size*(k+1)]
        X1 = np.zeros((batch_size, 256, 256, 13), dtype=float)
        X2 = np.zeros((batch_size, 432, 80, 1), dtype=float)
        X5 = np.zeros((batch_size, 15), dtype=float)
        X6 = np.zeros((batch_size, 1441, 2, 1), dtype=float)
        X7 = np.zeros((batch_size, 8640, 7, 1), dtype=float)
        Y = np.zeros((batch_size, 1), dtype=float)
        for r in range(batch_size):
            mag = pickle.load(open(DATA_DIR+'magnetograms/mag_'+f_batch[r][:12]+'.p','rb'))
            mag = (1/65535)*(mag.astype(float))
            X1[r, :, :, :] = mag
            X1[r, :, :, :] = input_dict['M']*X1[r, :, :, :]

            if f_batch[r][:12] in dw:
                X2[r, :, :, 0] = pickle.load(open(DATA_DIR + 'wind_waves/ww_' +
                                                  f_batch[r][:12]+'.p', 'rb'))
            X2[r, :, :, 0] = input_dict['W']*X2[r, :, :, 0]

            idx = pn.index(int(f_batch[r][:12]))
            X5[r, :] = np.asarray(df_INSITU.iloc[idx, :].astype('float'))
            X5[r, :] = input_dict['U']*X5[r, :]

            if f_batch[r][:12] in ds:
                xray = pd.read_csv(DATA_DIR + 'x_ray/' +
                                   f_batch[r][:12]+'.csv', header=1)
                xray = xray.fillna(0)
                xray = xray[['0.45nm', '0.175nm']].to_numpy()
                for i in range(2):
                    if np.max(xray[:, i]) > 0:
                        xray[:, i] = xray[:, i]/np.max(xray[:, i])
                xray = xray[:1441, :]

            X6[r, :, :, 0] = input_dict['X']*xray

            if f_batch[r][:12] in de:
                elec = pd.read_csv(DATA_DIR + 'electrons_L1/' +
                                   f_batch[r][:12] + '.csv')
                elec = elec.fillna(0)
                elec = elec[['fepm_E_0', 'fepm_E_1', 'fepm_E_2', 'fepm_E_3',
                             'fepm_E_4', 'fepm_E_5', 'fepm_E_6']].to_numpy()
                if elec.shape[0] > 8640:
                    elec = elec[-8640:, :]
                for i in range(7):
                    if np.max(elec[:, i]) > 0:
                        elec[:, i] = elec[:, i]/np.max(elec[:, i])

            X7[r, (8640-elec.shape[0]):, :, 0] = input_dict['E']*elec

            if f_batch[r][13] == '1':
                Y[r, 0] = 1.0
            else:
                Y[r, 0] = 0.0
        k = k+1
        if k == num:
            random.shuffle(file_list)
        yield ([X1, X2, X5, X6, X7], Y)
