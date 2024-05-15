'''
Traning MEMPSEP
@author: Subhamoy Chatterjee
'''
import numpy as np
import pickle
from dataloader import trn_val_split, dataLoader
from classification_model import MEMPSEP
import tensorflow as tf
from classification_model_inference import infer
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = '/d1/sep_data/'
MODEL_DIR = '/d1/sep_data/models/'
MNAME = 'MUFWXE'

input_dict = {'M': int('M' in MNAME), 'U': int('U' in MNAME),
                  'X': int('X' in MNAME), 'E': int('E' in MNAME),
                  'W': int('W' in MNAME)}
filename_o = 'all_ML_parameters_int_flux.csv'
df_o = pd.read_csv(DATA_DIR + filename_o, header=1)

insitu_cols = ['SW Temp', 'SW Velocity', 'SW Density',
                'IMF B', 'IMF Bx', 'IMF By', 'IMF Bz',
                'Fe/O Lo', 'Fe/O Hi', 'H Lo', 'H Hi',
                'O Lo', 'O Hi', 'Fe Lo', 'Fe Hi']

etype0 = list(df_o['event_type'] == 0)
etype1 = list(df_o['event_type'] == 1)
# fill missing data
for col in insitu_cols:
    ind = np.logical_and(list(df_o[col] == -9999), etype0)
    ind_c = np.logical_and(list(df_o[col] != -9999), etype0)
    df_o.loc[ind, col] = df_o[col][ind_c].sample(np.sum(ind), replace=True,
                                                    random_state=239).values

for col in insitu_cols:
    ind = np.logical_and(list(df_o[col] == -9999), etype1)
    ind_c = np.logical_and(list(df_o[col] != -9999), etype1)
    df_o.loc[ind, col] = df_o[col][ind_c].sample(np.sum(ind),
                                                    replace=True,
                                                    random_state=239).values
# normalize
scale = MinMaxScaler()
for col in insitu_cols:
    if col in ['IMF Bx', 'IMF By', 'IMF Bz']:
        df_o[col] = df_o[col]/np.max(np.abs(df_o[col]))
    else:
        df_o[col] = pd.DataFrame(scale.fit_transform(df_o[[col]].values),
                                    columns=[col], index=df_o.index)

df_INSITU = df_o[insitu_cols]
pn = list(df_o['FlrOnset'])
pn = [d[:4]+d[5:7]+d[8:10]+d[11:13]+d[14:16] for d in pn]

dw = []
for _, _, files in os.walk(DATA_DIR + 'wind_waves/'):
    for name in files:
        dw.append(name[3:-len('.p')])

ds = []
for _, _, files in os.walk(DATA_DIR + 'x_ray/'):
    for name in files:
        ds.append(name[:12])

de = []
for _, _, files in os.walk(DATA_DIR + 'electrons_L1/'):
    for name in files:
        de.append(name[:12])

model = MEMPSEP(sz=256)


for en in range(0,10):
   ### ensemble number ###
    ensemble = 1+en
    trn_p = pd.read_csv(DATA_DIR+'sep_trn_p.csv')
    iden_tp = trn_p['flare start']
    strength_tp = trn_p['peak']
    tag_tp = ['1']*len(iden_tp)
    trn_n = pd.read_csv(DATA_DIR+'sep_trn_n_ensemble.csv')
    iden_tn = trn_n['flare start'][trn_n['ensemble number']==ensemble]
    strength_tn = trn_n['peak'][trn_n['ensemble number']==ensemble]
    tag_tn = ['0']*len(iden_tn)
    val_p = pd.read_csv(DATA_DIR+'sep_val_p.csv')
    iden_vp = val_p['flare start']
    strength_vp = val_p['peak']
    tag_vp = ['1']*len(iden_vp)
    val_n = pd.read_csv(DATA_DIR+'sep_val_n_ensemble.csv')
    iden_vn = val_n['flare start'][val_n['ensemble number']==ensemble]
    strength_vn = val_n['peak'][val_n['ensemble number']==ensemble]
    tag_vn = ['0']*len(iden_vn)
    iden_v = list(iden_vp) + list(iden_vn)
    iden_t = list(iden_tp) + list(iden_tn)
    strength_v = list(strength_vp) + list(strength_vn)
    strength_t = list(strength_tp) + list(strength_tn)
    tag_v = list(tag_vp) + list(tag_vn)
    tag_t = list(tag_tp) + list(tag_tn)
    trn_list = []
    val_list = []

    for root,dirs,files in os.walk(DATA_DIR+'magnetograms/'):
        for file in files:
            if file.startswith('mag'):
                fid = int(file[4:16])
                if fid in iden_t:
                    ind = iden_t.index(fid)
                    trn_list.append(str(iden_t[ind]) + '_' + tag_t[ind])
                if fid in iden_v:
                    ind = iden_v.index(fid)
                    val_list.append(str(iden_v[ind]) + '_' + tag_v[ind])
    
    ff_t = trn_list
    ff_v = val_list
    f_strength_t = []
    f_strength_v = []
    
    for i in range(len(ff_t)):
        ind = iden_t.index(int(ff_t[i][:12]))
        f_strength_t.append(strength_t[ind])
    
    for i in range(len(ff_v)):
        ind = iden_v.index(int(ff_v[i][:12]))
        f_strength_v.append(strength_v[ind])

    pos = 0
    neg = 0
    tp = 0
    tn = 0
   
    data = [[0]*14]
    for i in range(len(ff_t)):
        identifier = ff_t[i][:12]
        m = infer(model, identifier, input_dict, pn, df_INSITU, dw, ds, de, calibrate = False)
        data.append([identifier]+[f_strength_t[i]]+[ff_t[i][13]]+m+[np.median(m)])
        print(m)
        print('median prob = '+str(np.median(m)))
        print(str(ff_t[i][13]))
        print('flare peak = '+str(f_strength_t[i]))
        if ff_t[i][13]=='0':
           neg = neg + 1
           if np.median(m)<0.5:
              tn = tn + 1
        if ff_t[i][13]=='1':
           pos = pos + 1
           if np.median(m)>=0.5:
              tp = tp + 1
    print(pos)
    print(neg)
    print(tp)
    print(tn)
    print(tp/pos)
    print(tn/neg)
    
    data = data[1:]
    pickle.dump(data,open(MODEL_DIR+f"model_ensemble_on_trn_ens_{MNAME}_"+str(ensemble).zfill(2)+".p","wb"))
    