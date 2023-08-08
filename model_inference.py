import numpy as np
import pandas as pd
import pickle
from probability_calibration import probability_calibration
DATA_DIR = '/d1/sep_data/'
MODEL_DIR = '/d1/sep_data/models/'
MNAME = 'MUFWXE'
input_dict = {'M': int('M' in MNAME), 'U': int('U' in MNAME),
              'X': int('X' in MNAME), 'E': int('E' in MNAME),
              'W': int('W' in MNAME)}


def infer(model, identifier, input_dict, pn, df_INSITU, dw, ds, de):
    '''
    MEMPSEP inference
    @author: Subhamoy Chatterjee
    Parameters:
        model (tf.keras.model): MEMPSEP architecture
        identifier (str): flareonset (YYYYMMDDHHSS) for model inference
        input_dict (dict): dictionary of model inputs vs. binary mask
        pn (list): list of identifiers in the dataset
        df_INSITU (dataFrame): dataframe for insitu properties
        dw: list of identifiers for wind/waves images
        ds: list of identifiers for X-ray time-series
        de: list of identifiers for L1 electron time-series

    returns:
        model-ensemble inference for query flareonset
    '''
    X1 = np.zeros((1, 256, 256, 13), dtype=float)
    X2 = np.zeros((1, 432, 80, 1), dtype=float)
    X5 = np.zeros((1, 15), dtype=float)
    X6 = np.zeros((1, 1441, 2, 1), dtype=float)
    X7 = np.zeros((1, 8640, 7, 1), type=float)
    mag = pickle.load(open(DATA_DIR + 'mag_'+identifier+'.p', 'rb'))
    mag = (1/65535)*(mag.astype(float))
    X1[0, :, :, :] = mag
    X1[0, :, :, :] = input_dict['M']*X1[0, :, :, :]

    if identifier in dw:
        X2[0, :, :, 0] = pickle.load(open(DATA_DIR + 'wind_waves/ww_' +
                                          identifier + '.p', 'rb'))
    X2[0, :, :, 0] = input_dict['W']*X2[0, :, :, 0]

    if identifier in pn:
        idx = pn.index(identifier)
        X5[0, :] = np.asarray(df_INSITU.iloc[idx, :].astype('float'))

    X5[0, :] = input_dict['U']*X5[0, :]

    if identifier in ds:
        xray = pd.read_csv(DATA_DIR + 'x_ray/' + identifier+'.csv', header=1)
        xray = xray.fillna(0)
        xray = xray[['0.45nm', '0.175nm']].to_numpy()
        for i in range(2):
            if np.max(xray[:, i]) > 0:
                xray[:, i] = xray[:, i]/np.max(xray[:, i])
        xray = xray[:1441, :]
        X6[0, :, :, 0] = input_dict['X']*xray

    if identifier in de:
        elec = pd.read_csv(DATA_DIR + 'electrons_L1/' + identifier+'.csv')
        elec = elec.fillna(0)
        elec = elec[['fepm_E_0', 'fepm_E_1', 'fepm_E_2',
                     'fepm_E_3', 'fepm_E_4',
                     'fepm_E_5', 'fepm_E_6']].to_numpy()
        if elec.shape[0] > 8640:
            elec = elec[-8640:, :]
        for i in range(7):
            if np.max(elec[:, i]) > 0:
                elec[:, i] = elec[:, i]/np.max(elec[:, i])

        X7[0, (8640-elec.shape[0]):, :, 0] = input_dict['E']*elec

    infr = []
    for nn in range(10):
        ens = pickle.load(open(MODEL_DIR + "model_ensemble_on_trn_ens_"
                               + MNAME + "_" + str(nn + 1).zfill(2)
                               + ".p", "rb"))
        gt_t = []
        p_t = []
        for i in range(len(ens)):
            p_t.append(ens[i][3 + nn])
            gt_t.append(float(int(ens[i][2]) == 1))
        model.load_weights(MODEL_DIR + '2_class_' + MNAME + '_model_ensemble_'
                           + str(nn+1).zfill(2) + '.h5')
        p = model.predict([X1, X2, X5, X6, X7])
        p_cal = probability_calibration(p_t, gt_t, [p[0][0]])
        prob = p_cal.calibrateProbability(n_sel=20)
        infr.append(prob[0])
    return infr
