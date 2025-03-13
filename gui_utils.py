import streamlit as st
#from __future__ import absolute_import, division, print_function
import os, glob
import drms
import numpy as np
import requests
from bs4 import BeautifulSoup
import julian
from datetime import datetime
import csv
import json
import pickle
import matplotlib.pyplot as plt
from model_inference_R import infer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from probability_calibration import probability_calibration
from regression_model import MEMPSEP_R
from sunpy.map import Map
import astropy.units as u
import numpy as np
from astropy.io import fits
# from hmi_converter import hmi_converter
import shutil
from skimage.transform import resize
from scipy.interpolate import interp1d
import streamlit as st

def hmi_converter(file_m):
    MDI_fits = fits.open(file_m)
    MDI_fits.verify('fix')
    #MDI_fits[1].header['cunit1'] = 'arcsec' 
    #MDI_fits[1].header['cunit2'] = 'arcsec' 
    MDImap = Map(MDI_fits[1].data, MDI_fits[1].header)
    MDImap = MDImap.resample((1024,1024)*u.pix)
    #print(MDI_fits[1].header)
    ####rescaled EIT map
    scale_factor = 1.0167176487233658
    # Pad image, if necessary
    target_shape = int(1024)
    # Reform map to new size if original shape is too small
    new_fov = np.zeros((target_shape, target_shape)) * np.nan
    new_meta = MDImap.meta
    new_meta['crpix1'] = new_meta['crpix1'] - MDImap.data.shape[0] / 2 + new_fov.shape[0] / 2
    new_meta['crpix2'] = new_meta['crpix2'] - MDImap.data.shape[1] / 2 + new_fov.shape[1] / 2
    # Identify the indices for appending the map original FoV
    i1 = int(new_fov.shape[0] / 2 - MDImap.data.shape[0] / 2)
    i2 = int(new_fov.shape[0] / 2 + MDImap.data.shape[0] / 2)
    # Insert original image in new field of view
    new_fov[i1:i2, i1:i2] = MDImap.data[:,:] 
    # Assemble Sunpy map
    MDImap = Map(new_fov, new_meta)
    MDImap = MDImap.rotate(scale=scale_factor, recenter=True)
    sz_x_diff = (MDImap.data.shape[0]-target_shape)//2
    sz_y_diff = (MDImap.data.shape[0]-target_shape)//2

    MDImap.meta['crpix1'] = MDImap.meta['crpix1']-sz_x_diff
    MDImap.meta['crpix2'] = MDImap.meta['crpix2']-sz_y_diff

    MDImap = Map(MDImap.data[sz_x_diff:sz_x_diff+target_shape, sz_y_diff:sz_y_diff+target_shape].copy(), MDImap.meta)

    x, y = np.meshgrid(*[np.arange(v.value) for v in MDImap.dimensions]) * u.pixel
    hpc_coords = MDImap.pixel_to_world(x, y)
    rSun = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / MDImap.rsun_obs
    MDImap.data[rSun>1] = -1000
    conv =  np.clip(MDImap.data,-1000,1000)
    return conv


def find_nearest(identifier):
    files = glob.glob('/d1/sep_data/magnetograms/mag_*.p')
    idens_int = [np.abs(int(file[-14:-2])-int(identifier)) for file in files]
    idx = np.argmin(idens_int)
    return files[idx]
    
    

def infer(model, identifier, input_dict, pn, df_INSITU, dw, ds, de, calibrate = False):
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
        calibrate: perform probability calibration

    returns:
        model-ensemble inference for query flareonset
    '''
    DATA_DIR = '/d1/sep_data/'
    MODEL_DIR = '/d1/sep_data/models/'
    MNAME = 'MUFWXE'
    X1 = np.zeros((1, 256, 256, 13), dtype=float)
    X2 = np.zeros((1, 432, 80, 1), dtype=float)
    X5 = np.zeros((1, 15), dtype=float)
    X6 = np.zeros((1, 1441, 2, 1), dtype=float)
    X7 = np.zeros((1, 8640, 7, 1), dtype=float)
    mag = pickle.load(open('/d1/NRT/gui_data/'+identifier+'/mag_'+identifier+'.p', 'rb'))
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
        model.load_weights(MODEL_DIR+'gated_regression_' + MNAME +
                                    '_model_ensemble_' + str(nn+1).zfill(2)+'.h5')
        p = model.predict([X1, X2, X5, X6, X7])

        if calibrate == True:
            ens = pickle.load(open(MODEL_DIR + "model_ensemble_on_trn_ens_"
                                + MNAME + "_" + str(nn + 1).zfill(2)
                                + ".p", "rb"))
            gt_t = []
            p_t = []
            for i in range(len(ens)):
                p_t.append(ens[i][3 + nn])
                gt_t.append(float(int(ens[i][2]) == 1))

            p_cal = probability_calibration(p_t, gt_t, [p[0][0]])
            prob = p_cal.calibrateProbability(n_sel=20)
            infr.append(prob[0])
        else:
            infr.append(p[0])
    return infr



def inference(identifier):
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

    model = MEMPSEP_R(sz=256)
    m = infer(model, identifier, input_dict, pn, df_INSITU, dw, ds, de, calibrate=False)
    m = np.array(m)
    return m


def downloader(identifier, instrument='hmi'):
    email = 'subhamoy@boulder.swri.edu'
    export_protocol = 'fits'
    instrument = instrument
    dict = {'mdi':'mdi.fd_M_96m_lev182','hmi':'hmi.M_720s'}
    series = dict[instrument]
    dt = datetime(int(identifier[:4]),
                  int(identifier[4:6]),
                  int(identifier[6:8]),
                  int(identifier[8:10]),
                  int(identifier[10:12]),
                  0,0)
    
    jd = julian.to_jd(dt, fmt='jd')
    t_i = julian.from_jd(jd-3, fmt='jd')
    ar_i = str(t_i.year) + '.' + str(t_i.month).zfill(2) + '.' +\
        str(t_i.day).zfill(2) + '_' + str(t_i.hour).zfill(2) + ':' + \
            str(t_i.minute).zfill(2) + ':' + str(t_i.second).zfill(2)
    tsel = ar_i + '/3d@360m'
    qstr = '%s[%s]' % (series, tsel)
    #print(qstr)
    c = drms.Client(verbose=True)
    with st.spinner("Downloading data..."):
        try:
            r = c.export(qstr, method='url', protocol=export_protocol, email=email)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print("The server might be returning an empty or invalid response.")
    
    
        if '%s' % r.request_url != 'None':
            out_dir = '/d1/NRT/gui_data/'+identifier
            os.makedirs(out_dir)
            r.download(out_dir)
    st.success("Download completed")
def packager(identifier, instrument='hmi'):
    DIR = '/d1/NRT/gui_data/' + identifier +'/'
    instrument = instrument
    dict = {'mdi':'mdi.fd_M_96m_lev182','hmi':'hmi.M_720s.'}
    series = dict[instrument]
    f = glob.glob(DIR + '*.fits')
    dt = datetime(int(identifier[:4]),
                  int(identifier[4:6]),
                  int(identifier[6:8]),
                  int(identifier[8:10]),
                  int(identifier[10:12]),
                  0,0)
    jd0 = julian.to_jd(dt, fmt='jd')
    jd = []
    f = sorted(f)
    for name in f:
        fl = name[len(DIR + series):len(DIR + series)+13]
        dt = datetime(int(fl[:4]),
        int(fl[4:6]),
        int(fl[6:8]),
        int(fl[9:11]),
        int(fl[11:13]),
        0,0)
        jdf = julian.to_jd(dt, fmt='jd')
        jd.append((jdf-jd0)*24) 
    mn = np.min(jd)
    mx = np.max(jd)
    arr = np.linspace(0,72,13)-72
    mdi_stack = np.zeros((256,256,len(f)))
    my_bar = st.progress(0)
    st.write('Packaging data')
    for i in range(len(f)):
        mdi_stack[:,:,i] = resize(hmi_converter(f[i])/1000,(256,256),anti_aliasing = True)
        my_bar.progress((i+1)/13)
    st.success('Packaging completed')
    mdi_stack_interp = np.zeros((256,256,13),dtype='uint16')
    for i in range(13):
        if arr[i] <= mn:
            mdi = mdi_stack[:,:,0]
            mdi = 65535*0.5*(1+mdi)
            mdi_stack_interp[:,:,i] = mdi
        if arr[i] >= mx:
            mdi = mdi_stack[:,:,len(f)-1]
            mdi = 65535*0.5*(1+mdi)
            mdi_stack_interp[:,:,i] = mdi
    #print(arr)
    mask = (arr >= mn)*(arr <= mx)
    ind = np.where(mask==1)[0]
    for m in range(256):
        for n in range(256):
            f = interp1d(jd, mdi_stack[m,n,:],kind='nearest')
            mdi_stack_interp[m,n,ind] = 65535*0.5*(1 + f(arr[ind])) 
    pickle.dump(mdi_stack_interp, open(DIR+"mag_"+identifier+ ".p", "wb" ))


    #folder_path = '/d1/NRT/gui_data/'+identifier

    # # Delete the folder if it exists
    # if os.path.exists(folder_path):
    #     shutil.rmtree(folder_path)

# def infer():
#     pass