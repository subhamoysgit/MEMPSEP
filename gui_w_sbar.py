##meeting note Feb 17, 2025
#indicate data availability window (data latency, data lag time, currently available, data gap etc.)
#json 
# MEMPSEP training period

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
from gui_utils import *
import pickle

def reset():
    st.session_state['file'] = None
    st.session_state['id'] = None
    st.session_state['fr'] = 0
    st.session_state['vis'] = 0
    
def download_input():
    DIR = '/d1/NRT/gui_data/'
    date = st.session_state['dt']
    yyyy = date.split('/')[0]
    mm = date.split('/')[1]
    dd = date.split('/')[2].split('-')[0]
    hh = date.split('/')[2].split('-')[1].split(':')[0]
    mm_ = date.split('/')[2].split('-')[1].split(':')[1]
    identifier = yyyy+mm+dd+hh+mm_ 
    downloader(identifier)
    packager(identifier)
    st.session_state['id']= identifier
    st.session_state['file'] = DIR+identifier+"/mag_"+identifier+ ".p"

def visualize_input():
    st.session_state['vis'] = 1
    

def forecast():
    st.session_state['fr'] = 1


def main():

    if 'file' not in st.session_state:
        st.session_state['file'] = None
    if 'id' not in st.session_state:
        st.session_state['id'] = None
    if 'vis' not in st.session_state:
        st.session_state['vis'] = 0
    if 'fr' not in st.session_state: 
        st.session_state['fr'] = 0
        
    
    now = datetime.now()
    col1, col2 = st.columns(2)

    # Format the date
    default = now.strftime("%Y/%m/%d")+ '-' +now.strftime("%H:%M:%S")[:-3]
    if 'dt' not in st.session_state: 
        st.session_state['dt'] = default
    st.sidebar.header("MEMPSEP")

    st.sidebar.markdown('This GUI allows download NRT in-situ + remote sensing data and infer SEP occurrence+properties using MEMPSEP trained on data from 1998-2013')
    
    selected_date = st.sidebar.text_input('Change Forecast Date (YYYY/MM/DD-HH:MM)', 
                          value = st.session_state['dt'],
                          help='Date-time at which a forecast is made', on_change=reset)
    st.session_state['dt'] = selected_date
    st.sidebar.button("Download Input", help='Downloads In-situ + Remote Sensing Input over 3 days prior to forecast date',
                      on_click=download_input)
    
    if st.session_state['file'] is not None:
        st.sidebar.multiselect("Select Visualization Input",
                                        ['Magnetograms', 'X-ray', 'Electron', 'Plasma Properties'], help='inputs to be visualized')
                                        
        st.sidebar.button("Visualize Input", help='Visualizes',
                        on_click=visualize_input)
        
        if st.session_state['vis']==1:
            with col1:
                st.write('Magnetograms')
                p = pickle.load(open(st.session_state['file'], 'rb'))
                fig, axes = plt.subplots(2, 3, figsize=(5*3, 5*2))  # 1 column, n rows
                axes = axes.ravel()
                # Plot each image
                for i in range(6):
                    axes[i].imshow(p[:, :, 2*i], cmap="gray", vmin=0, vmax=65535)
                    axes[i].axis("off")  # Remove axes for a cleaner look
                # Display the figure in Streamlit
                st.pyplot(fig)

        st.sidebar.button("Forecast SEP", help='forecast SEP ocurrence probability and properties in 6 hrs from input date',
                        on_click=forecast)
        
        
        if st.session_state['fr'] == 1:
            m =inference(st.session_state['id'])
            with col2:
                fig1, ax = plt.subplots(2,1,figsize=(5, 5*2))
                ax[0].boxplot(m[:,0], vert=False)
                ax[0].set_title('SEP probability')
                m =inference(st.session_state['file'][-14:-2])
                t50 = np.quantile(m[:,6], 0.5)
                t25 = np.quantile(m[:,6], 0.25)
                t75 = np.quantile(m[:,6], 0.75)
                colors = ['r', 'g', 'b', 'k', 'yellow']
                for i in range(5):
                    p25 = 10**np.quantile(m[:,1+i], 0.25)
                    p50 = 10**np.quantile(m[:,1+i], 0.5)
                    p75 = 10**np.quantile(m[:,1+i], 0.75)
                    ax[1].errorbar([t50], [p50], xerr=[t75-t25], yerr=[p75-p25], fmt='o', ecolor=colors[i], capsize=5)
                ax[1].set_yscale('log')
                ax[1].set_title('SEP properties')
                st.pyplot(fig1)
                

        file_path = st.session_state['file']


        if file_path is not None:
            with open(file_path, "rb") as f:
                pickle_data = f.read()
            st.sidebar.download_button("Download data", 
                                data=pickle_data,
                                file_name="data.pkl",
                                mime="application/octet-stream",
                                help='Download Model Input, Target and Forecasted value')
    
           
    
    

if __name__=='__main__':
    main()
    

