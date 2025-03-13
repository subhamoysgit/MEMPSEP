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

def reset():
    st.session_state['fe'] = 0
    st.session_state['fr'] = 0

def main():
    DIR = '/d1/NRT/gui_data/'
    if 'fe' not in st.session_state: 
        st.session_state['fe'] = 0
    if 'fr' not in st.session_state: 
        st.session_state['fr'] = 0
    # Create a 2x2 grid
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    now = datetime.now()

    
    # Format the date
    default = now.strftime("%Y/%m/%d")+ '-' +now.strftime("%H:%M:%S")[:-3]
    if 'dt' not in st.session_state: 
        st.session_state['dt'] = default

    # Fill the grid with content
    with col1:
        st.write("Input Date")
        selected_date = st.text_input("YYYY/MM/DD-HH:MM",
                                      value=st.session_state['dt'])
        st.session_state['dt'] = selected_date
        reset()
        yyyy = selected_date.split('/')[0]
        mm = selected_date.split('/')[1]
        dd = selected_date.split('/')[2].split('-')[0]
        hh = selected_date.split('/')[2].split('-')[1].split(':')[0]
        mm_ = selected_date.split('/')[2].split('-')[1].split(':')[1]
        identifier = yyyy+mm+dd+hh+mm_ 
        file = DIR+identifier+"/mag_"+identifier+ ".p"
        # file = find_nearest(identifier)
        # st.write(file)
            

    with col2:
        st.write("SEP Occurrence Probability")
        if st.button("Forecast"):
            st.session_state['fr'] = 1
            
            #st.write(m)
        if st.session_state['fr'] == 1:
            #m =inference(file[-14:-2])
            m =inference(identifier)
            fig1, ax1 = plt.subplots()
            ax1.boxplot(m[:,0], vert=False)
            st.pyplot(fig1)
            

    with col3:
        st.write("Magnetograms")
        if st.button("Fetch"):
            st.session_state['fe'] = 1
            downloader(identifier)
            st.write('downloaded')
            packager(identifier)
            st.write('packaged')
        if st.session_state['fe']==1:
            p = pickle.load(open(file, 'rb'))
            st.write(p.shape)
            fig, axes = plt.subplots(1, p.shape[2]//2, figsize=(5, p.shape[2]//2 * 5))  # 1 column, n rows

            # Plot each image
            for i in range(p.shape[2]//2):
                axes[i].imshow(p[:, :, 2*i], cmap="gray", vmin=0, vmax=65535)
                axes[i].axis("off")  # Remove axes for a cleaner look

            # Display the figure in Streamlit
            st.pyplot(fig)
            

    with col4:
        st.write("SEP Properties")
        if st.session_state['fr'] == 1:
            m =inference(file[-14:-2])
            fig1, ax1 = plt.subplots()
            t50 = np.quantile(m[:,6], 0.5)
            t25 = np.quantile(m[:,6], 0.25)
            t75 = np.quantile(m[:,6], 0.75)
            colors = ['r', 'g', 'b', 'k', 'yellow']
            for i in range(5):
                p25 = 10**np.quantile(m[:,1+i], 0.25)
                p50 = 10**np.quantile(m[:,1+i], 0.5)
                p75 = 10**np.quantile(m[:,1+i], 0.75)
                ax1.errorbar([t50], [p50], xerr=[t75-t25], yerr=[p75-p25], fmt='o', ecolor=colors[i], capsize=5)
            ax1.set_yscale('log')
            st.pyplot(fig1)
    

if __name__=='__main__':
    main()
    

