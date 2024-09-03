#!/usr/bin/env python
# coding: utf-8

#other packages imports
import pandas as pd
import h5py
import pathlib
import numpy as np
import tensorflow as tf

#dlomix imports
import dlomix.dlomix.constants
from dlomix.dlomix.reports.postprocessing import normalize_intensity_predictions

class utility():

    def get_spectral_angle(data,predictions):
        predictions_df = pd.DataFrame()
        predictions_df['sequences']=data["test"]['modified_sequence']
        predictions_df['intensities_pred'] = predictions.tolist()
        predictions_df['intensities_raw']=data["test"]['intensities_raw']
        predictions_df['precursor_charge_onehot']=data["test"]['precursor_charge_onehot']
        predictions_acc = normalize_intensity_predictions(predictions_df, 128)

        spectral_angle = predictions_acc['spectral_angle'].describe()

        return spectral_angle
    
    def write_spectral_angle_to_file(save_path,modification,trainsize,spectral_angle_mean,AA,case):
        with open(f'{save_path}/{case}_ft_{modification}_{trainsize}.csv', "a") as myfile:
            df_spectral_angle = spectral_angle_mean.to_frame(name=f'{AA}').T
            get_row = df_spectral_angle.values
            count = get_row[0][0]
            mean = get_row[0][1]
            std = get_row[0][2]
            min = get_row[0][3]
            per_25 = get_row[0][4]
            per_50 = get_row[0][5]
            per_75 = get_row[0][6]
            max = get_row[0][7]
            final_string = f"{AA},{count},{mean},{std},{min},{per_25},{per_50},{per_75},{max},{modification}"
            myfile.write(final_string+"\n")

        
    def write_spectral_angle_to_file2(save_path,modification,trainsize,spectral_angle_mean,AA,case):
        with open(f'{save_path}', "a") as myfile:
            df_spectral_angle = spectral_angle_mean.to_frame(name=f'{AA}').T
            get_row = df_spectral_angle.values
            count = get_row[0][0]
            mean = get_row[0][1]
            std = get_row[0][2]
            min = get_row[0][3]
            per_25 = get_row[0][4]
            per_50 = get_row[0][5]
            per_75 = get_row[0][6]
            max = get_row[0][7]
            final_string = f"{AA},{count},{mean},{std},{min},{per_25},{per_50},{per_75},{max},{modification},{case}"
            myfile.write(final_string+"\n")
    
    def get_embedding_of_AA(file, AA):
        f = h5py.File(file, 'r')
        group = f['layers']
        layers = group['embedding']
        embedding = layers['vars']
        embedding_weights = embedding['0'][:]
        f.close()

        aa = dlomix.dlomix.constants.ALPHABET_UNMOD.keys()
        dict_AA_row = { amino : row for amino,row in zip(aa,embedding_weights[1:])}

        return dict_AA_row[AA]
    
    def change_embedding_of_weights_file(file):
        f = h5py.File(file, 'r')
        group = f['layers']
        layers = group['embedding']
        embedding = layers['vars']
        embedding_weights = embedding['0'][:]
        f.close()
        
        #Instantiate random uniform distribution 
        initializer = tf.keras.initializers.RandomUniform(
                        minval=-0.05, maxval=0.05, seed=None
                        )
        #create an array for the modified amino acid
        new_row = initializer(shape=(1, 32))
        embedding_weights_with_mod_row = np.vstack([embedding_weights,new_row])
     

        with h5py.File(file,'r+') as ds:
            del ds['layers/embedding/vars'] # delete old, differently sized dataset
            ds.create_dataset('layers/embedding/vars',data=embedding_weights_with_mod_row) # implant new-shaped dataset "X1"

    
    def write_only_mean_SA_to_file(save_path,modification,trainsize,amino_acid,spectral_angle_mean_test,spectral_angle_mean_val=None):
        with open(f'{save_path}/ft_{modification}_test_val_{trainsize}.csv', "a") as myfile:
            myfile.write(str(amino_acid)+","+str(spectral_angle_mean_test)+","+str(spectral_angle_mean_val)+"\n")

    def get_modification_in_UNIMOD(modification):
        mod_dict = {
            "cR": "R[UNIMOD:7]",
            "aK": "K[UNIMOD:1]",
            "prK": "K[UNIMOD:58]",
            "fK": "K[UNIMOD:122]",
            "mK": "K[UNIMOD:747]",
            "nY": 'Y[UNIMOD:354]',
            "hP": "P[UNIMOD:35]",
            "pS": "S[UNIMOD:21]",
            "pT": "T[UNIMOD:21]",
            "pY": "Y[UNIMOD:21]",
            "ahS": "S[UNIMOD:43]",
            "meK": "K[UNIMOD:34]",
            "meR": "R[UNIMOD:34]",
            "ahT": "T[UNIMOD:43]",
            "pyE": "E[UNIMOD:27]",
            "pyQ": "Q[UNIMOD:28]",
            "uK": "K[UNIMOD:121]",
            "biK": "K[UNIMOD:3]",
            "buK": "K[UNIMOD:1289]",
            "crK": "K[UNIMOD:1363]",
            "diK": "K[UNIMOD:36]",
            "glK": "K[UNIMOD:1848]",
            "gyK": "K[UNIMOD:121]",
            "hsK": "K[UNIMOD:1849]",
            "mlK": "K[UNIMOD:747]",
            "suK": "K[UNIMOD:64]",
            "trK": "K[UNIMOD:37]",
            "daR": "R[UNIMOD:36]",
            "dsR": "R[UNIMOD:36]",
            "me2R": "R[UNIMOD:34]",
            "a2K": "K[UNIMOD:1]"
        }

        return mod_dict[modification]
    
    def get_shortcut_of_mod(modification):
        mod_dict = {
            'R[UNIMOD:7]': 'cR',
            'K[UNIMOD:1]': 'a2K', 
            'K[UNIMOD:58]': 'prK', 
            'K[UNIMOD:122]': 'fK', 
            'K[UNIMOD:747]': 'mlK', 
            'Y[UNIMOD:354]': 'nY', 
            'P[UNIMOD:35]': 'hP', 
            'S[UNIMOD:21]': 'pS', 
            'T[UNIMOD:21]': 'pT', 
            'Y[UNIMOD:21]': 'pY', 
            'S[UNIMOD:43]': 'ahS', 
            'K[UNIMOD:34]': 'me2R', 
            'R[UNIMOD:34]': 'me2R', 
            'T[UNIMOD:43]': 'ahT', 
            'E[UNIMOD:27]': 'pyE', 
            'Q[UNIMOD:28]': 'pyQ', 
            'K[UNIMOD:121]': 'gyK', 
            'K[UNIMOD:3]': 'biK', 
            'K[UNIMOD:1289]': 'buK', 
            'K[UNIMOD:1363]': 'crK', 
            'K[UNIMOD:36]': 'diK', 
            'K[UNIMOD:1848]': 'glK', 
            'K[UNIMOD:1849]': 'hsK', 
            'K[UNIMOD:64]': 'suK', 
            'K[UNIMOD:37]': 'trK', 
            'R[UNIMOD:36]': 'dsR', 
            'R[UNIMOD:36]': 'dsR'
        }
        return mod_dict[modification]