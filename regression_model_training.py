'''
Traning MEMPSEP
@author: Subhamoy Chatterjee
'''
import numpy as np
import pickle
from dataloader import trn_val_split, dataLoader, dataLoader_r
from classification_model import MEMPSEP_R
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K

def my_loss(y_true, y_pred):
    t = tf.tile(tf.reshape(y_pred[:,0],(-1,1)),(1,7))
    sq_diff = K.mean(K.square(y_true[:,1:]  - y_pred[:,1:])*t)
    return sq_diff

def main():
    seed = 203
    np.random.seed(seed)
    DATA_DIR = '/d1/sep_data/'
    MODEL_DIR = '/d1/sep_data/models/'
    MODEL_NAME = 'MUFWXE'
    # input to dataloader for masking inputs
    INPUT_DICT = {'M': int('M' in MODEL_NAME), 'U': int('U' in MODEL_NAME),
                'X': int('X' in MODEL_NAME), 'E': int('E' in MODEL_NAME),
                'W': int('W' in MODEL_NAME)}
    BATCH_SIZE = 10
    # training the ensemble of models
    for n in range(10):
        ensemble = str(n+1)
        trn_list, val_list = trn_val_split(n+1)
        model = MEMPSEP_R(sd=256,
                        ensemble=n)
        sgd = tf.keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        model.compile(loss =my_loss,  optimizer = sgd, metrics =[my_loss],run_eagerly = True)
        checkpoint = ModelCheckpoint(MODEL_DIR+'gated_regression_' + MODEL_NAME +
                                    '_model_ensemble_' + ensemble.zfill(2)+'.h5',
                                    monitor='val_loss', verbose=1,
                                    save_best_only=True, save_weights_only=True,
                                    mode='auto', period=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                    patience=5, verbose=1, mode='auto')
        history = model.fit(dataLoader_r(trn_list, BATCH_SIZE, INPUT_DICT),
                            steps_per_epoch=len(trn_list)/BATCH_SIZE, epochs=100,
                            callbacks=[checkpoint, early_stopping],
                            validation_data=dataLoader_r(val_list, BATCH_SIZE,
                                                    INPUT_DICT),
                            validation_steps=len(val_list)/BATCH_SIZE)
        pickle.dump(history.history['loss'], open(MODEL_DIR + 'gated_regression_' +
                                                MODEL_NAME +
                                                '_trn_loss_ensemble_'
                                                + ensemble.zfill(2)+'.p',
                                                'wb'))
        pickle.dump(history.history['val_loss'], open(MODEL_DIR+'gated_regression_' +
                                                    MODEL_NAME +
                                                    '_val_loss_ensemble_' +
                                                    ensemble.zfill(2)+'.p',
                                                    'wb'))

if __name__=='__main__':
    main()