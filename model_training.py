'''
Traning MEMPSEP
@author: Subhamoy Chatterjee
'''
import numpy as np
import pickle
from dataloader import trn_val_split, dataLoader
from model import MEMPSEP
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping

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
    model = MEMPSEP(sd=256)
    sgd = tf.keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    checkpoint = ModelCheckpoint(MODEL_DIR+'2_class_' + MODEL_NAME +
                                 '_model_ensemble_' + ensemble.zfill(2)+'.h5',
                                 monitor='val_accuracy', verbose=1,
                                 save_best_only=True, save_weights_only=True,
                                 mode='auto', period=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0,
                                   patience=5, verbose=1, mode='auto')
    history = model.fit(dataLoader(trn_list, BATCH_SIZE, INPUT_DICT),
                        steps_per_epoch=len(trn_list)/BATCH_SIZE, epochs=40,
                        callbacks=[checkpoint],
                        validation_data=dataLoader(val_list, BATCH_SIZE,
                                                   INPUT_DICT),
                        validation_steps=len(val_list)/BATCH_SIZE)
    pickle.dump(history.history['loss'], open(MODEL_DIR + '2_class_' +
                                              MODEL_NAME +
                                              '_trn_loss_ensemble_'
                                              + ensemble.zfill(2)+'.p',
                                              'wb'))
    pickle.dump(history.history['accuracy'], open(MODEL_DIR + '2_class_' +
                                                  MODEL_NAME +
                                                  '_trn_accu_ensemble_' +
                                                  ensemble.zfill(2) + '.p',
                                                  'wb'))
    pickle.dump(history.history['val_loss'], open(MODEL_DIR+'2_class_' +
                                                  MODEL_NAME +
                                                  '_val_loss_ensemble_' +
                                                  ensemble.zfill(2)+'.p',
                                                  'wb'))
    pickle.dump(history.history['val_accuracy'], open(MODEL_DIR + '2_class_' +
                                                      MODEL_NAME +
                                                      '_val_accu_ensemble_' +
                                                      ensemble.zfill(2)+'.p',
                                                      'wb'))
