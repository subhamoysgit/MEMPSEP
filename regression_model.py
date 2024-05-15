from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation,\
    MaxPooling2D, ZeroPadding2D, Flatten, GlobalMaxPooling2D,\
    concatenate, Dropout, Dense
    
import pickle
from probability_calibration import probability_calibration
from classification_model import MEMPSEP_C
MODEL_DIR = '/d1/sep_data/models/'
NAME = 'MUFWXE'
def MEMPSEP_R(sz=256,
              ensemble=0):
    '''
    Archiecture of MEMPSEP
    @author: Subhamoy Chatterjee
    Parameters:
        sz (256): number of x/y pixels of magnetograms
        ensemble (str): classfication model ensemble member
                        to perform probability gating

    returns:
        Model (tf.keras.Model)
    '''
    # Magnetogram Sequence
    mag = Input(shape=(sz, sz, 13))
    m = Conv2D(32, (3, 3), padding='same',
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros")(mag)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(m)

    m = Conv2D(32, (3, 3), padding='valid',
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros")(m)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(m)

    m = Conv2D(64, (3, 3), padding='same',
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros")(m)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(m)

    m = ZeroPadding2D((2, 2))(m)
    m = Conv2D(128, (3, 3), padding='valid',
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros")(m)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(m)

    m = Conv2D(256, (3, 3), padding='valid',
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros")(m)
    m = Activation('relu')(m)
    m = Flatten()(m)

    # Wind/waves time-frequency Image
    win = Input(shape=(432, 80, 1))
    w = Conv2D(4, (3, 3), padding='valid',
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros")(win)
    w = BatchNormalization()(w)
    w = Activation('relu')(w)
    w = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(w)
    w = Conv2D(8, (3, 3), padding='valid',
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros")(w)
    w = BatchNormalization()(w)
    w = Activation('relu')(w)
    w = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(w)
    w = Conv2D(16, (3, 3), padding='valid',
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros")(w)
    w = BatchNormalization()(w)
    w = Activation('relu')(w)
    w = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(w)
    w = Flatten()(w)

    # Upstrean parameters
    upstr = Input(shape=(15))

    # multi-channel x-ray time-series
    xray_l1 = Input(shape=(1441, 2, 1))
    xrl1 = Conv2D(20, (2, 2), padding='valid', activation="tanh")(xray_l1)
    xrl1 = GlobalMaxPooling2D()(xrl1)

    # multi-channel electron time-series
    elec_l1 = Input(shape=(8640, 7, 1))
    ell1 = Conv2D(20, (7, 7), padding='valid', activation="tanh")(elec_l1)
    ell1 = GlobalMaxPooling2D()(ell1)

    # contacatenation of embeddings derived from multiple inputs
    x = concatenate([m, w, upstr, xrl1, ell1])
    x = Dense(100, kernel_initializer='random_normal',
              bias_initializer='zeros')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5, seed=23)(x)
    x = Dense(7,kernel_initializer='random_normal',bias_initializer='zeros')(x)
    out_r = Activation('linear')(x)
    
    #load pretrained classification model
    model_c = MEMPSEP_C(sz=sz)
    model_c.load_weights(MODEL_DIR+'2_class_'+NAME+'_model_ensemble_'
                         +str(ensemble + 1).zfill(2)+'.h5')
    model_layers = model_c.layers
    for i in range(len(model_layers)):
        model_layers[i].trainable = False
    out_c = model_c([mag, win, upstr, xray_l1, elec_l1])
    
    #convert classification model outcome to probability
    ens = pickle.load(open(MODEL_DIR + "model_ensemble_on_trn_ens_"
                           + NAME + "_" + str(ensemble + 1).zfill(2)
                           + ".p", "rb"))
    gt_t = []
    p_t = []
    for i in range(len(ens)):
        p_t.append(ens[i][3 + ensemble])
        gt_t.append(float(int(ens[i][2]) == 1))

    p_cal = probability_calibration(p_t, gt_t, [out_c[0][0]])
    out_c = p_cal.calibrateProbability(n_sel=20)
    
    # final output as concatenation of sep probability and properties 
    out = concatenate([out_c, out_r])

    # building the model
    model = Model(inputs=[mag, win, upstr, xray_l1, elec_l1], outputs=[out])
    return model
