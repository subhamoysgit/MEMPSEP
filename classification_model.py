from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation,\
    MaxPooling2D, ZeroPadding2D, Flatten, GlobalMaxPooling2D,\
    concatenate, Dropout, Dense


def MEMPSEP_C(sz=256):
    '''
    Archiecture of MEMPSEP
    @author: Subhamoy Chatterjee
    Parameters:
        sz (256): number of x/y pixels of magnetograms

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
    x = Dense(1, kernel_initializer='random_normal',
              bias_initializer='zeros')(x)
    out = Activation('sigmoid')(x)

    # building the model
    model = Model(inputs=[mag, win, upstr, xray_l1, elec_l1], outputs=[out])
    return model
