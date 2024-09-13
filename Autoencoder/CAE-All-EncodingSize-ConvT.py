#!/usr/bin/env python
# coding: utf-8
# In[1]:

import keras
from sklearn.model_selection import train_test_split
import numpy as np
from keras import layers
import os
import math
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import KFold


# If using TensorFlow, this will make GPU ops as deterministic as possible,
# but it will affect the overall performance, so be mindful of that.
tf.config.experimental.enable_op_determinism()

# 90 PIDs - both spontaneous and induced datasets.
pids = [1,2,7,10,11,15,29,35,36,48,55,61,67,69,72,73,75,76,77,80,88,90,91,93,97,
        99,102,104,106,107,108,110,111,113,120,124,126,129,136,138,141,144,152,
        156,161,169,170,173,179,183,189,191,193,195,197,6, 14, 19, 20, 26, 27, 31, 
        44, 50, 52, 56, 60, 65, 66, 82, 86, 98, 101, 103, 114, 117, 121, 122, 
        125, 127, 131, 135, 165, 167, 176, 181, 190, 192, 19,169]
pids.sort()

exp_dir = '/home/chinmai/src/Oura/Data/All/Linear_Interpolation_y/'

kf = KFold(n_splits=11)
kf.get_n_splits(pids)

#print(kf)
#train_pids = []
#test_pids = []

for i, (train_index, test_index) in enumerate(kf.split(pids)):
    print(f"Fold {i}:")
    tr_pids = []
    test_pids = []    
    for j in train_index:
        tr_pids.append(pids[j])
    for k in test_index:
        test_pids.append(pids[k])
    
    #print(f"  Train: index={train_index}")
    #print(f"  Test:  index={test_index}")

    train_pids = tr_pids[:-8]
    val_pids   = tr_pids[-8:]

    print(train_pids,val_pids)
    
    # In this block we want to read 5min avg temperature data for training, validation, and test PIDs
    x_train = []
    count = 0

    for pid in train_pids:

        fname = os.path.join(exp_dir,str(pid)+'_5temp_linIP_y.csv')
        #print('Processing pid: ',pid)
        data = np.loadtxt(fname,delimiter=',')
        d1 = data[:,0:288]
        if count == 0:
            x_train = d1
        else:
            x_train = np.concatenate((x_train,d1),axis=0)
        #print(x_train.shape)
        count += 1
    
    x_val = []
    count = 0
    for pid in val_pids:
        fname = os.path.join(exp_dir,str(pid)+'_5temp_linIP_y.csv')
        #print('Processing pid: ',pid)
        data = np.loadtxt(fname,delimiter=',')
        d1 = data[:,0:288]
        if count == 0:
            x_val = d1
        else:
            x_val = np.concatenate((x_val,d1),axis=0)
        #print(x_train.shape)
        count += 1

    x_test = []
    count = 0
    for pid in test_pids:
        fname = os.path.join(exp_dir,str(pid)+'_5temp_linIP_y.csv')
        #print('Processing pid: ',pid)
        data = np.loadtxt(fname,delimiter=',')
        d1 = data[:,0:288]
        if count == 0:
            x_test = d1
        else:
            x_test = np.concatenate((x_test,d1),axis=0)
        #print(x_test.shape)
        count += 1

    print(x_train.shape, x_val.shape, x_test.shape)
    break

result_list = []

for e in range(8,256,8):

    # WE SHOULD SET THE RANDOM SEED IN THE LOOP
    # Setting the random Seed for replication of results
    keras.utils.set_random_seed(912)

    # Convoluttional AUTOENCODER
    enc_dim = e

    # Define the input shape
    input_img = keras.Input(shape=(288,1))
    # ENCODER PART
    # First 1D convolutional Layer: 16 filters of length 3 
    conv1 = layers.Conv1D(64, 7, activation='LeakyReLU', padding='same', kernel_initializer='glorot_uniform')(input_img)
    pool1 = layers.MaxPooling1D(2, padding='same')(conv1)
    # Second 1D convolutional Layer: 8 filters of length 3 
    conv2 = layers.Conv1D(32, 5, activation='LeakyReLU',padding='same', kernel_initializer='glorot_uniform')(pool1)
    pool2 = layers.MaxPooling1D(2, padding='same')(conv2)
    # Third 1D convolutional Layer: 16 filters of length 3 
    conv3 = layers.Conv1D(16, 3, activation='LeakyReLU', padding='same', kernel_initializer='glorot_uniform')(pool2)
    pool3 = layers.MaxPooling1D(2, padding='same')(conv3) 
    # Flatten the output of all convolutional filters and feed it to a dense fully-connected layer
    flat1 = layers.Flatten()(pool3)
    # Encoded Representatio of daily temperature data
    encoded = layers.Dense(enc_dim,activation='linear', kernel_initializer='glorot_uniform')(flat1)
    enc = tf.reshape(encoded,(-1,enc_dim,1))
    # Instead of reshaping, we can use the transpose layer on the 64 bit vector.
    convT1 = layers.Conv1DTranspose(16, 3,strides=2, padding = 'same', activation = 'LeakyReLU', kernel_initializer='glorot_uniform')(enc)
    convT2 = layers.Conv1DTranspose(32, 5,strides=2, padding = 'same', activation = 'LeakyReLU', kernel_initializer='glorot_uniform')(convT1)
    convT3 = layers.Conv1DTranspose(64, 7,strides=2, padding = 'same', activation = 'LeakyReLU', kernel_initializer='glorot_uniform')(convT2)
    flat2  = layers.Flatten()(convT3)
    decoded = layers.Dense(288, activation='linear')(flat2)

    # Keras API allows us to define the model, by specifying the input and final output.
    autoencoder = keras.Model(input_img,decoded)
    autoencoder.summary()

    # Define Callbacks
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # Compile the model using Adam optimizer and mean squared error loss function.
    autoencoder.compile(optimizer=keras.optimizers.Adam(), loss='mean_absolute_error')

    checkpoint_filepath = './tmp/checkpointAllT_size_'+str(e)
    check_point = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


    # Setting the random Seed for replication of results
    #keras.utils.set_random_seed(912)
    # Train the model for 100 epocs, batch size as 32, and use validation data for hyperparameter tuning.
    history = autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=32,
                shuffle=True,
                callbacks = [early_stop,check_point],
                validation_data=(x_val, x_val))


    # Run the Evaluate function on the test dataset.
    res = autoencoder.evaluate(x_test,x_test)
    print(res)
    result_list.append(res)

print(result_list)

# Encode and decode the temperature readings
# Note that we take them from the *test* set
#encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)


# "Loss"
#plt.plot(history.history['loss'][2:])
#plt.plot(history.history['val_loss'][2:])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()


# Save the Encoder model weights, to load and generate encodings later on.
#autoencoder.save('./Conv_autoencoder_all_transpose.keras')
#encoder.save('./Conv_encoder_all_transpose.keras')





