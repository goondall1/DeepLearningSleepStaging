
import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
tf.get_logger().setLevel('INFO')

import numpy as np


from tensorflow import keras

from keras.utils import plot_model
from keras.models import Model
from keras.layers import LSTM, RepeatVector
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate

def SleepStageCNN(in_size, out_size, optimizer, learning_rate, dropout, kernel_size_first, kernel_size_rest, pool_size, activation, double_convs, global_pool):
    kernel_size_first = int(kernel_size_first)
    kernel_size_rest = int(kernel_size_rest)
    pool_size = (int(pool_size))
    
    model = keras.models.Sequential([
        
        keras.layers.Input(shape=(in_size,1)),
        
        keras.layers.Conv1D(16, 3, activation=activation, padding='same'),
        keras.layers.MaxPool1D(pool_size, strides=2),

        keras.layers.Conv1D(32, 3, activation=activation,padding='same'),
        keras.layers.MaxPool1D(pool_size, strides=2),
        
        keras.layers.Conv1D(64, 3, activation=activation,padding='same'),        
        keras.layers.GlobalAveragePooling1D(),

        keras.layers.Flatten(),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(32,activation=activation),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(10,activation=activation),
        keras.layers.Dense(out_size, activation='softmax')
    ])
    
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer = optimizer,
        metrics=['accuracy'],
    )

    return model


def SimpleSleepStageCNN(in_size, out_size, learning_rate, dropout, kernel_size, channels):
    kernel_size = (int(kernel_size))
    channels = (int(channels))
    activation = 'relu'
    pool_size = 2
    print("I am here")
    model = keras.models.Sequential([
        
        keras.layers.Input(shape=(in_size,1)),
        
        keras.layers.Conv1D(2**channels, kernel_size+2, strides=3, activation=activation, padding='same'),
        keras.layers.Conv1D(2**channels, kernel_size, activation=activation, padding='same'),
        keras.layers.MaxPool1D(pool_size, strides=2),

        keras.layers.Conv1D(2*2**channels, kernel_size, activation=activation, padding='same'),
        keras.layers.MaxPool1D(pool_size, strides=2),
        
        keras.layers.Conv1D(4*2**channels, kernel_size, activation=activation, padding='same'), 
        keras.layers.MaxPool1D(pool_size, strides=2),
        keras.layers.GlobalAveragePooling1D(), 

        keras.layers.Flatten(),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(64,activation=activation),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(10,activation=activation),
        keras.layers.Dense(out_size, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer = optimizer,
        metrics=['accuracy']
    )

    return model

def res_block(X, f, kernel_size, dropout, activation, stride=2):
    
    shortcut = X
    ###Main Path###
    
    X = keras.layers.BatchNormalization(axis=1)(X)
    X = keras.layers.Activation(activation)(X)
    X = keras.layers.Dropout(dropout)(X)
    X = keras.layers.Conv1D(f[0], kernel_size, padding='same',strides=stride)(X)
    
    X = keras.layers.BatchNormalization(axis=1)(X)
    X = keras.layers.Activation(activation)(X)
    X = keras.layers.Dropout(dropout)(X)
    X = keras.layers.Conv1D(f[1], kernel_size, padding='same',strides=1)(X)

    
    ###shortcut Path###
    shortcut = keras.layers.MaxPool1D(pool_size=stride, strides=stride, padding='same')(shortcut)
    X = keras.layers.Add()([X, shortcut])
    X = keras.layers.Activation(activation)(X)
    return X


    
def SleepStageResCNN(in_size, out_size, learning_rate, dropout, kernel_size, ):
    kernel_size = (int(kernel_size))
    pool_size = (int(pool_size))
    activation = 'relu'
    pool_size = 3
    
    # Define the input as a tensor with shape 
    X_input = Input((in_size,1))
    

    # Stage 1
    X = keras.layers.Conv1D(64, 11, padding='same', strides=1)(X_input)
    
    X = res_block(X, [4*2*channels,4*2*channels], kernel_size, dropout, activation)
    X = res_block(X, [4*2*channels,4*2*channels], kernel_size, dropout, activation)
    X = res_block(X, [4*2*channels,4*2*channels], kernel_size, dropout, activation)
    
    X = keras.layers.Conv1D(128, 1, padding='same', strides=1)(X)
    
    X = res_block(X, [8*2*channels,8*2*channels], kernel_size, dropout, activation)
    X = res_block(X, [8*2*channels,8*2*channels], kernel_size, dropout, activation)
    X = res_block(X, [8*2*channels,8*2*channels], kernel_size, dropout, activation)
    
    X = GlobalAveragePooling1D()(X)
    X = Flatten()(X)
    
    X = keras.layers.Dense(4*2*channels,activation=activation)(X)
    X = keras.layers.Dense(out_size, activation='softmax')(X)
    
    model = keras.Model(inputs=X_input, outputs=X)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer = optimizer,
        metrics=['accuracy'],
    )

    return model

