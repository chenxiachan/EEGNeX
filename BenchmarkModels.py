from keras.models import Model, Sequential

from keras.layers import Input, GRU, ConvLSTM2D, TimeDistributed, AvgPool2D, AvgPool1D, SeparableConv2D, DepthwiseConv2D, AveragePooling1D, Add, AveragePooling2D
from keras.layers import Dense, Flatten, Dropout, LSTM
# from keras.layers.merge import concatenate
from keras.layers import concatenate
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Activation, LayerNormalization, Reshape
# import tensorflow_addons as tfa

from keras.constraints import max_norm
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model


from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow import Tensor

import csv
from datetime import datetime

## 改图片名称

##### Basic models
def Single_LSTM(n_timesteps, n_features, n_outputs):
    # define model
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_features, n_timesteps), return_sequences=True, kernel_regularizer=l2(0.0001)))
    model.add(LSTM(100, return_sequences=True, kernel_regularizer=l2(0.0001)))
    model.add(LSTM(100, kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


def Single_GRU(n_timesteps, n_features, n_outputs):
    # define model
    model = Sequential()
    model.add(GRU(100, input_shape=(n_features, n_timesteps), return_sequences=True, kernel_regularizer=l2(0.0001)))
    model.add(GRU(100, return_sequences=True, kernel_regularizer=l2(0.0001)))
    model.add(GRU(100, kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(n_outputs, activation='softmax'))
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='Single_GRU.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


def OneD_CNN(n_timesteps, n_features, n_outputs):
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, input_shape=(n_features, n_timesteps)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    
    model.add(Conv1D(filters=64, kernel_size=3))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(Conv1D(filters=64, kernel_size=3))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(Dropout(0.5))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(n_outputs, activation='softmax'))
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='OneD_CNN.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model 

def OneD_CNN_Dilated(n_timesteps, n_features, n_outputs):
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, input_shape=(n_features, n_timesteps)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    
    model.add(Conv1D(filters=64, kernel_size=3, dilation_rate=2))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(Conv1D(filters=64, kernel_size=3))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(Dropout(0.5))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(n_outputs, activation='softmax'))
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='OneD_CNN_Dilated.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model 

def OneD_CNN_Causal(n_timesteps, n_features, n_outputs):
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, padding='causal', input_shape=(n_features, n_timesteps)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(Conv1D(filters=64, kernel_size=3, padding='causal'))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(Conv1D(filters=64, kernel_size=3, padding='causal'))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(n_outputs, activation='softmax'))
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='OneD_CNN_Causal.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model 

    
def OneD_CNN_CausalDilated(n_timesteps, n_features, n_outputs):
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, padding='causal', input_shape=(n_features, n_timesteps)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=2))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(Conv1D(filters=64, kernel_size=3, padding='causal'))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(n_outputs, activation='softmax'))
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='OneD_CNN_CausalDilated.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model 


def TwoD_CNN(n_timesteps, n_features, n_outputs):
    # define model
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(1, 3), input_shape=(1, n_features, n_timesteps)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(Conv2D(filters=64, kernel_size=(1, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(Conv2D(filters=64, kernel_size=(1, 3)))
    model.add(Dropout(0.5))
    model.add(AvgPool2D(pool_size=(2, 1), padding='same'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(n_outputs, activation='softmax'))
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='TwoD_CNN.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def TwoD_CNN_Dilated(n_timesteps, n_features, n_outputs):
    # define model
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(1, 3), input_shape=(1, n_features, n_timesteps)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(Conv2D(filters=64, kernel_size=(1, 3), dilation_rate=2))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(Conv2D(filters=64, kernel_size=(1, 3)))
    model.add(Dropout(0.5))
    model.add(AvgPool2D(pool_size=(2, 1), padding='same'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(n_outputs, activation='softmax'))
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='TwoD_CNN_Dilated.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def TwoD_CNN_Separable(n_timesteps, n_features, n_outputs):
    # define model
    model = Sequential()
    model.add(SeparableConv2D(filters=64, kernel_size=(1, 3), input_shape=(1, n_features, n_timesteps)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(SeparableConv2D(filters=64, kernel_size=(1, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(SeparableConv2D(filters=64, kernel_size=(1, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(Dropout(0.5))
    model.add(AvgPool2D(pool_size=(2, 1), padding='same'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(n_outputs, activation='softmax'))
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='TwoD_CNN.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def TwoD_CNN_Depthwise(n_timesteps, n_features, n_outputs):
    # define model
    model = Sequential()
    model.add(DepthwiseConv2D(kernel_size=(1, 3), input_shape=(1, n_features, n_timesteps)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(SeparableConv2D(filters=64, kernel_size=(1, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(SeparableConv2D(filters=64, kernel_size=(1, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(Dropout(0.5))
    model.add(AvgPool2D(pool_size=(2, 1), padding='same'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(n_outputs, activation='softmax'))
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='TwoD_CNN.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

##### 
##### 
##### 
##### Combined models
##### 
##### 
##### 

def CNN_LSTM(n_timesteps, n_features, n_outputs):
    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3), input_shape=(1, n_features, n_timesteps)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(480, kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(n_outputs, activation='softmax'))
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='CNN_LSTM.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def CNN_GRU(n_timesteps, n_features, n_outputs):
    
    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3), input_shape=(1, n_features, n_timesteps)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(GRU(480, kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(n_outputs, activation='softmax'))
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='CNN_GRU.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def Single_ConvLSTM2D(n_timesteps, n_features, n_outputs):
    # kernel regulate good
    # define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), input_shape=(1, 1, n_features, n_timesteps), kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(n_outputs, activation='softmax'))
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='Single_ConvLSTM2D.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

##### 
##### 
##### 
##### Advanced models
##### 
##### 
##### 

def EEGNet_8_2(n_timesteps, n_features, n_outputs):
    # define model 原版 EEGNet_8,2
    model = Sequential()
    model.add(Input(shape=(1, n_features, n_timesteps)))
    # 可以分为三个kernelsize
    model.add(Conv2D(filters=8, kernel_size=(1, 64), use_bias = False, padding='same', data_format="channels_first"))
    model.add(BatchNormalization())
    
    model.add(DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2,  use_bias = False, depthwise_constraint=max_norm(1.), data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(AvgPool2D(pool_size=(1, 4), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))
    
    model.add(SeparableConv2D(filters=16, kernel_size=(1, 16),  use_bias = False, padding='same', data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(AvgPool2D(pool_size=(1, 8), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(n_outputs, kernel_constraint=max_norm(0.25)))
    model.add(Activation(activation='softmax'))
    
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='EEGNet.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model 

def EEGNet_4_2(n_timesteps, n_features, n_outputs):
    # define model 原版 EEGNet_8,2
    model = Sequential()
    model.add(Input(shape=(1, n_features, n_timesteps)))
    # 可以分为三个kernelsize
    model.add(Conv2D(filters=4, kernel_size=(1, 64), use_bias = False, padding='same', data_format="channels_first"))
    model.add(BatchNormalization())
    
    model.add(DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2,  use_bias = False, depthwise_constraint=max_norm(1.), data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(AvgPool2D(pool_size=(1, 4), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))
    
    model.add(SeparableConv2D(filters=8, kernel_size=(1, 16),  use_bias = False, padding='same', data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(AvgPool2D(pool_size=(1, 8), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(n_outputs, kernel_constraint=max_norm(0.25)))
    model.add(Activation(activation='softmax'))
    
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='EEGNet.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model 


# add conv more for feature extraction
def test1(n_timesteps, n_features, n_outputs):
    # define model 原版 EEGNet_8,2
    model = Sequential()
    model.add(Input(shape=(1, n_features, n_timesteps)))
    # 可以分为三个kernelsize
    model.add(Conv2D(filters=8, kernel_size=(1, 64), use_bias = False, padding='same', data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(Conv2D(filters=8, kernel_size=(1, 64), use_bias = False, padding='same', data_format="channels_first"))
    model.add(BatchNormalization())
    
    model.add(DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2,  use_bias = False, depthwise_constraint=max_norm(1.), data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(AvgPool2D(pool_size=(1, 4), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))
    
    model.add(SeparableConv2D(filters=16, kernel_size=(1, 16),  use_bias = False, padding='same', data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(AvgPool2D(pool_size=(1, 8), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(n_outputs, kernel_constraint=max_norm(0.25)))
    model.add(Activation(activation='softmax'))
    
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='EEGNet.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model 


# # wider output
# def test2(n_timesteps, n_features, n_outputs):
#     # define model 原版
#     model = Sequential()
#     model.add(Input(shape=(1, n_features, n_timesteps)))
#     # 可以分为三个kernelsize
#     model.add(Conv2D(filters=8, kernel_size=(1, 32), use_bias = False, padding='same', data_format="channels_first"))
#     model.add(BatchNormalization())
#     model.add(Activation(activation='elu'))
#     model.add(Conv2D(filters=8, kernel_size=(1, 32), use_bias = False, padding='same', data_format="channels_first"))
#     model.add(BatchNormalization())
#     model.add(Activation(activation='elu'))

#     model.add(DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2, use_bias = False, depthwise_constraint=max_norm(1.), data_format="channels_first"))
#     model.add(BatchNormalization())
#     model.add(Activation(activation='elu'))
#     model.add(AvgPool2D(pool_size=(1, 4), padding='same', data_format="channels_first"))
#     model.add(Dropout(0.5))
    
#     model.add(SeparableConv2D(filters=16, kernel_size=(1, 16),  use_bias = False, padding='same', data_format="channels_first"))
#     model.add(BatchNormalization())
#     model.add(Activation(activation='elu'))
#     model.add(AvgPool2D(pool_size=(1, 8), padding='same', data_format="channels_first"))
#     model.add(Dropout(0.5))
    
#     model.add(Flatten())
#     model.add(Dense(n_outputs, kernel_constraint=max_norm(0.25)))
#     model.add(Activation(activation='softmax'))
    
#     # save a plot of the model
#     plot_model(model, show_shapes=True, to_file='EEGNet.png')
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
#     return model 



# no SeparableConv2D and pooling, 2 Conv 
def test2(n_timesteps, n_features, n_outputs):
    # define model 原版
    model = Sequential()
    model.add(Input(shape=(1, n_features, n_timesteps)))
    # 可以分为三个kernelsize
    model.add(Conv2D(filters=8, kernel_size=(1, 64), use_bias = False, padding='same', data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(Conv2D(filters=8, kernel_size=(1, 64), use_bias = False, padding='same', data_format="channels_first"))
    model.add(BatchNormalization())
    
    model.add(DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2,  use_bias = False, depthwise_constraint=max_norm(1.), data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    # model.add(AvgPool2D(pool_size=(1, 4), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))


    model.add(Conv2D(filters=16, kernel_size=(1, 16), use_bias = False, padding='same', dilation_rate=(1, 2), data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    
    model.add(Conv2D(filters=8, kernel_size=(1, 16), use_bias = False, padding='same', dilation_rate=(1, 4),  data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(n_outputs, kernel_constraint=max_norm(0.25)))
    model.add(Activation(activation='softmax'))
    
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='EEGNet.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model 


# wider output 
def test3(n_timesteps, n_features, n_outputs):
    # define model 原版
    model = Sequential()
    model.add(Input(shape=(1, n_features, n_timesteps)))
    # 可以分为三个kernelsize
    model.add(Conv2D(filters=8, kernel_size=(1, 64), use_bias = False, padding='same', data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(Conv2D(filters=8, kernel_size=(1, 64), use_bias = False, padding='same', data_format="channels_first"))
    model.add(BatchNormalization())
    
    model.add(DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2,  use_bias = False, depthwise_constraint=max_norm(1.), data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    # model.add(AvgPool2D(pool_size=(1, 4), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))


    model.add(Conv2D(filters=32, kernel_size=(1, 16), use_bias = False, padding='same', dilation_rate=(1, 2), data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    
    model.add(Conv2D(filters=8, kernel_size=(1, 16), use_bias = False, padding='same', dilation_rate=(1, 4),  data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(n_outputs, kernel_constraint=max_norm(0.25)))
    model.add(Activation(activation='softmax'))
    
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='EEGNet.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model 




# wider output
def test4(n_timesteps, n_features, n_outputs):
    # define model 原版
    model = Sequential()
    model.add(Input(shape=(1, n_features, n_timesteps)))
    # 可以分为三个kernelsize
    model.add(Conv2D(filters=8, kernel_size=(1, 64), use_bias = False, padding='same', data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=(1, 64), use_bias = False, padding='same', data_format="channels_first"))
    model.add(BatchNormalization())
    
    model.add(DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2,  use_bias = False, depthwise_constraint=max_norm(1.), data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    # model.add(AvgPool2D(pool_size=(1, 4), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))


    model.add(Conv2D(filters=32, kernel_size=(1, 16), use_bias = False, padding='same', dilation_rate=(1, 2), data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    
    model.add(Conv2D(filters=8, kernel_size=(1, 16), use_bias = False, padding='same', dilation_rate=(1, 4),  data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(n_outputs, kernel_constraint=max_norm(0.25)))
    model.add(Activation(activation='softmax'))
    
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='EEGNet.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model 

# add pool
def EEGNeX_8_32(n_timesteps, n_features, n_outputs):
    # define model 原版
    model = Sequential()
    model.add(Input(shape=(1, n_features, n_timesteps)))
    # 可以分为三个kernelsize
    model.add(Conv2D(filters=8, kernel_size=(1, 64), use_bias = False, padding='same', data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=(1, 64), use_bias = False, padding='same', data_format="channels_first"))
    model.add(BatchNormalization())

    model.add(DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2, use_bias = False, depthwise_constraint=max_norm(1.), data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(AvgPool2D(pool_size=(1, 4), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))

    
    model.add(Conv2D(filters=32, kernel_size=(1, 16), use_bias = False, padding='same', dilation_rate=(1, 2), data_format='channels_first'))
    model.add(BatchNormalization())
    # model.add(Activation(activation='elu'))
    
    model.add(Conv2D(filters=8, kernel_size=(1, 16), use_bias = False, padding='same', dilation_rate=(1, 4),  data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(AvgPool2D(pool_size=(1, 4), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(n_outputs, kernel_constraint=max_norm(0.25)))
    model.add(Activation(activation='softmax'))
    
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='EEGNeX_8_32.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model 


def test7(n_timesteps, n_features, n_outputs):
    # define model 原版
    model = Sequential()
    model.add(Input(shape=(1, n_features, n_timesteps)))
    # 可以分为三个kernelsize
    model.add(Conv2D(filters=8, kernel_size=(1, 32), use_bias = False, padding='same', data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=(1, 32), use_bias = False, padding='same', data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))

    model.add(DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2, use_bias = False, depthwise_constraint=max_norm(1.), data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(AvgPool2D(pool_size=(1, 4), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))

    
    model.add(Conv2D(filters=32, kernel_size=(1, 16), use_bias = False, padding='same', dilation_rate=(1, 2), data_format='channels_first'))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    
    model.add(Conv2D(filters=8, kernel_size=(1, 16), use_bias = False, padding='same', dilation_rate=(1, 4),  data_format='channels_first'))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(n_outputs, kernel_constraint=max_norm(0.25)))
    model.add(Activation(activation='softmax'))
    
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='EEGNeX_8_32.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    return model 


def test8(n_timesteps, n_features, n_outputs):
    # define model 原版
    model = Sequential()
    model.add(Input(shape=(1, n_features, n_timesteps)))
    # 可以分为三个kernelsize
    model.add(Conv2D(filters=8, kernel_size=(1, 32), use_bias = False, padding='same', data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))

    model.add(DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2, use_bias = False, depthwise_constraint=max_norm(1.), data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    # model.add(AvgPool2D(pool_size=(1, 4), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))

    model.add(SeparableConv2D(filters=16, kernel_size=(1, 16),  use_bias = False, padding='same', data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    # model.add(AvgPool2D(pool_size=(1, 8), padding='same', data_format="channels_first"))
    # model.add(Dropout(0.5))
    
    model.add(Conv2D(filters=8, kernel_size=(1, 16), use_bias = False, padding='same', dilation_rate=(1, 4),  data_format='channels_first'))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(n_outputs, kernel_constraint=max_norm(0.25)))
    model.add(Activation(activation='softmax'))
    
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='EEGNeX_8_32.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model 


def test9(n_timesteps, n_features, n_outputs):
    # define model 原版
    model = Sequential()
    model.add(Input(shape=(1, n_features, n_timesteps)))
    # 可以分为三个kernelsize
    model.add(Conv2D(filters=8, kernel_size=(1, 32), use_bias = False, padding='same', data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))

    model.add(DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2, use_bias = False, depthwise_constraint=max_norm(1.), data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    # model.add(AvgPool2D(pool_size=(1, 4), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))

    # model.add(SeparableConv2D(filters=8, kernel_size=(1, 16),  use_bias = False, padding='same', data_format="channels_first"))
    # model.add(BatchNormalization())
    # model.add(Activation(activation='elu'))
    # # model.add(AvgPool2D(pool_size=(1, 8), padding='same', data_format="channels_first"))
    # model.add(Dropout(0.5))
    
    model.add(Conv2D(filters=8, kernel_size=(1, 16), use_bias = False, padding='same', dilation_rate=(1, 4),  data_format='channels_first'))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(n_outputs, kernel_constraint=max_norm(0.25)))
    model.add(Activation(activation='softmax'))
    
    # save a plot of the model
    plot_model(model, show_shapes=True, to_file='EEGNeX_8_32.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model 