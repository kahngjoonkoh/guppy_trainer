# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:26:28 2022

@author: johnk
"""
import os
import numpy as np
import tensorflow as tf
import struct
import chess
import functools
from enum import Enum
from enum import IntFlag
from datetime import datetime
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from halfkp import get_nnue_indeces
import data_generator

DENSE_LAYERS_WIDTH = 32
CLIP = 1.0
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

#%% Load Datasets

X_train
y_train

X_test
y_test

#%%
def build_model_inputs():
    return keras.Input(shape=(768,), sparse=True)

def build_hidden_layers(inputs):
    hidden_layer_1 = layers.Dense(DENSE_LAYERS_WIDTH, name='hidden_layer_1')
    hidden_layer_2 = layers.Dense(DENSE_LAYERS_WIDTH, name='hidden_layer_2')
    activation_1 = layers.ReLU(max_value=CLIP)
    activation_2 = layers.ReLU(max_value=CLIP)
    return activation_2(hidden_layer_2(activation_1(hidden_layer_1(inputs))))

def build_output_layer(inputs):
    output_layer = layers.Dense(1, name='output_layer')
    return output_layer(inputs)

def build_model():
    inputs = build_model_inputs()
    outputs = build_output_layer(build_hidden_layers(inputs))
    return keras.Model(inputs=inputs, outputs=outputs)

