# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:11:37 2022
using Spyder

@author: kahngjoonkoh
"""
import os
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot

import numpy as np

import data_generator

INPUT = 64 * 12
HIDDEN = 512

CLIP = 127 # or 1.0

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))


#%% Create Model

def build_model_inputs():
    return keras.Input(shape=(INPUT,), sparse=True, dtype='int16', name='input_layer')

def build_hidden_layers(inputs):
    hidden_layer_1 = layers.Dense(HIDDEN, name='hidden_layer')
    activation_1 = layers.ReLU()
    # activation_1 = layers.ReLU(max_value=CLIP) # Clipped Relu
    return activation_1(hidden_layer_1(inputs))


def build_output_layer(inputs):
    output_layer = layers.Dense(1, name='output_layer')
    return output_layer(inputs)

def build_model():
    inputs = build_model_inputs()
    outputs = build_output_layer(build_hidden_layers(inputs))
    return keras.Model(inputs=inputs, outputs=outputs)

tf.random.set_seed(int(datetime.utcnow().timestamp()))

pgn_path = "datasets/lichess_elite_2022-02.pgn"
engine_path = "stockfish_15_x64_avx2.exe"

train_dataset = tf.data.Dataset.from_generator(
  data_generator.generate, args=[pgn_path, engine_path],
  output_signature=(tf.SparseTensorSpec((768,)), tf.TensorSpec(()))
)

test_dataset = tf.data.Dataset.from_generator(
  data_generator.generate, args=[pgn_path, engine_path],
  output_signature=(tf.SparseTensorSpec((768,)), tf.TensorSpec(()))
)

model = build_model()

checkpoint_path = "training/training_11/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(
  log_dir=logdir,
  update_freq=5,
  histogram_freq=1,
)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
opt = keras.optimizers.Adadelta()
model.compile(
  optimizer=opt,
  loss='mse',
  metrics=[
    tf.keras.metrics.MeanSquaredError(),
    tf.keras.metrics.MeanAbsoluteError()
  ]
)

keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

#%% Train Model

model.fit(
  train_dataset.batch(32),
  callbacks=[tensorboard_callback, cp_callback],
  validation_data=test_dataset.batch(8),
  steps_per_epoch=32,
  validation_steps=8,
  epochs=1
)

#%% Save Model Weights

model.save_weights("ckpt")

#%% Load Model Weights

load_status = model.load_weights("ckpt")
load_status.assert_consumed()

#%% Get Model Weights

weights = [layer.get_weights() for layer in model.layers]

print(model.layers[1].weights[0], model.layers[1].weights[1])
print(model.layers[3].get_weights(), model.layers[3].weights[1])

# print(model.layers[2].get_weights(), model.layers[3].weights[0])

# for layer in model.layers:
#     print(layer.name)
#     weights = layer.get_weights()
#     for i in weights:
#         print(i[0])

#%% Define Quantized Model

quantize_model = tfmot.quantization.keras.quantize_model

q_aware_model = quantize_model(model)

q_aware_model.compile(
  optimizer=opt,
  loss='mse',
  metrics=[
    tf.keras.metrics.MeanSquaredError(),
    tf.keras.metrics.MeanAbsoluteError()
  ]
)

keras.utils.plot_model(q_aware_model, show_shapes=True, show_layer_names=True, to_file='model.png')

#%% Train Quantized Model

q_aware_model.fit(
  train_dataset.batch(32),
  callbacks=[tensorboard_callback, cp_callback],
  validation_data=test_dataset.batch(8),
  steps_per_epoch=32,
  validation_steps=8,
  epochs=1
)

#%% Quantitize Model

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

print(tflite_quant_model.layers)

#%% Convert to Guppy NNUE format.

np.savetxt("save.nnue", weights, fmt='%.18e', delimiter='', newline='\n', header='', footer='', comments='', encoding=None)
