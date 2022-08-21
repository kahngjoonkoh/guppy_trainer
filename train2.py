#!/usr/bin/env python3
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

#%%
model.fit(
  train_dataset.batch(32),
  callbacks=[tensorboard_callback, cp_callback],
  validation_data=test_dataset.batch(8),
  steps_per_epoch=32,
  validation_steps=8,
  epochs=1
)

#%%
Xs, Ys = data_generator.load("datasets/test.pgn", engine_path, 10)
print(Xs, Ys)
X_train = np.array(Xs)
y_train = np.array(Ys)
# X_test
# y_test

#%%
loss, acc = model.evaluate(test_dataset, batch_size=8, steps=1, verbose=2)
print(acc)

#%%
model.loss_tracker.result().numpy()
#%%
val = model.predict_on_batch(test_dataset.batch(8))
print(val)
#%%
model = build_model()

model.compile(
    optimizer=keras.optimizers.Adadelta(),
    loss='mse',
    metrics=[
        tf.keras.metrics.MeanSquaredError(),
        tf.keras.metrics.MeanAbsoluteError()
    ]
)

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest).expect_partial()
#%%

board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
print(get_nnue_indeces(board))
val = model.predict(get_nnue_indeces(board))
print(val)


#%%
loss, acc = model.evaluate(test_dataset, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#%%
model.save_weights("ckpt")

#%%
load_status = model.load_weights("ckpt")
load_status.assert_consumed()

loss, acc = model.evaluate(test_dataset.batch(8), verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))