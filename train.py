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

from halfkp import get_halfkp_indeces
import data_generator

FEATURE_TRANSFORMER_HALF_DIMENSIONS = 256
DENSE_LAYERS_WIDTH = 32

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

#%% Build Model
def build_model_inputs():
    return keras.Input(shape=(41024,), sparse=True, dtype=tf.int8), keras.Input(shape=(41024,), sparse=True, dtype=tf.int8)

def build_feature_transformer(inputs1, inputs2):
    ft_dense_layer = layers.Dense(FEATURE_TRANSFORMER_HALF_DIMENSIONS, name='feature_transformer')
    clipped_relu = layers.ReLU(max_value=127)
    transformed1 = clipped_relu(ft_dense_layer(inputs1))
    transformed2 = clipped_relu(ft_dense_layer(inputs2))
    return tf.keras.layers.Concatenate()([transformed1, transformed2])

# NNUE ReLU formula -> f(x) = max(0, min(127, x / 64))
def nnue_relu(x):
    return tf.maximum(0, tf.minimum(127, tf.dtypes.cast(tf.math.floordiv(x, 64), tf.int32)))

def build_hidden_layers(inputs):
    hidden_layer_1 = layers.Dense(DENSE_LAYERS_WIDTH, name='hidden_layer_1')
    hidden_layer_2 = layers.Dense(DENSE_LAYERS_WIDTH, name='hidden_layer_2')
    activation_1 = layers.Activation(nnue_relu)
    activation_2 = layers.Activation(nnue_relu)
    return activation_2(hidden_layer_2(activation_1(hidden_layer_1(inputs))))

def build_output_layer(inputs):
    output_layer = layers.Dense(1, name='output_layer')
    return output_layer(inputs)

def build_model():
    inputs1, inputs2 = build_model_inputs()
    outputs = build_output_layer(build_hidden_layers(build_feature_transformer(inputs1, inputs2)))
    return keras.Model(inputs=[inputs1, inputs2], outputs=outputs)

model = build_model()

print(model.summary())
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

#%% Define Datasets
#Settings
pgn_path = "datasets/lichess_elite_2022-02.pgn"
engine_path = "stockfish_15_x64_avx2.exe"


tf.random.set_seed(int(datetime.utcnow().timestamp()))

train_dataset = tf.data.Dataset.from_generator(
  data_generator.generate, args=[pgn_path, engine_path],
  output_signature=((tf.SparseTensorSpec((41024,)), tf.SparseTensorSpec((41024,))), tf.TensorSpec(()))
)

test_dataset = tf.data.Dataset.from_generator(
  data_generator.generate, args=[pgn_path, engine_path],
  output_signature=((tf.SparseTensorSpec((41024,)), tf.SparseTensorSpec((41024,))), tf.TensorSpec(()))
)

#%%

print(np.stack(list(train_dataset.batch(8))))
# print(tf.sparse.to_dense(test_dataset.batch(8)[0]).numpy())
# print(test_dataset.as_numpy_iterator())
      
#%% Compile Model
#Settings
checkpoint_path = "training/training_2/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# tensorboard --logdir logs/scalars
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=logdir,
    update_freq=100,
    histogram_freq=1,
)
                    


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 monitor='loss',
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=1,
                                                 mode='min')

model.compile(
    optimizer=keras.optimizers.Adadelta(),
    loss='mse',
    metrics=[
        tf.keras.metrics.MeanSquaredError(),
        tf.keras.metrics.MeanAbsoluteError()
    ]
)

#%% Train Model
#default train batch: 32, validation batch: 8, steps per epoch: 256, epochs, 100
model.fit(
    train_dataset.batch(32),
    callbacks=[tensorboard_callback, cp_callback],
    validation_data=test_dataset.batch(8),
    steps_per_epoch=16,
    validation_steps=8,
    epochs=1
)

#%% Evaluate

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
model.load_weights(latest)

loss, acc = model.evaluate(test_dataset.batch(8), verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#%%
board = chess.Board("1rb1k2r/p3ppb1/2pp1np1/q5Np/1p1PPN1P/n3BP2/PPP3P1/K1QR1B1R b k - 7 15")
X = get_halfkp_indeces(board)
def nn_value_to_centipawn(nn_value):
    return (((nn_value // 16) * 100) // 208) / 100

prediction = model.predict(get_halfkp_indeces(chess.Board()))[0][0]
print(f'Start position prediction: {nn_value_to_centipawn(prediction)}')

prediction = model.predict(get_halfkp_indeces(chess.Board('rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')))[0][0]
print(f'Position with big white advantage prediction: {nn_value_to_centipawn(prediction)}')

prediction = model.predict(get_halfkp_indeces(chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1')))[0][0]
print(f'Position with big black advantage prediction: {nn_value_to_centipawn(prediction)}')

#%% Load Model
model = keras.models.load_model(checkpoint_path)
# np.testing.assert_allclose(model.predict(x_train), new_model.predict(x_train), 1e-5)

# Re-evaluate the model
loss, acc = model.evaluate(test_dataset.batch(32), verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#%%

#%% Load Models
# Create a new model 
latest = tf.train.latest_checkpoint(checkpoint_dir)
# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_dataset.batch(32), verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#%% Load NNUE