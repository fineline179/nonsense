#%%
import os
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import confusion_matrix, plot_confusion_matrix

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
  Dense,
  Input,
  Dropout,
  LSTM,
  Activation,
  SimpleRNN,
  Bidirectional,
)
from tensorflow.keras.metrics import BinaryAccuracy, Recall, Precision

from utils.import_data import import_word_data

INPUT_PATH = "/home/fineline/projects/nonsense/data"


#%% Input data
file_list = [
  "words_labeled_0_1000.csv",
  "words_labeled_2000_3000.csv",
  "words_labeled_3000_4000.csv",
]

X, Y = import_word_data(INPUT_PATH, file_list)

print(len(X))

skf = StratifiedKFold(n_splits=5)

TEST_FRAC = 0.2
X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=TEST_FRAC, shuffle=False)
# X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=TEST_FRAC, random_state=42,
#                                           shuffle=True, stratify=Y)
Y_tr, Y_ts = np.asarray(Y_tr), np.asarray(Y_ts)

wordLen = len(X_tr[0])

# covert string to list of integer char reps: 'abce' -> [0, 1, 2, 4]
word_to_indices = lambda x: [ord(i) - 97 for i in list(x)]

# input strings converted to index format
X_tr_indices = np.array([word_to_indices(i) for i in X_tr])
X_ts_indices = np.array([word_to_indices(i) for i in X_ts])

# number of negative and pos examples in train set
neg_tr = np.count_nonzero(Y_tr == 0)
pos_tr = np.count_nonzero(Y_tr == 1)


#%%
metrics = [
  Precision(name="precision"),
  Recall(name="recall"),
  BinaryAccuracy(name="accuracy"),
]


def RNNModel(input_shape, metrics, bidirectional=False, output_bias=None, lr=0.001):

  if output_bias is not None:
    output_bi = tf.keras.initializers.Constant(output_bias)

  word_indices = Input(input_shape, dtype="int32")
  one_hots = tf.one_hot(word_indices, depth=26)

  if bidirectional:
    X = Bidirectional(SimpleRNN(128, return_sequences=True))(one_hots)
  else:
    X = SimpleRNN(128, return_sequences=True)(one_hots)

  X = Dropout(0.5)(X)

  if bidirectional:
    X = Bidirectional(SimpleRNN(128))(X)
  else:
    X = SimpleRNN(128)(X)

  X = Dropout(0.5)(X)
  X = Dense(1, activation="sigmoid", bias_initializer=output_bi)(X)

  model = Model(inputs=word_indices, outputs=X)

  model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    metrics=metrics,
  )

  return model


#%% set us up the model
BIDIRECTIONAL = False
LR = 0.001
EPOCHS = 600
BATCH_SIZE = 1024
CLASS_WEIGHT = False

# set initial output bias to account for class imbalance
initial_bias = np.log([pos_tr / neg_tr])
model = RNNModel(
  input_shape=(wordLen,),
  metrics=metrics,
  bidirectional=BIDIRECTIONAL,
  output_bias=initial_bias,
  lr=LR,
)
model.summary()

# tensorboard setup
# log string
log_string = (
  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  + "_"
  + ("BIDIR_" if BIDIRECTIONAL else "RNN_")
  + ("LR-" + str(LR) + "_")
  + ("EP-" + str(EPOCHS) + "_")
  + ("BS-" + str(BATCH_SIZE) + "_")
  + ("CW" if CLASS_WEIGHT else "_")
)

log_dir = "logs/fit/" + log_string
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#
tf.random.set_seed(1)

neg_weight = (1 / neg_tr) * len(Y_tr) / 2.0
pos_weight = (1 / pos_tr) * len(Y_tr) / 2.0
class_weight = {0: neg_weight, 1: pos_weight} if CLASS_WEIGHT else None

model.fit(
  X_tr_indices,
  Y_tr,
  epochs=EPOCHS,
  batch_size=BATCH_SIZE,
  shuffle=True,
  validation_data=(X_ts_indices, Y_ts),
  callbacks=[tensorboard_callback],
  class_weight=class_weight,
)


#%%
pred_ts_prob = model.predict(X_ts_indices)
pred_tr_prob = model.predict(X_tr_indices)
pred_ts = (pred_ts_prob > 0.5).ravel().astype(int)
pred_tr = (pred_tr_prob > 0.5).ravel().astype(int)

confusion_matrix(Y_ts, pred_ts)


#%%
def print_res(type):
  assert type in ["TP", "FP", "FN"]
  count = 1
  cond = None
  for i in range(len(X_ts)):
    if type == "TP":
      cond = pred_ts[i] == 1 and Y_ts[i] == 1
    elif type == "FP":
      cond = pred_ts[i] == 1 and Y_ts[i] == 0
    elif type == "FN":
      cond = pred_ts[i] == 0 and Y_ts[i] == 1

    if cond:
      print(
        str(count)
        + " Expected label:"
        + str(Y_ts[i])
        + " prediction: "
        + X_ts[i]
        + " "
        + str(pred_ts[i])
      )
      count += 1

  print("\n")


print_res("TP")
print_res("FP")
print_res("FN")


#%% load in more test cases
with open(os.path.join(INPUT_PATH, "words_10000_15000.csv"), "r") as f:
  data = [x.strip() for x in f.readlines()]


#%%
X_ts_2 = np.asarray(data)
X_ts_2_indices = np.array([word_to_indices(i) for i in X_ts_2])
pred_ts_prob_2 = model.predict(X_ts_2_indices)
pred_ts_2 = (pred_ts_prob_2 > 0.5).ravel().astype(int)

out_num = 1
for i in range(len(X_ts_2)):
  x = X_ts_2_indices
  if pred_ts_2[i] == 1:
    print(str(out_num) + " prediction: " + X_ts_2[i] + " " + str(pred_ts_2[i]))
    out_num += 1
