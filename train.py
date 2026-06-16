import numpy as np
import tensorflow as tf
import json
import os
import random
from sklearn.metrics import confusion_matrix
from preprocess import preprocess

os.makedirs("model", exist_ok=True)

# -----------------------
# REPRODUCIBILITY (REAL-WORLD FIX)
# -----------------------
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# -----------------------
# LOAD DATA
# -----------------------
X_train, X_test, y_train, y_test = preprocess()

X_train = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

y_train = np.array(y_train)
y_test = np.array(y_test)

# -----------------------
# MODEL (CLEAN IDS VERSION)
# -----------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], 1)),

    tf.keras.layers.Conv1D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(32, activation='relu'),

    # KEEP softmax (multi-class ready architecture)
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------
# TRAIN
# -----------------------
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=256,
    validation_data=(X_test, y_test),
    verbose=1
)

# -----------------------
# PREDICT
# -----------------------
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

# -----------------------
# SAVE MODEL
# -----------------------
model.save("model/model.h5")

# -----------------------
# SAVE HISTORY (SAFE FORMAT FIX)
# -----------------------
with open("model/history.json", "w") as f:
    json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)

# -----------------------
# SAVE TEST DATA (FOR REAL APP INFERENCE)
# -----------------------
np.save("model/X_test.npy", X_test)
np.save("model/y_test.npy", y_test)
np.save("model/y_pred.npy", y_pred)

# -----------------------
# CONFUSION MATRIX (optional logging)
# -----------------------
cm = confusion_matrix(y_test, y_pred)
np.save("model/confusion_matrix.npy", cm)

print("✅ Training complete (Production-ready IDS pipeline)")