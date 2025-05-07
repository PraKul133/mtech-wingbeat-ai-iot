import os
import math
import numpy as np
import librosa
from keras.models import Model
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Lambda
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.applications import Xception
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import shuffle
import xgboost as xgb
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Constants
seed = 2018
np.random.seed(seed)
tf.random.set_seed(seed)
SR = 8000
N_FFT = 256
HOP_LEN = N_FFT // 6
input_shape = (129, 120, 3)  # Updated to 3 channels for Xception
batch_size = 32
epochs = 10
es_patience = 7
rlr_patience = 3
target_names = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae', 
                'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']

# Specify the dataset path here
dataset_path = 'E:/archive/Wingbeats'  # Replace with your actual dataset path

# Prepare data paths
X_names = []
y = []
target_count = []

# Loop through each target (species) and collect file paths
for i, target in enumerate(target_names):
    target_count.append(0)
    path = os.path.join(dataset_path, target)
    for root, dirs, files in os.walk(path, topdown=False):
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext.lower() == '.wav':
                file_path = os.path.join(root, filename)
                y.append(i)
                X_names.append(file_path)
                target_count[i] += 1
    print(f"{target} #recs = {target_count[i]}")

print(f"Total #recs = {len(y)}")

# Shuffle and split data into training and testing
X_names, y = shuffle(X_names, y, random_state=seed)
X_train, X_test, y_train, y_test = train_test_split(
    X_names, y, stratify=y, test_size=0.20, random_state=seed)
print(f"Train #recs = {len(X_train)}")
print(f"Test #recs = {len(X_test)}")

# Data generation functions
def audio_to_spectrogram(file_path):
    data, _ = librosa.load(file_path, sr=SR)
    spectrogram = librosa.stft(data, n_fft=N_FFT, hop_length=HOP_LEN)
    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram))
    spectrogram = np.flipud(spectrogram)
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    spectrogram = np.repeat(spectrogram, 3, axis=-1)
    return spectrogram

def train_generator():
    while True:
        for start in range(0, len(X_train), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(X_train))
            train_batch = X_train[start:end]
            labels_batch = y_train[start:end]
            for i in range(len(train_batch)):
                data = audio_to_spectrogram(train_batch[i])
                x_batch.append(data)
                y_batch.append(labels_batch[i])
            x_batch = np.array(x_batch, dtype=np.float32)
            y_batch = np.array(y_batch, dtype=np.float32)
            y_batch = to_categorical(y_batch, len(target_names))
            yield x_batch, y_batch

def valid_generator():
    while True:
        for start in range(0, len(X_test), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(X_test))
            test_batch = X_test[start:end]
            labels_batch = y_test[start:end]
            for i in range(len(test_batch)):
                data = audio_to_spectrogram(test_batch[i])
                x_batch.append(data)
                y_batch.append(labels_batch[i])
            x_batch = np.array(x_batch, dtype=np.float32)
            y_batch = np.array(y_batch, dtype=np.float32)
            y_batch = to_categorical(y_batch, len(target_names))
            yield x_batch, y_batch

# Define the input shape
img_input = Input(shape=input_shape)

def normalize_input(x):
    return (x / 255.0) * 2.0 - 1.0

x = Lambda(normalize_input)(img_input)
xception_model = Xception(include_top=False, weights='imagenet', input_tensor=x)

x = GlobalAveragePooling2D(name='global_average_pooling2d')(xception_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(len(target_names), activation='softmax')(x)

model = Model(inputs=img_input, outputs=x)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=es_patience, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', patience=rlr_patience, factor=0.1, verbose=1),
    CSVLogger('training_log_xception.csv')
]

start_train_time = time.time()

history = model.fit(
    train_generator(),
    steps_per_epoch=math.ceil(len(X_train) / batch_size),
    validation_data=valid_generator(),
    validation_steps=math.ceil(len(X_test) / batch_size),
    epochs=epochs,
    callbacks=callbacks
)

end_train_time = time.time()
train_time = end_train_time - start_train_time
print(f"Training Time: {train_time:.2f} seconds")

# Save Xception model as .hdf5
model.save('xception_model.hdf5')
print("Xception model saved as xception_model.hdf5")

train_loss, train_acc = model.evaluate(train_generator(), steps=math.ceil(len(X_train) / batch_size))
print(f"Train accuracy of the Xception model: {train_acc}")

loss, xception_acc = model.evaluate(valid_generator(), steps=math.ceil(len(X_test) / batch_size))
print(f"Test accuracy of the Xception model: {xception_acc}")

# Feature Extraction for XGBoost
def extract_features_xception(model, X_data):
    feature_model = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d').output)
    features = []
    for file in X_data:
        data = audio_to_spectrogram(file)
        data = np.expand_dims(data, axis=0)
        feature = feature_model.predict(data)
        features.append(feature.flatten())
    return np.array(features)

print("Extracting features from training data...")
X_train_features = extract_features_xception(model, X_train)
print("Extracting features from testing data...")
X_test_features = extract_features_xception(model, X_test)

start_test_time = time.time()

print("Training XGBoost classifier...")
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=seed)
xgb_model.fit(X_train_features, y_train)

# Save XGBoost model as pickle file
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("XGBoost model saved as xgboost_model.pkl")

print("Predicting with XGBoost classifier...")
y_pred = xgb_model.predict(X_test_features)
xgb_acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of the hybrid model (Xception + XGBoost): {xgb_acc}")

end_test_time = time.time()
test_time = end_test_time - start_test_time
print(f"Testing Time: {test_time:.2f} seconds")

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision (XGBoost): {precision:.4f}")
print(f"Recall (XGBoost): {recall:.4f}")
print(f"F1-Score (XGBoost): {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix for Hybrid Model (Xception + XGBoost)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()
