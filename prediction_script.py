import numpy as np
import librosa
import pickle
from keras.models import load_model, Model

# ---------- Constants ----------
SR = 8000
N_FFT = 256
HOP_LEN = N_FFT // 6
TARGET_NAMES = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae',
                'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']

# ---------- Disease Mapping ----------
MOSQUITO_DISEASES = {
    'Ae. aegypti': ['Dengue fever', 'Zika virus', 'Chikungunya', 'Yellow fever'],
    'Ae. albopictus': ['Dengue fever', 'Chikungunya', 'Zika virus'],
    'An. gambiae': ['Malaria', 'Lymphatic filariasis'],
    'An. arabiensis': ['Malaria', 'Lymphatic filariasis'],
    'C. pipiens': ['West Nile virus', 'St. Louis encephalitis', 'Lymphatic filariasis'],
    'C. quinquefasciatus': ['Lymphatic filariasis', 'West Nile virus', 'Japanese encephalitis']
}

# ---------- Custom Function ----------
def normalize_input(x):
    return (x / 255.0) * 2.0 - 1.0

# ---------- Load Models ----------
xception_model = load_model('xception_model.hdf5', custom_objects={'normalize_input': normalize_input})
with open('xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# ---------- Feature Extractor ----------
feature_model = Model(inputs=xception_model.input,
                      outputs=xception_model.get_layer('global_average_pooling2d').output)

# ---------- Audio Preprocessing ----------
def audio_to_spectrogram(file_path):
    data, _ = librosa.load(file_path, sr=SR)
    spectrogram = librosa.stft(data, n_fft=N_FFT, hop_length=HOP_LEN)
    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram))
    spectrogram = np.flipud(spectrogram)
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    spectrogram = np.repeat(spectrogram, 3, axis=-1)
    return np.expand_dims(spectrogram.astype(np.float32), axis=0)

# ---------- Prediction Pipeline ----------
def predict_species(audio_path):
    spectrogram = audio_to_spectrogram(audio_path)
    features = feature_model.predict(spectrogram)
    features = features.flatten().reshape(1, -1)
    prediction = xgb_model.predict(features)[0]
    predicted_label = TARGET_NAMES[prediction]

    print("\n")
    print(f"🦟 Predicted Mosquito Species: {predicted_label}")
    
    # Display diseases
    if predicted_label in MOSQUITO_DISEASES:
        diseases = MOSQUITO_DISEASES[predicted_label]
        print(f"🦠 Diseases Transmitted: {', '.join(diseases)}")
    else:
        print("❓ No disease information available for this species.")
        

# ---------- Example Usage ----------
if __name__ == "__main__":
    audio_path = input("🔍 Enter path to mosquito wingbeat .wav file: ").strip()
    predict_species(audio_path)
