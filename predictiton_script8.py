import numpy as np
import librosa
import pickle
import threading
from keras.models import load_model, Model
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pyttsx3

# ---------- Constants ----------
SR = 8000
DURATION = 5  # seconds
N_FFT = 256
HOP_LEN = N_FFT // 6
TARGET_NAMES = ['Aedes aegypti', 'Aedes albopictus', 'Anopheles gambiae',
                'Anopheles arabiensis', 'Culex pipiens', 'Culex quinquefasciatus']
MODEL_INPUT_SHAPE = (129, 120, 3)

# ---------- Disease Mapping ----------
MOSQUITO_DISEASES = {
    'Aedes aegypti': ['Dengue fever', 'Zika virus', 'Chikungunya', 'Yellow fever'],
    'Aedes albopictus': ['Dengue fever', 'Chikungunya', 'Zika virus'],
    'Anopheles gambiae': ['Malaria', 'Lymphatic filariasis'],
    'Anopheles arabiensis': ['Malaria', 'Lymphatic filariasis'],
    'Culex pipiens': ['West Nile virus', 'St. Louis encephalitis', 'Lymphatic filariasis'],
    'Culex quinquefasciatus': ['Lymphatic filariasis', 'West Nile virus', 'Japanese encephalitis']
}

# ---------- Text-to-Speech Engine ----------
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

def speak_result(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def speak_welcome():
    speak_result("Hi, I am mosquito species analyzer. Please enter the path to the audio file for the prediction of mosquito species.")

# ---------- Custom Normalization ----------
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
def audio_to_spectrogram(file_path, target_width=120):
    data, _ = librosa.load(file_path, sr=SR)
    spectrogram = librosa.stft(data, n_fft=N_FFT, hop_length=HOP_LEN)
    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram))
    spectrogram = np.flipud(spectrogram)

    if spectrogram.shape[1] < target_width:
        pad_width = target_width - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        start = (spectrogram.shape[1] - target_width) // 2
        spectrogram = spectrogram[:, start:start + target_width]

    spectrogram = np.expand_dims(spectrogram, axis=-1)
    spectrogram = np.repeat(spectrogram, 3, axis=-1)

    return np.expand_dims(spectrogram.astype(np.float32), axis=0)

# ---------- GUI ----------
root = tk.Tk()
root.title("Mosquito Species Identifier")
root.geometry("1500x650")
root.resizable(False, False)

# Background Image
try:
    img = Image.open("bag.jpg")
    img = img.resize((1500, 650))
    bgg = ImageTk.PhotoImage(img)
    lbl = tk.Label(root, image=bgg)
    lbl.place(x=0, y=0)
except Exception as e:
    print(f"Background image load failed: {e}")

# GUI Title
tk.Label(root, text="🎙️ Mosquito Wingbeat Sound Analyzer", font=("Helvetica", 16, "bold")).pack(pady=15)

# Frame for Entry and Label
entry_frame = tk.Frame(root)
entry_frame.pack(pady=10)

tk.Label(entry_frame, text="Enter path to .wav file here:", font=("Helvetica", 12)).pack(side=tk.LEFT, padx=5)
file_entry = tk.Entry(entry_frame, width=60, font=("Helvetica", 12))
file_entry.pack(side=tk.LEFT)

# ---------- Prediction Pipeline ----------
def predict_species(audio_path):
    try:
        # Append .wav if not present
        audio_path = audio_path.strip()
        if not audio_path.lower().endswith('.wav'):
            audio_path += '.wav'

        spectrogram = audio_to_spectrogram(audio_path, target_width=MODEL_INPUT_SHAPE[1])
        features = feature_model.predict(spectrogram)
        features = features.flatten().reshape(1, -1)
        prediction = xgb_model.predict(features)[0]
        predicted_label = TARGET_NAMES[prediction]

        result = f"🦟 Predicted Mosquito Species: {predicted_label}"
        diseases = MOSQUITO_DISEASES.get(predicted_label, ["Unknown"])
        disease_str = ', '.join(diseases)
        result += f"\n🦠 Diseases Transmitted: {disease_str}"

        # Speak result
        threading.Thread(
            target=speak_result,
            args=(f"The mosquito species is {predicted_label}. It transmits: {disease_str}.",),
            daemon=True
        ).start()

        # Show messagebox and clear entry after clicking OK
        messagebox.showinfo("Prediction Result", result)
        file_entry.delete(0, tk.END)  # Clear the input field
        threading.Thread(target=speak_welcome, daemon=True).start()

    except Exception as e:
        threading.Thread(
            target=speak_result,
            args=("Prediction failed.",),
            daemon=True
        ).start()
        messagebox.showerror("Error", f"Prediction failed: {str(e)}")
        file_entry.delete(0, tk.END)  # Clear input even on error
        threading.Thread(target=speak_welcome, daemon=True).start()

# Analyze Button
tk.Button(root, text="Analyse", width=20,
          command=lambda: predict_species(file_entry.get()),
          bg="blue", fg="white", font=("Helvetica", 12, "bold")).pack(pady=10)

# ---------- Initial Welcome ----------
threading.Thread(target=speak_welcome, daemon=True).start()

root.mainloop()
