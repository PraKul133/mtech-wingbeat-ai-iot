# mtech-wingbeat-ai-iot
# Overview : 
This project presents an IoT-based intelligent system for classifying mosquito species based on their wingbeat acoustic signatures. Leveraging Deep Learning (Xception) for feature extraction and XGBoost for classification, this system aims to support vector-based disease monitoring and control efforts with real-time deployment on edge devices like ESP32 DevKit v1 WROOM.

# Objectives
Capture wingbeat sounds using a MEMS microphone (INMP441) connected to ESP32.

Classify mosquito species using a trained hybrid Xception + XGBoost model.

Display detection results and system status on an OLED display.

Enable wireless data communication between ESP32 and a local prediction system.

Also Enable local storage of raw wingbeat audio and sensor data using an SD card module to support offline analysis, backup, and future model retraining in areas with limited or no internet connectivity.

# System Design (IoT + ML)
Hardware:
ESP32 DevKit v1 WROOM

Captures audio via I2S interface from INMP441 MEMS microphone.

Sends data to local system via Wi-Fi using TCP socket.

Controls OLED display for real-time alerts and status.

INMP441 Microphone: High-quality, low-noise MEMS microphone.

0.96" OLED Display: Displays detected species and system feedback.

DHT22 Sensor (optional): Measures temperature and humidity to support ecological research.

Software:
Arduino C++ code for data acquisition, display control, and Wi-Fi transmission.

Python backend for receiving audio data and performing inference using the Xception + XGBoost model (.h5 and .pkl files respectively).

Outputs: Species label, confidence score, and performance metrics (accuracy, confusion matrix).
