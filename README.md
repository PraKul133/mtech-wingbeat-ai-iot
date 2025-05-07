# mtech-wingbeat-ai-iot
# Overview : 
This project presents an IoT-based intelligent system for classifying mosquito species based on their wingbeat acoustic signatures. Leveraging Deep Learning (Xception) for feature extraction and XGBoost for classification, this system aims to support vector-based disease monitoring and control efforts with real-time deployment on edge devices like ESP32 DevKit v1 WROOM.

# Objectives
Capture wingbeat sounds using a MEMS microphone (INMP441) connected to ESP32.

Classify mosquito species using a trained hybrid Xception + XGBoost model.

Display detection results and system status on an OLED display.

Enable wireless data communication between ESP32 and a local prediction system.

Also Enable local storage of raw wingbeat audio and sensor data using an SD card module to support offline analysis, backup, and future model retraining in areas with limited or no internet connectivity.
