# Heart Disease Detection from ECG Images

A deep learning project for classifying ECG images as Normal or Abnormal using transfer learning (EfficientNetB0).

## Overview

- Built with TensorFlow, Keras, and Streamlit  
- Achieves 81% accuracy on both training and test sets  
- Real-time ECG image prediction via web interface  
- Includes confidence display and adjustable threshold

## Project Structure

- `scripts/`: Training, evaluation, and app scripts  
- `models/`: Trained `.keras` model file  
- `train_binary/`, `valid_binary/`, `test_binary/`: Image datasets

## Run Streamlit App

- streamlit run scripts/streamlit_app.py
