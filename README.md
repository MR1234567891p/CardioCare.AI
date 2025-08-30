# ECG Arrhythmia Detection with Deep Learning

A comprehensive deep learning system for automatic detection and classification of cardiac arrhythmias from ECG signals, featuring advanced preprocessing techniques and a CNN-LSTM hybrid model.

## 🏥 Project Overview

This project implements an end-to-end ECG arrhythmia detection system that can classify ECG signals into 7 different categories:
- **Normal** - Normal sinus rhythm
- **Supraventricular** - Supraventricular arrhythmias (PAC, atrial fibrillation, etc.)
- **Ventricular** - Ventricular arrhythmias (PVC, ventricular tachycardia, etc.)
- **Fusion** - Fusion beats
- **Unknown/Noise** - Signal quality issues or unclassifiable beats
- **AFib** - Atrial fibrillation
- **ST changes** - ST segment changes (potential ischemia)

## ✨ Key Features

### 🔧 Advanced Preprocessing Pipeline
- **Baseline Wander Removal** - Polynomial fitting, median filtering, and wavelet-based methods
- **Powerline Noise Filtering** - Adaptive notch filters for 50/60 Hz interference
- **High-Frequency Noise Reduction** - Low-pass filtering with configurable cutoff frequencies
- **Motion Artifact Detection** - Adaptive thresholding and interpolation
- **Signal Normalization** - Multiple normalization methods (Z-score, min-max, robust)

### 🧠 Deep Learning Model
- **CNN-LSTM Architecture** - Combines convolutional layers for feature extraction with LSTM layers for temporal pattern recognition
- **Multi-class Classification** - Handles 7 different arrhythmia types
- **Transfer Learning Ready** - Pre-trained models available for immediate use
- **Real-time Prediction** - Optimized for fast inference on new ECG data

### 📊 Comprehensive Analysis
- **Signal Quality Assessment** - SNR improvement calculations
- **Visualization Tools** - Step-by-step preprocessing visualization
- **Clinical Recommendations** - Severity-based medical recommendations
- **Confidence Scoring** - Probability-based predictions with uncertainty quantification

## 📁 Project Structure

```
├── ecg_preprocessing.py              # Core preprocessing module
├── ecg_model_with_preprocessing.py   # Main training and model code
├── model test.py                     # Prediction and testing utilities
├── best_ecg_model_updated.h5         # Pre-trained model weights
├── best_ecg_model_updated_info.json  # Model metadata
├── MIT-BIH Arrhythmia Database/      # MIT-BIH database files
├── ecg_prediction_result.png         # Sample prediction visualization
├── ecg_preprocessing_steps.png       # Sample preprocessing visualization
└── README.md                         # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- NumPy, SciPy, Matplotlib
- WFDB (for MIT-BIH database reading)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ecg-arrhythmia-detection
   ```

2. **Install dependencies**
   ```bash
   pip install tensorflow numpy scipy matplotlib seaborn wfdb scikit-learn pywt
   ```

3. **Download the MIT-BIH Database** (if not included)
   ```bash
   # The database should be in the MIT-BIH Arrhythmia Database/ directory
   # If not present, download from PhysioNet
   ```

## 📖 Usage

### 1. Preprocessing ECG Signals

```python
from ecg_preprocessing import ECGPreprocessor

# Initialize preprocessor
preprocessor = ECGPreprocessor(sampling_rate=250)

# Load your ECG data (2500 samples for 10 seconds at 250 Hz)
ecg_data = your_ecg_signal

# Apply complete preprocessing pipeline
processed_ecg = preprocessor.complete_preprocessing(ecg_data, visualize=True)
```

### 2. Training a New Model

```python
from ecg_model_with_preprocessing import train_model_with_preprocessing

# Train model with preprocessing
model, results = train_model_with_preprocessing()

# Model will be saved as 'ecg_arrhythmia_model_with_preprocessing.h5'
```

### 3. Making Predictions

```python
from model_test import ECGPredictor

# Initialize predictor
predictor = ECGPredictor(model_path='best_ecg_model_updated.h5')

# Make prediction on ECG data
result = predictor.predict(ecg_data)

print(f"Predicted: {result['class']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Recommendation: {result['recommendation']['action']}")
```

### 4. Demo with Visualization

```python
from model_test import demo

# Run interactive demo
demo()
```

## 🔬 Technical Details

### Model Architecture
- **Input**: 2500 samples (10 seconds at 250 Hz)
- **Convolutional Layers**: 3 layers with increasing filters (64→128→256)
- **LSTM Layers**: 2 layers for temporal pattern recognition
- **Dense Layers**: 2 hidden layers with dropout for classification
- **Output**: 7-class softmax probabilities

### Preprocessing Pipeline
1. **Baseline Removal** - Polynomial fitting (3rd order)
2. **Powerline Filtering** - Butterworth notch filter (50 Hz)
3. **High-Frequency Filtering** - Low-pass filter (40 Hz cutoff)
4. **Motion Artifact Removal** - Adaptive thresholding
5. **Normalization** - Z-score standardization

### Performance Metrics
- **Accuracy**: >90% on test set
- **SNR Improvement**: 15-25 dB typical
- **Inference Time**: <100ms per 10-second window

## 📊 Results and Visualizations

The system generates comprehensive visualizations:

- **Preprocessing Steps**: Shows signal transformation at each stage
- **Prediction Results**: Displays ECG signal, class probabilities, and clinical recommendations
- **Model Performance**: Confusion matrices and classification reports

## 🏥 Clinical Applications

This system is designed for:
- **Screening**: Initial arrhythmia detection in primary care
- **Monitoring**: Continuous ECG monitoring in hospitals
- **Research**: Large-scale ECG analysis studies
- **Education**: Teaching cardiac rhythm recognition

## ⚠️ Important Notes

### Medical Disclaimer
This software is for **research and educational purposes only**. It should not be used for clinical decision-making without proper validation and regulatory approval.

### Data Requirements
- **Sampling Rate**: 250 Hz recommended (360 Hz supported with resampling)
- **Signal Length**: 10-second windows (2500 samples at 250 Hz)
- **Lead**: Lead II preferred, but other leads supported
- **Quality**: Clean signals work best, but preprocessing handles noise

### Limitations
- Not FDA-approved for clinical use
- Requires validation on diverse populations
- Performance may vary with signal quality
- Should be used as adjunct to clinical judgment

## Acknowledgments

- **MIT-BIH Arrhythmia Database** - PhysioNet for providing the benchmark dataset
- **TensorFlow/Keras** - Deep learning framework
- **SciPy/NumPy** - Scientific computing libraries
- **WFDB** - Waveform database tools


**Made with ❤️ for advancing cardiac care through AI**
