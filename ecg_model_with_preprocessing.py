#main ecg detection model
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import wfdb
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from ecg_preprocessing import ECGPreprocessor

class ECGDatasetWithPreprocessing:
    def __init__(self, data_dir='MIT-BIH Arrhythmia Database/mit-bih-arrhythmia-database-1.0.0'):
        """Initialize dataset with preprocessing capabilities"""
        self.data_dir = data_dir
        self.records = self._get_records()
        self.preprocessor = ECGPreprocessor(sampling_rate=250)
        
        # Updated arrhythmia types mapping to 7 categories
        self.arrhythmia_types = {
            'N': 0,    # Normal
            'S': 1,    # Supraventricular (PAC etc.)
            'A': 1,    # Atrial premature beat (Supraventricular)
            'J': 1,    # Nodal premature beat (Supraventricular)
            'a': 1,    # Aberrated atrial premature beat (Supraventricular)
            'n': 1,    # Supraventricular escape beat
            'e': 1,    # Atrial escape beat (Supraventricular)
            'j': 1,    # Nodal escape beat (Supraventricular)
            'V': 2,    # Ventricular (PVC etc.)
            'r': 2,    # R-on-T premature ventricular contraction
            'E': 2,    # Ventricular escape beat
            'F': 3,    # Fusion
            'f': 3,    # Fusion of paced and normal beat
            'Q': 4,    # Unknown/Noise
            '?': 4,    # Signal quality change
            'AF': 5,   # AFib (if present in annotations)
            'ST': 6,   # ST changes
            'L': 0,    # Left bundle branch block (treated as normal for now)
            'R': 0,    # Right bundle branch block (treated as normal for now)
            'B': 0,    # Bundle branch block (treated as normal for now)
            '/': 4,    # Paced beat (treated as noise/unknown)
        }
        
        print(f"✅ Dataset initialized with {len(self.records)} records")
        print("✅ Preprocessing pipeline ready")
    
    def _get_records(self):
        """Get list of available records"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Database directory not found: {self.data_dir}")
        
        records = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.hea'):
                records.append(file.replace('.hea', ''))
        
        return sorted(records)
    
    def load_record(self, record_name):
        """Load ECG record and annotations"""
        try:
            record_path = os.path.join(self.data_dir, record_name)
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            return record, annotation
        except Exception as e:
            print(f"Error loading record {record_name}: {e}")
            return None, None
    
    def get_lead_ii_data(self, record):
        """Extract Lead II data (preferred) or first available lead"""
        if hasattr(record, 'p_signal'):
            # Try to find Lead II
            if record.n_sig >= 2:  # At least 2 leads available
                return record.p_signal[:, 1]  # Lead II is usually second
            else:
                return record.p_signal[:, 0]  # Use first lead if only one
        return None
    
    def resample_signal(self, signal, original_rate=360, target_rate=250):
        """Resample signal to target sampling rate"""
        if original_rate == target_rate:
            return signal
        
        # Calculate resampling factor
        resample_factor = target_rate / original_rate
        new_length = int(len(signal) * resample_factor)
        
        # Resample using scipy
        from scipy import signal as scipy_signal
        resampled = scipy_signal.resample(signal, new_length)
        
        return resampled
    
    def extract_windows(self, ecg_data, annotation, window_size=2500, overlap_size=1250):
        """Extract windows with preprocessing"""
        if ecg_data is None or annotation is None:
            print("❌ ECG data or annotation is None")
            return [], []
        
        print(f"📊 Original ECG data length: {len(ecg_data)}")
        print(f"📊 Number of annotations: {len(annotation.sample)}")
        
        # Resample to 250 Hz if needed
        if len(ecg_data) > 0:
            # Estimate original sampling rate (MIT-BIH is typically 360 Hz)
            estimated_rate = 360
            ecg_data = self.resample_signal(ecg_data, estimated_rate, 250)
            print(f"📊 Resampled ECG data length: {len(ecg_data)}")
        
        windows = []
        labels = []
        stride = window_size - overlap_size
        
        # Get annotation positions
        ann_positions = annotation.sample
        ann_symbols = annotation.symbol
        
        # Convert annotation positions to new sampling rate
        if len(ann_positions) > 0:
            # Estimate original rate from first few positions
            if len(ann_positions) > 1:
                time_diff = (ann_positions[1] - ann_positions[0]) / 360  # Assuming 360 Hz
                original_rate = 1 / time_diff
            else:
                original_rate = 360
            
            # Resample annotation positions
            resample_factor = 250 / original_rate
            ann_positions = (ann_positions * resample_factor).astype(int)
            print(f"📊 Resampled annotation positions: {len(ann_positions)}")
        
        # Check if we have enough data for at least one window
        if len(ecg_data) < window_size:
            print(f"❌ ECG data too short: {len(ecg_data)} < {window_size}")
            return [], []
        
        # Extract windows
        num_windows = (len(ecg_data) - window_size) // stride + 1
        print(f"📊 Will extract {num_windows} windows")
        
        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            window = ecg_data[start:end]
            
            # Apply preprocessing to the window
            try:
                processed_window = self.preprocessor.complete_preprocessing(window, visualize=False)
                
                # Determine label for this window
                window_label = self._get_window_label(start, end, ann_positions, ann_symbols)
                
                if window_label is not None:
                    windows.append(processed_window)
                    labels.append(window_label)
                    
                    if len(windows) % 10 == 0:
                        print(f"✅ Processed {len(windows)} windows...")
                        
            except Exception as e:
                print(f"❌ Error processing window {i}: {e}")
                continue
        
        print(f"✅ Successfully extracted {len(windows)} windows with {len(set(labels))} unique labels")
        return windows, labels
    
    def _get_window_label(self, start, end, ann_positions, ann_symbols):
        """Determine the label for a window based on annotations"""
        if len(ann_positions) == 0:
            return 0  # Default to Normal if no annotations
        
        # Find annotations within this window
        window_annotations = []
        for i, pos in enumerate(ann_positions):
            if start <= pos < end:
                symbol = ann_symbols[i]
                if symbol in self.arrhythmia_types:
                    window_annotations.append(self.arrhythmia_types[symbol])
        
        if not window_annotations:
            return 0  # Normal if no relevant annotations
        
        # Return the most common arrhythmia type in the window
        from collections import Counter
        most_common = Counter(window_annotations).most_common(1)[0][0]
        return most_common

class ECGArrhythmiaModelWithPreprocessing:
    def __init__(self, input_shape=(2500, 1), num_classes=7):
        """Initialize model with preprocessing"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.category_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown/Noise', 'AFib', 'ST changes']
        self.model = None
        self.preprocessor = ECGPreprocessor(sampling_rate=250)
        
        print("✅ Model initialized with preprocessing capabilities")
    
    def build_model(self):
        """Build the CNN-LSTM model"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Convolutional layers for feature extraction
            layers.Conv1D(64, kernel_size=7, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            # LSTM layers for temporal pattern recognition
            layers.LSTM(128, return_sequences=True, dropout=0.2),
            layers.LSTM(64, dropout=0.2),
            
            # Dense layers for classification
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("✅ Model built successfully")
    
    def prepare_data(self, windows, labels):
        """Prepare data for training with preprocessing"""
        if len(windows) == 0:
            print("❌ No windows provided - checking data loading...")
            raise ValueError("No windows provided")
        
        print(f"📊 Preparing {len(windows)} windows for training...")
        
        # Convert to numpy arrays
        X = np.array(windows).reshape(-1, self.input_shape[0], 1)
        y = np.array(labels)
        
        print(f"📊 X shape: {X.shape}")
        print(f"📊 y shape: {y.shape}")
        print(f"📊 Unique labels: {np.unique(y, return_counts=True)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Data prepared: {len(X_train)} training, {len(X_test)} testing samples")
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=16):
        """Train the model"""
        print("🚀 Starting training with preprocessing...")
        
        # Add early stopping and model checkpoint
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                'best_ecg_model_with_preprocessing.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        # Get unique classes in test data
        unique_classes = np.unique(y_test)
        available_category_names = [self.category_names[i] for i in unique_classes]
        
        report = classification_report(y_test, y_pred_classes, target_names=available_category_names)
        cm = confusion_matrix(y_test, y_pred_classes)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_test,
            'available_classes': unique_classes.tolist()
        }
    
    def predict_single(self, ecg_data):
        """Make prediction on single ECG window with preprocessing"""
        # Ensure correct shape
        if len(ecg_data) != self.input_shape[0]:
            raise ValueError(f"ECG data must be {self.input_shape[0]} samples")
        
        # Apply preprocessing
        processed_data = self.preprocessor.complete_preprocessing(ecg_data, visualize=False)
        
        # Reshape for model input
        input_data = processed_data.reshape(1, self.input_shape[0], 1)
        
        # Make prediction
        prediction = self.model.predict(input_data, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return {
            'class': self.category_names[predicted_class],
            'confidence': confidence,
            'probabilities': prediction[0].tolist()
        }
    
    def save_model(self, filepath):
        """Save model and metadata"""
        self.model.save(filepath)
        
        # Save model info
        model_info = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'category_names': self.category_names
        }
        
        with open(filepath.replace('.h5', '_info.json'), 'w') as f:
            json.dump(model_info, f)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = tf.keras.models.load_model(filepath)
        
        # Load model info
        try:
            with open(filepath.replace('.h5', '_info.json'), 'r') as f:
                model_info = json.load(f)
            
            self.input_shape = tuple(model_info['input_shape'])
            self.num_classes = model_info['num_classes']
            self.category_names = model_info['category_names']
        except FileNotFoundError:
            print("⚠️ Model info file not found, using default values")
        
        print(f"✅ Model loaded from {filepath}")

def train_model_with_preprocessing():
    """Train model with preprocessing"""
    print("ECG Arrhythmia Detection with Preprocessing")
    print("=" * 60)
    
    # Initialize dataset
    dataset = ECGDatasetWithPreprocessing()
    
    # Load and process data
    all_windows = []
    all_labels = []
    
    # Process first 5 records for demonstration (reduced from 10)
    records_to_process = dataset.records[:5]
    print(f"📊 Processing {len(records_to_process)} records: {records_to_process}")
    
    for i, record_name in enumerate(records_to_process):
        print(f"\n{'='*50}")
        print(f"Processing record {i+1}/{len(records_to_process)}: {record_name}")
        print(f"{'='*50}")
        
        record, annotation = dataset.load_record(record_name)
        
        if record is not None and annotation is not None:
            # Extract Lead II data
            lead_ii_data = dataset.get_lead_ii_data(record)
            
            if lead_ii_data is not None:
                print(f"✅ Lead II data extracted, length: {len(lead_ii_data)}")
                
                # Extract windows with preprocessing
                windows, labels = dataset.extract_windows(lead_ii_data, annotation)
                
                if len(windows) > 0:
                    all_windows.extend(windows)
                    all_labels.extend(labels)
                    print(f"✅ Added {len(windows)} windows from {record_name}")
                else:
                    print(f"❌ No windows extracted from {record_name}")
            else:
                print(f"❌ Could not extract Lead II data from {record_name}")
        else:
            print(f"❌ Could not load record {record_name}")
    
    if len(all_windows) == 0:
        print("❌ No windows extracted from any records!")
        print("Please check:")
        print("1. Database files are in the correct location")
        print("2. Database files are not corrupted")
        print("3. Preprocessing pipeline is working")
        return None, None
    
    all_windows = np.array(all_windows)
    all_labels = np.array(all_labels)
    
    print(f"\n✅ Total windows extracted: {len(all_windows)}")
    print(f"✅ Window shape: {all_windows.shape}")
    print(f"✅ Unique label types: {np.unique(all_labels, return_counts=True)}")
    
    # Initialize and build model
    model = ECGArrhythmiaModelWithPreprocessing(input_shape=(2500, 1), num_classes=7)
    model.build_model()
    model.model.summary()
    
    # Prepare data
    X_train, X_test, y_train, y_test = model.prepare_data(all_windows, all_labels)
    
    # Train model
    print("\n🚀 Training model with preprocessing...")
    history = model.train(X_train, y_train, X_test, y_test, epochs=30, batch_size=16)
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    results = model.evaluate(X_test, y_test)
    
    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Save model
    model.save_model('ecg_arrhythmia_model_with_preprocessing.h5')
    
    return model, results

if __name__ == "__main__":
    train_model_with_preprocessing()
