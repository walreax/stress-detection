#!/usr/bin/env python3

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class ImprovedStressDetectionModel:
    
    def __init__(self, sequence_length=240, n_features=8):  # Changed to 8 features
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = RobustScaler()
        
    def configure_gpu(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                return True
            except RuntimeError:
                return False
        return False
    
    def extract_features_improved(self, data_path):
        subjects = []
        signals_list = []
        labels_list = []
        
        for subject_dir in os.listdir(data_path):
            if not subject_dir.startswith('S'):
                continue
                
            subject_path = os.path.join(data_path, subject_dir)
            pkl_file = os.path.join(subject_path, f"{subject_dir}.pkl")
            
            if not os.path.exists(pkl_file):
                continue
                
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                
                chest_data = data['signal']['chest']
                labels = data['label']
                
                sampling_rate = 700
                window_size = int(6 * sampling_rate)
                overlap = int(0.75 * window_size)
                
                acc_x = chest_data['ACC'][:, 0]
                acc_y = chest_data['ACC'][:, 1] 
                acc_z = chest_data['ACC'][:, 2]
                ecg = chest_data['ECG'].flatten()
                eda = chest_data['EDA'].flatten()
                emg = chest_data['EMG'].flatten()
                resp = chest_data['Resp'].flatten()
                temp = chest_data['Temp'].flatten()
                
                min_length = min(len(acc_x), len(acc_y), len(acc_z), len(ecg), 
                               len(eda), len(emg), len(resp), len(temp), len(labels))
                
                if min_length < window_size:
                    continue
                
                signals = np.column_stack([
                    acc_x[:min_length], acc_y[:min_length], acc_z[:min_length],
                    ecg[:min_length], eda[:min_length], emg[:min_length],
                    resp[:min_length], temp[:min_length]
                ])
                
                subject_signals = []
                subject_labels = []
                
                for start in range(0, min_length - window_size + 1, overlap):
                    end = start + window_size
                    window_signals = signals[start:end]
                    window_labels = labels[start:end]
                    
                    stress_ratio = np.sum(np.isin(window_labels, [2, 3])) / len(window_labels)
                    non_stress_ratio = np.sum(window_labels == 1) / len(window_labels)
                    
                    if stress_ratio > 0.2:
                        features = self.compute_advanced_features(window_signals)
                        if features is not None:
                            subject_signals.append(features)
                            subject_labels.append(1)
                    elif non_stress_ratio > 0.6:
                        features = self.compute_advanced_features(window_signals)
                        if features is not None:
                            subject_signals.append(features)
                            subject_labels.append(0)
                
                if subject_signals:
                    subjects.append(subject_dir)
                    signals_list.extend(subject_signals)
                    labels_list.extend(subject_labels)
                    print(f"  {subject_dir}: {len(subject_signals)} windows")
                    
            except Exception as e:
                print(f"Error processing {subject_dir}: {e}")
                continue
                
        return np.array(signals_list), np.array(labels_list)
    
    def compute_advanced_features(self, window_signals):
        try:
            downsampled = window_signals[::int(len(window_signals)/self.sequence_length)]
            
            if len(downsampled) < self.sequence_length:
                padding_needed = self.sequence_length - len(downsampled)
                last_samples = downsampled[-min(10, len(downsampled)):]
                padding = np.tile(last_samples, (padding_needed // len(last_samples) + 1, 1))[:padding_needed]
                downsampled = np.vstack([downsampled, padding])
            elif len(downsampled) > self.sequence_length:
                downsampled = downsampled[:self.sequence_length]
                
            return downsampled
            
        except Exception:
            return None
    
    def generate_advanced_synthetic_data(self, X, y, target_samples_per_class=50):
        """Advanced synthetic data generation with realistic stress patterns"""
        unique_classes, counts = np.unique(y, return_counts=True)
        
        synthetic_X = []
        synthetic_y = []
        
        minority_class = 1 if len(unique_classes) > 1 else 1
        majority_samples = X[y == 0] if len(unique_classes) > 1 else X
        
        n_stress_samples = target_samples_per_class
        
        print(f"Generating {n_stress_samples} high-quality synthetic stress samples...")
        
        for i in range(n_stress_samples):
            base_idx = np.random.randint(0, len(majority_samples))
            base_sample = majority_samples[base_idx].copy()
            
            stress_sample = base_sample.copy()
            
            time_points = np.arange(len(stress_sample))
            
            # EDA increase with temporal patterns (index 4)
            eda_increase = 0.3 + 0.2 * np.sin(2 * np.pi * time_points / len(time_points))
            eda_noise = np.random.normal(0, 0.1, len(time_points))
            stress_sample[:, 4] += eda_increase + eda_noise
            
            # Heart rate variability patterns (ECG - index 3)
            hr_pattern = 0.2 * np.sin(4 * np.pi * time_points / len(time_points))
            hr_noise = np.random.normal(0, 0.15, len(time_points))
            stress_sample[:, 3] += hr_pattern + hr_noise
            
            # Movement increase (accelerometer indices 0,1,2)
            for acc_idx in [0, 1, 2]:
                movement_pattern = 0.1 * np.random.choice([-1, 1]) * np.sin(
                    np.random.uniform(1, 6) * np.pi * time_points / len(time_points)
                )
                movement_noise = np.random.normal(0, 0.05, len(time_points))
                stress_sample[:, acc_idx] += movement_pattern + movement_noise
            
            # Temperature increase (index 7)
            temp_increase = np.random.normal(0.15, 0.05)
            temp_drift = temp_increase * time_points / len(time_points)
            stress_sample[:, 7] += temp_drift
            
            # EMG tension increase (index 5)
            emg_tension = 0.1 + 0.1 * np.random.random()
            emg_bursts = np.random.random(len(time_points)) > 0.95
            stress_sample[:, 5] += emg_tension + emg_bursts * 0.2
            
            # Respiration changes (index 6)
            resp_pattern = 0.1 * np.sin(8 * np.pi * time_points / len(time_points))
            stress_sample[:, 6] += resp_pattern
            
            synthetic_X.append(stress_sample)
            synthetic_y.append(1)
        
        return np.array(synthetic_X), np.array(synthetic_y)
    
    def build_improved_model(self):
        """Enhanced model with better architecture for stress detection"""
        inputs = keras.Input(shape=(self.sequence_length, self.n_features), name='signal_input')
        
        # Multi-scale feature extraction
        conv_features = []
        for kernel_size in [3, 5, 7]:
            conv = layers.Conv1D(64, kernel_size, padding='same', activation='relu')(inputs)
            conv = layers.BatchNormalization()(conv)
            conv_features.append(conv)
        
        # Concatenate multi-scale features
        x = layers.Concatenate()(conv_features)
        x = layers.Dropout(0.3)(x)
        
        # Deeper convolutional layers
        x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Bidirectional LSTM
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
        x = layers.Bidirectional(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
        
        # Enhanced classification head
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(2, activation='softmax', name='stress_output')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0003),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def train_improved(self, X, y, validation_split=0.2, epochs=100, batch_size=32):  # Changed batch size
        """Improved training with better data handling"""
        
        print(f"Original class distribution: {np.bincount(y)}")
        
        # Generate high-quality synthetic data
        synthetic_X, synthetic_y = self.generate_advanced_synthetic_data(X, y, target_samples_per_class=len(X))
        X_combined = np.vstack([X, synthetic_X])
        y_combined = np.hstack([y, synthetic_y])
        
        print(f"After synthetic data: {np.bincount(y_combined)}")
        
        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_combined), y=y_combined)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"Class weights: {class_weight_dict}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)
        ]
        
        # Train with class weights and more epochs
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Comprehensive evaluation
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        f1_macro = f1_score(y_test, y_pred_classes, average='macro')
        f1_weighted = f1_score(y_test, y_pred_classes, average='weighted')
        f1_per_class = f1_score(y_test, y_pred_classes, average=None)
        
        results = {
            'history': history,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_loss': test_loss,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
            'classification_report': classification_report(y_test, y_pred_classes, target_names=['Non-Stress', 'Stress']),
            'confusion_matrix': confusion_matrix(y_test, y_pred_classes),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        return results
    
    def save_model(self, filepath):
        if self.model:
            self.model.save(filepath)
    
    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)

def main():
    print("🚀 SIGNIFICANTLY IMPROVED STRESS DETECTION MODEL")
    print("=" * 70)
    
    detector = ImprovedStressDetectionModel()
    
    gpu_available = detector.configure_gpu()
    print(f"GPU Available: {gpu_available}")
    
    data_path = 'WESAD'
    print("\nExtracting features with improved method...")
    X, y = detector.extract_features_improved(data_path)
    
    if len(X) == 0:
        print("No data extracted!")
        return
    
    print(f"\nExtracted {len(X)} samples")
    print(f"Class distribution: {np.bincount(y)}")
    
    print("\nBuilding significantly improved model...")
    detector.build_improved_model()
    
    print(f"\nModel has {detector.model.count_params():,} parameters")
    
    print("\nTraining improved model...")
    results = detector.train_improved(X, y)
    
    print("\n" + "=" * 70)
    print("🎯 SIGNIFICANTLY IMPROVED MODEL RESULTS")
    print("=" * 70)
    print(f"✅ Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
    print(f"📊 F1 Macro: {results['f1_macro']:.4f} ({results['f1_macro']*100:.2f}%)")
    print(f"📊 F1 Weighted: {results['f1_weighted']:.4f} ({results['f1_weighted']*100:.2f}%)")
    print(f"🔵 F1 Non-Stress: {results['f1_per_class'][0]:.4f} ({results['f1_per_class'][0]*100:.2f}%)")
    print(f"🔴 F1 Stress: {results['f1_per_class'][1]:.4f} ({results['f1_per_class'][1]*100:.2f}%)")
    
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print(results['classification_report'])
    
    print(f"\n🔍 CONFUSION MATRIX:")
    cm = results['confusion_matrix']
    print(f"                 Predicted")
    print(f"Actual    Non-Stress  Stress")
    print(f"Non-Stress    {cm[0,0]:3d}      {cm[0,1]:3d}")
    print(f"Stress        {cm[1,0]:3d}      {cm[1,1]:3d}")
    
    detector.save_model('significantly_improved_stress_model.h5')
    print("\n✅ Significantly improved model saved!")
    
    print("\n🎉 MODEL IMPROVEMENT COMPLETE!")
    return results

if __name__ == "__main__":
    results = main()
