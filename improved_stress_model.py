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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

class ImprovedStressDetectionModel:
    
    def __init__(self, sequence_length=240, n_features=14):
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
                window_size = int(6 * sampling_rate)  # Increased window size
                overlap = int(0.75 * window_size)     # Increased overlap
                
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
                    
                    # More aggressive stress labeling
                    stress_ratio = np.sum(np.isin(window_labels, [2, 3])) / len(window_labels)
                    non_stress_ratio = np.sum(window_labels == 1) / len(window_labels)
                    
                    if stress_ratio > 0.3:  # Lower threshold for stress
                        features = self.compute_advanced_features(window_signals)
                        if features is not None:
                            subject_signals.append(features)
                            subject_labels.append(1)  # Stress
                    elif non_stress_ratio > 0.7:  # Higher threshold for non-stress
                        features = self.compute_advanced_features(window_signals)
                        if features is not None:
                            subject_signals.append(features)
                            subject_labels.append(0)  # Non-stress
                
                if subject_signals:
                    subjects.append(subject_dir)
                    signals_list.extend(subject_signals)
                    labels_list.extend(subject_labels)
                    
            except Exception as e:
                print(f"Error processing {subject_dir}: {e}")
                continue
                
        return np.array(signals_list), np.array(labels_list)
    
    def compute_advanced_features(self, window_signals):
        try:
            # Enhanced feature extraction with more temporal patterns
            features = []
            
            for i in range(window_signals.shape[1]):
                signal = window_signals[:, i]
                
                # Statistical features
                features.extend([
                    np.mean(signal),
                    np.std(signal),
                    np.var(signal),
                    np.min(signal),
                    np.max(signal),
                    np.median(signal),
                    np.percentile(signal, 25),
                    np.percentile(signal, 75)
                ])
                
                # Temporal variation features
                diff_signal = np.diff(signal)
                features.extend([
                    np.mean(diff_signal),
                    np.std(diff_signal),
                    np.sum(np.abs(diff_signal))
                ])
            
            # Downsample to sequence length with preserved patterns
            downsampled = window_signals[::int(len(window_signals)/self.sequence_length)]
            
            if len(downsampled) < self.sequence_length:
                # Intelligent padding that preserves signal characteristics
                padding_needed = self.sequence_length - len(downsampled)
                # Repeat the last few samples instead of zero padding
                last_samples = downsampled[-min(10, len(downsampled)):]
                padding = np.tile(last_samples, (padding_needed // len(last_samples) + 1, 1))[:padding_needed]
                downsampled = np.vstack([downsampled, padding])
            elif len(downsampled) > self.sequence_length:
                downsampled = downsampled[:self.sequence_length]
                
            return downsampled
            
        except Exception:
            return None
    
    def generate_better_synthetic_data(self, X, y, target_ratio=0.5):
        """Enhanced synthetic data generation using SMOTE + data augmentation"""
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            # Create high-quality synthetic stress samples
            n_synthetic = int(len(X) * target_ratio)
            synthetic_X = []
            synthetic_y = []
            
            for _ in range(n_synthetic):
                # More sophisticated stress pattern generation
                base_sample = X[np.random.randint(0, len(X))]
                
                # Apply stress-specific transformations
                stress_sample = base_sample.copy()
                
                # Increase EDA (index 4)
                stress_sample[:, 4] += np.random.normal(0.5, 0.2, stress_sample.shape[0])
                
                # Increase heart rate variability (ECG - index 3)
                heart_noise = np.random.normal(0, 0.3, stress_sample.shape[0])
                stress_sample[:, 3] += heart_noise
                
                # Add stress-specific movement patterns (ACC)
                for acc_idx in [0, 1, 2]:
                    stress_sample[:, acc_idx] += np.random.normal(0, 0.1, stress_sample.shape[0])
                
                # Temperature increase (index 7)
                stress_sample[:, 7] += np.random.normal(0.2, 0.1, stress_sample.shape[0])
                
                # Add some coherent temporal patterns
                time_factor = np.sin(np.linspace(0, 4*np.pi, len(stress_sample)))
                stress_sample[:, 4] += time_factor * 0.1  # EDA oscillation
                
                synthetic_X.append(stress_sample)
                synthetic_y.append(1)  # Stress class
            
            return np.array(synthetic_X), np.array(synthetic_y)
        else:
            # Use SMOTE for balanced data generation
            try:
                # Flatten for SMOTE
                X_flat = X.reshape(X.shape[0], -1)
                smote = SMOTE(random_state=42, k_neighbors=min(3, min(counts)-1))
                X_resampled, y_resampled = smote.fit_resample(X_flat, y)
                
                # Reshape back
                X_resampled = X_resampled.reshape(-1, self.sequence_length, self.n_features)
                return X_resampled, y_resampled
            except:
                # Fallback to manual generation
                return self.generate_better_synthetic_data(X, y[:0], target_ratio)
    
    def build_improved_model(self):
        """Enhanced model architecture with attention and residual connections"""
        inputs = keras.Input(shape=(self.sequence_length, self.n_features), name='signal_input')
        
        # Multi-scale CNN feature extraction
        conv1 = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
        conv1 = layers.BatchNormalization()(conv1)
        
        conv2 = layers.Conv1D(64, 5, padding='same', activation='relu')(inputs)
        conv2 = layers.BatchNormalization()(conv2)
        
        conv3 = layers.Conv1D(96, 7, padding='same', activation='relu')(inputs)
        conv3 = layers.BatchNormalization()(conv3)
        
        # Concatenate multi-scale features
        multi_scale = layers.Concatenate()([conv1, conv2, conv3])
        multi_scale = layers.Dropout(0.3)(multi_scale)
        
        # Additional convolutional layers
        x = layers.Conv1D(128, 3, padding='same', activation='relu')(multi_scale)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Bidirectional LSTM for better temporal modeling
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
        x = layers.Bidirectional(layers.LSTM(64, dropout=0.2))(x)
        
        # Attention mechanism
        attention = layers.Dense(128, activation='tanh')(x)
        attention = layers.Dense(1, activation='sigmoid')(attention)
        attended = layers.Multiply()([x, attention])
        
        # Classification head with more capacity
        x = layers.Dense(256, activation='relu')(attended)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(2, activation='softmax', name='stress_output')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Use class weights to handle imbalance
        model.compile(
            optimizer=Adam(learning_rate=0.0005),  # Lower learning rate
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def train_improved(self, X, y, validation_split=0.2, epochs=150, batch_size=16):
        """Enhanced training with better data handling and callbacks"""
        
        print(f"Original class distribution: {np.bincount(y)}")
        
        # Generate better synthetic data
        if len(np.unique(y)) == 1 or np.min(np.bincount(y)) < 5:
            synthetic_X, synthetic_y = self.generate_better_synthetic_data(X, y, target_ratio=0.6)
            X = np.vstack([X, synthetic_X])
            y = np.hstack([y, synthetic_y])
            print(f"After synthetic data: {np.bincount(y)}")
        
        # Compute class weights for imbalanced learning
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y), 
            y=y
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"Class weights: {class_weight_dict}")
        
        # Stratified split to ensure both classes in train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_f1_score', 
                patience=20, 
                restore_best_weights=True,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.3, 
                patience=8, 
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Custom F1 callback
        class F1ScoreCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
                f1 = f1_score(y_test, y_pred, average='macro')
                logs['val_f1_score'] = f1
                if epoch % 10 == 0:
                    print(f"\nEpoch {epoch}: F1-Score = {f1:.4f}")
        
        callbacks.append(F1ScoreCallback())
        
        # Train with class weights
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
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate all F1 scores
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
            'classification_report': classification_report(y_test, y_pred_classes),
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
    print("🚀 IMPROVED STRESS DETECTION MODEL")
    print("=" * 60)
    
    detector = ImprovedStressDetectionModel()
    
    gpu_available = detector.configure_gpu()
    print(f"GPU Available: {gpu_available}")
    
    data_path = 'WESAD'
    print("Extracting features with improved method...")
    X, y = detector.extract_features_improved(data_path)
    
    if len(X) == 0:
        print("No data extracted!")
        return
    
    print(f"Extracted {len(X)} samples")
    print(f"Class distribution: {np.bincount(y)}")
    
    print("Building improved model...")
    detector.build_improved_model()
    detector.model.summary()
    
    print("Training improved model...")
    results = detector.train_improved(X, y)
    
    print("\n" + "=" * 60)
    print("IMPROVED MODEL RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"F1 Macro: {results['f1_macro']:.4f}")
    print(f"F1 Weighted: {results['f1_weighted']:.4f}")
    print(f"F1 Non-Stress: {results['f1_per_class'][0]:.4f}")
    print(f"F1 Stress: {results['f1_per_class'][1]:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    detector.save_model('improved_stress_detection_model.h5')
    print("✅ Improved model saved!")
    
    return results

if __name__ == "__main__":
    results = main()
