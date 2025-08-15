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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class StressDetectionModel:
    
    def __init__(self, sequence_length=240, n_features=14):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler()
        
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
    
    def extract_features(self, data_path):
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
                window_size = int(4 * sampling_rate)
                overlap = int(0.5 * window_size)
                
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
                    
                    if len(np.unique(window_labels)) == 1:
                        label = window_labels[0]
                        if label in [1, 2, 3]:
                            features = self.compute_features(window_signals)
                            if features is not None:
                                subject_signals.append(features)
                                subject_labels.append(1 if label in [2, 3] else 0)
                
                if subject_signals:
                    subjects.append(subject_dir)
                    signals_list.extend(subject_signals)
                    labels_list.extend(subject_labels)
                    
            except Exception:
                continue
                
        return np.array(signals_list), np.array(labels_list)
    
    def compute_features(self, window_signals):
        try:
            features = []
            
            for i in range(window_signals.shape[1]):
                signal = window_signals[:, i]
                
                features.extend([
                    np.mean(signal),
                    np.std(signal)
                ])
            
            downsampled = window_signals[::int(len(window_signals)/self.sequence_length)]
            
            if len(downsampled) < self.sequence_length:
                padding = np.zeros((self.sequence_length - len(downsampled), window_signals.shape[1]))
                downsampled = np.vstack([downsampled, padding])
            elif len(downsampled) > self.sequence_length:
                downsampled = downsampled[:self.sequence_length]
                
            return downsampled
            
        except Exception:
            return None
    
    def generate_synthetic_data(self, X, y, n_synthetic=None):
        if n_synthetic is None:
            class_counts = np.bincount(y)
            max_count = np.max(class_counts)
            n_synthetic = max_count - np.min(class_counts)
        
        synthetic_X = []
        synthetic_y = []
        
        minority_class = np.argmin(np.bincount(y))
        minority_indices = np.where(y == minority_class)[0]
        
        for _ in range(n_synthetic):
            idx1, idx2 = np.random.choice(minority_indices, 2, replace=True)
            alpha = np.random.random()
            
            synthetic_sample = alpha * X[idx1] + (1 - alpha) * X[idx2]
            synthetic_sample += np.random.normal(0, 0.1, synthetic_sample.shape)
            
            synthetic_X.append(synthetic_sample)
            synthetic_y.append(minority_class)
        
        return np.array(synthetic_X), np.array(synthetic_y)
    
    def build_model(self):
        inputs = keras.Input(shape=(self.sequence_length, self.n_features), name='signal_input')
        
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.LSTM(64)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(2, activation='softmax', name='stress_output')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        if len(np.unique(y)) == 1:
            synthetic_X, synthetic_y = self.generate_synthetic_data(X, y)
            X = np.vstack([X, synthetic_X])
            y = np.hstack([y, synthetic_y])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        results = {
            'history': history,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_loss': test_loss,
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
    
    def predict(self, X):
        if self.model:
            return self.model.predict(X)
        return None

def main():
    detector = StressDetectionModel()
    
    gpu_available = detector.configure_gpu()
    
    data_path = 'WESAD'
    X, y = detector.extract_features(data_path)
    
    if len(X) == 0:
        return
    
    detector.build_model()
    
    results = detector.train(X, y)
    
    detector.save_model('stress_detection_model.h5')
    
    return results

if __name__ == "__main__":
    results = main()
