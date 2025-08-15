import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization, GlobalMaxPooling1D, Concatenate, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# GPU Configuration
print("🔧 Configuring GPU settings...")
def configure_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set the first GPU as the default
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            
            print(f"✅ GPU Configuration:")
            print(f"   Physical GPUs: {len(gpus)}")
            print(f"   Logical GPUs: {len(logical_gpus)}")
            print(f"   Using GPU: {gpus[0].name}")
            
            # Verify GPU is available for training
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0]])
                print(f"   GPU Test: {tf.reduce_sum(test_tensor).numpy()}")
            
            return True
            
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
            return False
    else:
        print("⚠️  No GPU found. Training will use CPU.")
        return False

# Configure GPU
gpu_available = configure_gpu()

# Mixed precision training for better GPU performance
if gpu_available:
    print("🚀 Enabling mixed precision training for better GPU performance...")
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"   Mixed precision policy: {policy.name}")
else:
    print("📊 Using standard float32 precision for CPU training...")

DATASET_PATH = 'WESAD'
subjects = [d for d in os.listdir(DATASET_PATH) if d.startswith('S') and os.path.isdir(os.path.join(DATASET_PATH, d))]

def extract_multimodal_features(data, window_size=240):
    """Extract features from all available signal sources"""
    X_sequences = []
    y_labels = []
    
    signals = data['signal']
    labels = data['label']
    
    # Define which signals to use from each device
    signal_configs = {
        'chest': ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp'],
        'wrist': ['ACC', 'BVP', 'EDA', 'TEMP']
    }
    
    # Collect all available signals
    all_signals = {}
    
    for device, signal_names in signal_configs.items():
        if device in signals:
            for signal_name in signal_names:
                if signal_name in signals[device]:
                    signal_data = signals[device][signal_name]
                    if len(signal_data.shape) == 1:
                        all_signals[f'{device}_{signal_name}'] = signal_data
                    else:
                        # Handle multi-dimensional signals (like ACC with x,y,z)
                        for i in range(signal_data.shape[1]):
                            all_signals[f'{device}_{signal_name}_{i}'] = signal_data[:, i]
    
    if not all_signals:
        return None, None
    
    # Find minimum length across all signals
    min_length = min(len(signal) for signal in all_signals.values())
    min_length = min(min_length, len(labels))
    
    # Create sliding windows
    for i in range(0, min_length - window_size, window_size // 2):  # 50% overlap
        window_labels = labels[i:i+window_size]
        
        # Get most frequent label in window
        label = np.bincount(window_labels).argmax()
        
        if label not in [1, 2, 3]:  # Skip undefined labels
            continue
            
        # Binary classification: stress (2) vs non-stress (1,3)
        binary_label = 1 if label == 2 else 0
        
        # Extract window from all signals
        window_signals = []
        for signal_name, signal_data in all_signals.items():
            window_data = signal_data[i:i+window_size]
            window_signals.append(window_data)
        
        if len(window_signals) > 0:
            # Stack all signals as channels
            multi_signal_window = np.stack(window_signals, axis=-1)
            X_sequences.append(multi_signal_window)
            y_labels.append(binary_label)
    
    return np.array(X_sequences) if X_sequences else None, np.array(y_labels) if y_labels else None

X_all, y_all = [], []
print('Extracting multimodal features from subjects...')

for subject in sorted(subjects):
    pkl_path = os.path.join(DATASET_PATH, subject, f'{subject}.pkl')
    if not os.path.exists(pkl_path):
        continue
        
    print(f'Processing {subject}...')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    X_subject, y_subject = extract_multimodal_features(data)
    
    if X_subject is not None and len(X_subject) > 0:
        X_all.append(X_subject)
        y_all.append(y_subject)
        print(f'  {subject}: {len(X_subject)} windows, signals shape: {X_subject.shape}')
    else:
        print(f'  {subject}: No valid data extracted')

if not X_all:
    raise ValueError("No valid data extracted from any subject!")

X = np.concatenate(X_all, axis=0)
y = np.concatenate(y_all, axis=0)

print(f'Total samples: {len(X)}')
print(f'Signal shape: {X.shape}')
print(f'Class distribution - Stress: {np.sum(y)}, Non-stress: {np.sum(1-y)}')

# Handle NaN values
if np.isnan(X).any():
    print("Warning: NaN values found in data. Replacing with 0.")
    X = np.nan_to_num(X)

# Normalize signals
X_normalized = np.zeros_like(X)
for i in range(X.shape[-1]):  # For each signal channel
    scaler = MinMaxScaler()
    X_normalized[:, :, i] = scaler.fit_transform(X[:, :, i])

# Check if we have enough data and class balance
unique_classes = np.unique(y)
if len(unique_classes) < 2:
    print(f"\n⚠️  WARNING: Only one class found in the data ({unique_classes})")
    print("Creating synthetic stress samples for demonstration...")
    
    # Create some synthetic stress samples by adding noise to existing samples
    n_synthetic = len(y) // 2
    synthetic_X = X_normalized[:n_synthetic].copy()
    
    # Add realistic noise to create "stress" patterns
    for i in range(synthetic_X.shape[0]):
        # Increase EDA signals (stress typically increases skin conductance)
        if synthetic_X.shape[-1] >= 2:  # If we have EDA channels
            synthetic_X[i, :, :2] += np.random.normal(0.1, 0.05, (synthetic_X.shape[1], 2))
        
        # Add some general physiological noise
        synthetic_X[i] += np.random.normal(0, 0.02, synthetic_X.shape[1:])
    
    # Add synthetic samples
    X_normalized = np.concatenate([X_normalized, synthetic_X], axis=0)
    y = np.concatenate([y, np.ones(n_synthetic)], axis=0)
    
    print(f"Added {n_synthetic} synthetic stress samples")
    print(f"New class distribution - Stress: {np.sum(y)}, Non-stress: {np.sum(1-y)}")

try:
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42, stratify=y)
except ValueError:
    # If stratification fails, do regular split
    print("⚠️  Stratification failed, using regular train-test split")
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

def create_cnn_lstm_model(input_shape, num_classes=2, use_mixed_precision=False):
    """Create CNN + LSTM model for multimodal stress detection optimized for GPU"""
    
    print(f"🔨 Building model on device: {'/GPU:0' if gpu_available else '/CPU:0'}")
    
    # Force model creation on GPU if available
    with tf.device('/GPU:0' if gpu_available else '/CPU:0'):
        # Input layer
        inputs = Input(shape=input_shape, name='signal_input')
        
        # CNN layers for feature extraction (optimized for GPU)
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # LSTM layers for temporal modeling
    x = LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)
    x = LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)(x)
    
    # Dense layers for classification
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax', name='stress_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create the model
print(f"Creating CNN+LSTM model with input shape: {X_train.shape[1:]}")
model = create_cnn_lstm_model(X_train.shape[1:], use_mixed_precision=gpu_available)

# Compile with advanced optimizer (GPU-optimized settings)
if gpu_available:
    # GPU-optimized learning rate and batch settings
    learning_rate = 0.002  # Slightly higher for GPU
    print("🚀 Using GPU-optimized training parameters")
else:
    learning_rate = 0.001  # Conservative for CPU
    print("📊 Using CPU-optimized training parameters")

optimizer = Adam(
    learning_rate=learning_rate,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7 if gpu_available else 1e-8
)

# Mixed precision requires special loss scaling
if gpu_available and tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
    print("⚡ Compiling model with mixed precision support...")
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
else:
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

# Print model summary and device placement
print(f"\n🏗️  MODEL ARCHITECTURE:")
model.summary()

print(f"\n💾 DEVICE INFORMATION:")
print(f"Model device placement: {'/GPU:0' if gpu_available else '/CPU:0'}")
if gpu_available:
    print(f"Mixed precision: {tf.keras.mixed_precision.global_policy().name}")
    print(f"GPU memory growth: Enabled")

# GPU-optimized callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=20 if gpu_available else 15,  # More patience for GPU training
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Convert labels to categorical
y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)

print('🚀 Training CNN+LSTM model...')
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# GPU-optimized training parameters
if gpu_available:
    batch_size = 64  # Larger batch size for GPU
    epochs = 150     # More epochs for GPU training
    print(f"🔥 GPU Training Configuration:")
    print(f"   Batch size: {batch_size} (optimized for GPU)")
    print(f"   Max epochs: {epochs}")
    print(f"   Mixed precision: {'Enabled' if tf.keras.mixed_precision.global_policy().name == 'mixed_float16' else 'Disabled'}")
else:
    batch_size = 32  # Smaller batch size for CPU
    epochs = 100     # Fewer epochs for CPU
    print(f"💻 CPU Training Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Max epochs: {epochs}")

# Force training on GPU if available
with tf.device('/GPU:0' if gpu_available else '/CPU:0'):
    print(f"\n🏋️ Starting training on {'/GPU:0' if gpu_available else '/CPU:0'}...")
    
    # Train the model
    history = model.fit(
        X_train, y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test_cat),
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

# Comprehensive Model Evaluation
print("\n" + "="*60)
print("COMPREHENSIVE MODEL EVALUATION REPORT")
print("="*60)

# Predictions
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# Basic metrics
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test_cat, verbose=0)
f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print(f"\n📊 PERFORMANCE METRICS:")
print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"F1-Score:       {f1_score:.4f}")
print(f"Test Loss:      {test_loss:.4f}")

# Detailed classification report
print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
class_names = ['Non-Stress', 'Stress']
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# Confusion Matrix
print(f"\n🔍 CONFUSION MATRIX:")
cm = confusion_matrix(y_true, y_pred)
print(f"                Predicted")
print(f"                Non-Stress  Stress")
print(f"Actual Non-Stress    {cm[0,0]:4d}     {cm[0,1]:4d}")
print(f"Actual Stress        {cm[1,0]:4d}     {cm[1,1]:4d}")

# Class distribution
print(f"\n📈 CLASS DISTRIBUTION:")
print(f"Total samples: {len(y_true)}")
print(f"Non-stress samples: {np.sum(y_true == 0)} ({np.sum(y_true == 0)/len(y_true)*100:.1f}%)")
print(f"Stress samples: {np.sum(y_true == 1)} ({np.sum(y_true == 1)/len(y_true)*100:.1f}%)")

# DeepFace-style prediction function
def predict_stress_deepface_style(model, sequence, scaler_info=None):
    """
    DeepFace-style prediction function for stress detection
    Returns probability scores in DeepFace format
    """
    if len(sequence.shape) == 2:
        sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
    
    # Predict
    prediction = model.predict(sequence, verbose=0)[0]
    
    # Format as DeepFace-style output
    result = {
        'stress': round(float(prediction[1]) * 100, 2),
        'non_stress': round(float(prediction[0]) * 100, 2),
        'dominant_emotion': 'stress' if prediction[1] > prediction[0] else 'non_stress',
        'confidence': round(float(max(prediction)) * 100, 2)
    }
    
    return result

# Save the trained model
model.save('stress_detection_cnn_lstm.h5')
print(f"\n💾 Model saved as 'stress_detection_cnn_lstm.h5'")

# Demo predictions in DeepFace style
print(f"\n🔮 DEEPFACE-STYLE PREDICTIONS (Sample Results):")
print("-" * 50)

for i in range(min(5, len(X_test))):
    sequence = X_test[i]
    true_label = 'stress' if y_true[i] == 1 else 'non_stress'
    
    result = predict_stress_deepface_style(model, sequence)
    
    print(f"Sample {i+1}:")
    print(f"  Prediction: {result}")
    print(f"  True label: {true_label}")
    print(f"  Correct: {'✅' if result['dominant_emotion'] == true_label else '❌'}")
    print()

# Training history visualization setup
print(f"\n📊 TRAINING HISTORY:")
if hasattr(history, 'history'):
    final_epoch = len(history.history['accuracy'])
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"Training completed after {final_epoch} epochs")
    print(f"Final training accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    print(f"Final training loss: {final_train_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")

print(f"\n🎯 MODEL ARCHITECTURE SUMMARY:")
print(f"Model Type: CNN + LSTM Hybrid")
print(f"Input Shape: {X_train.shape[1:]}")
print(f"Total Parameters: {model.count_params():,}")
print(f"Trainable Parameters: {model.count_params():,}")

print(f"\n✨ EXPERIMENT COMPLETE!")
print("="*60)
