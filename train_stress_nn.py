import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# This relative path will work because we are running the script from this folder
DATASET_PATH = 'WESAD'
subjects = [d for d in os.listdir(DATASET_PATH) if d.startswith('S') and os.path.isdir(os.path.join(DATASET_PATH, d))]

X, y = [], []
print('Extracting features from subjects...')
for subject in sorted(subjects):
    pkl_path = os.path.join(DATASET_PATH, subject, f'{subject}.pkl')
    if not os.path.exists(pkl_path):
        continue
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    if 'wrist' not in data['signal']:
        print(f'Skipping {subject}: no wrist data')
        continue
    wrist = data['signal']['wrist']

    try:
        # We only use EDA and TEMP because HR is missing from the dataset
        eda = wrist['EDA']
        temp = wrist['TEMP']
    except KeyError:
        print(f"Skipping {subject}: missing EDA or TEMP data")
        continue

    labels = data['label']
    window_size = 60 * 4 # 60 seconds * 4 Hz sampling rate
    for i in range(0, len(labels) - window_size, window_size):
        window_label = labels[i:i+window_size]
        label = np.bincount(window_label).argmax()

        if label not in [1, 2, 3]: # Baseline, Stress, or Amusement
            continue
        
        # Binary classification: 1 for stress, 0 for non-stress (baseline/amusement)
        y.append(1 if label == 2 else 0)
        
        # We only create features for EDA and TEMP
        features = [
            np.mean(eda[i:i+window_size]), np.std(eda[i:i+window_size]),
            np.mean(temp[i:i+window_size]), np.std(temp[i:i+window_size]),
        ]
        X.append(features)

X = np.array(X)
y = np.array(y)
print(f'Total samples: {len(X)}')

# Handle potential NaN values from data
if np.isnan(X).any():
    print("Warning: NaN values found in data. Replacing with 0.")
    X = np.nan_to_num(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)

model = Sequential([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),
    Dense(8, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('Training neural network (3 epochs for demo)...')
model.fit(X_train, y_train_cat, epochs=3, batch_size=32, validation_data=(X_test, y_test_cat))

print("\nEvaluating model performance...")
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
