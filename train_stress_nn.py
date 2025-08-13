import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


DATASET_PATH = '/Users/sakshamdua/Desktop/stress-detection/WESAD'
subjects = [d for d in os.listdir(DATASET_PATH) if d.startswith('S') and os.path.isdir(os.path.join(DATASET_PATH, d))]

X, y = [], []
print('Extracting features from subjects...')
for subject in sorted(subjects):
    pkl_path = os.path.join(DATASET_PATH, subject, f'{subject}.pkl')
    if not os.path.exists(pkl_path):
        continue
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    print(f'{subject} signal keys:', list(data['signal'].keys()))
    if 'wrist' not in data['signal']:
        print(f'Skipping {subject}: no wrist data')
        continue
    wrist = data['signal']['wrist']
    print(f'{subject} wrist keys:', list(wrist.keys()))
    try:
        eda = wrist['EDA']
        temp = wrist['TEMP']
        # hr = wrist['HR']  <- MODIFIED: Removed this line
    except KeyError:
        print(f'Skipping {subject}: missing EDA, TEMP, or HR')
        continue
    labels = data['label']
    window_size = 60 * 4
    for i in range(0, len(labels) - window_size, window_size):
        window_label = labels[i:i+window_size]
        label = np.bincount(window_label).argmax()
        if label not in [1, 2, 3]:
            continue
        y.append(1 if label == 3 else 0)
        # MODIFIED: Removed the two lines for 'hr' from the list below
        features = [
            np.mean(eda[i:i+window_size]), np.std(eda[i:i+window_size]),
            np.mean(temp[i:i+window_size]), np.std(temp[i:i+window_size]),
        ]
        X.append(features)

X = np.array(X)
y = np.array(y)
print(f'Total samples: {len(X)}')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = Sequential([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),
    Dense(8, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

y_train_cat = to_categorical(y_train, 2)
print('Training neural network (3 epochs for demo)...')
model.fit(X_train, y_train_cat, epochs=3, batch_size=32, validation_split=0.1)

def predict_stress(features):
    features_scaled = scaler.transform([features])
    prob = model.predict(features_scaled, verbose=0)[0]
    return {
        'stress': round(float(prob[1]) * 100, 2),
        'non_stress': round(float(prob[0]) * 100, 2)
    }

print('\nSample predictions (DeepFace style):')
for i in range(5):
    features = X_test[i]
    result = predict_stress(features)
    print(f'Example {i+1}:', result, '| True label:', 'stress' if y_test[i]==1 else 'non_stress')
