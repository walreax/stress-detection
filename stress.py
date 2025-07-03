import os
import pickle

DATASET_PATH = 'WESAD'
subjects = [d for d in os.listdir(DATASET_PATH) if d.startswith('S') and os.path.isdir(os.path.join(DATASET_PATH, d))]

for subject in sorted(subjects):
    pkl_path = os.path.join(DATASET_PATH, subject, f'{subject}.pkl')
    if os.path.exists(pkl_path):
        print(f'{subject}')
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        print(list(data.keys()))
        if 'signal' in data:
            print(list(data['signal'].keys()))
            for device, signals in data['signal'].items():
                print(f'{device}: {list(signals.keys())}')
                for sig, arr in signals.items():
                    try:
                        print(f'{sig}: shape {arr.shape}')
                    except AttributeError:
                        print(f'{sig}: length {len(arr)}')
        if 'label' in data:
            print(set(data['label']))
        print()
    else:
        print(f'No PKL file found for {subject}')
