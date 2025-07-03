import os
import pickle
import pandas as pd

DATASET_PATH = 'WESAD'
subjects = [d for d in os.listdir(DATASET_PATH) if d.startswith('S') and os.path.isdir(os.path.join(DATASET_PATH, d))]

rows = []
for subject in sorted(subjects):
    pkl_path = os.path.join(DATASET_PATH, subject, f'{subject}.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        devices = list(data['signal'].keys()) if 'signal' in data else []
        for device in devices:
            signals = data['signal'][device]
            for sig, arr in signals.items():
                n_samples = getattr(arr, 'shape', [len(arr)])[0] if hasattr(arr, 'shape') else len(arr)
                rows.append({
                    'Subject': subject,
                    'Device': device,
                    'Signal': sig,
                    'Samples': n_samples
                })
        if 'label' in data:
            label_series = pd.Series(data['label'])
            value_counts = label_series.value_counts().sort_index()
            label_str = ', '.join([f'{int(l)}: {int(c)}' for l, c in zip(value_counts.index, value_counts.values)])
        else:
            label_str = 'N/A'
        rows.append({
            'Subject': subject,
            'Device': 'ALL',
            'Signal': 'Label Distribution',
            'Samples': label_str
        })
    else:
        rows.append({'Subject': subject, 'Device': 'N/A', 'Signal': 'No PKL', 'Samples': 0})

df = pd.DataFrame(rows)
print('\nWESAD Dataset EDA Table:\n')
try:
    md_table = df.to_markdown(index=False)
    print(md_table)
    with open('wesad_eda_table.md', 'w', encoding='utf-8') as f:
        f.write('# WESAD Dataset EDA Table\n\n')
        f.write(md_table)
        f.write('\n')
except ImportError:
    print(df)
    df.to_csv('wesad_eda_table.csv', index=False)
    print('Saved as CSV: wesad_eda_table.csv')
