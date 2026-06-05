import os
import numpy as np

HERE    = os.path.dirname(os.path.abspath(__file__))
SPLITS  = os.path.join(HERE, 'fold_splits', 'unet')
FOLDERS = ['20260413-095040', '20260410-144309', '20260410-144309']
N_FOLDS = 3

def load_splits(folder):
    result = {}
    for f in range(1, N_FOLDS + 1):
        path = os.path.join(SPLITS, folder, f'fold_{f}.npz')
        if not os.path.exists(path):
            result[f] = None
            continue
        d = np.load(path, allow_pickle=True)
        result[f] = {
            'train': [tuple(x) for x in d['train_samples']],
            'val':   [tuple(x) for x in d['val_samples']],
        }
    return result

data = {folder: load_splits(folder) for folder in FOLDERS}

for fold in range(1, N_FOLDS + 1):
    print(f'\n{"="*60}  FOLD {fold}  {"="*60}')
    for folder in FOLDERS:
        splits = data[folder][fold]
        if splits is None:
            print(f'  [{folder}]  NOT FOUND')
            continue
        val_str   = ', '.join(f'{s}/{l}' for s, l in splits['val'])
        train_str = ', '.join(f'{s}/{l}' for s, l in splits['train'])
        print(f'  [{folder}]')
        print(f'    VAL   ({len(splits["val"]):2d}): {val_str}')
        print(f'    TRAIN ({len(splits["train"]):2d}): {train_str}')

# Cross-folder val agreement check
print(f'\n{"="*60}  VAL AGREEMENT  {"="*60}')
for fold in range(1, N_FOLDS + 1):
    sets = []
    for folder in FOLDERS:
        s = data[folder][fold]
        sets.append(set(s['val']) if s else set())
    same = all(s == sets[0] for s in sets)
    print(f'  Fold {fold}: {"IDENTICAL" if same else "DIFFER"}  '
          + '  |  '.join(f'[{FOLDERS[i]}] {sorted(sets[i])}' for i in range(len(FOLDERS))))
