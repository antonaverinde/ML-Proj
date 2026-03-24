# Plan: V2 HDF5 Data Loading Pipeline for NNsTorchV2

**Status:** Ready to implement — all exploration complete.

## Summary

Create an alternative data loading pipeline that reads from consolidated HDF5 files
(`Data_ML_V1_h5`) instead of per-sample NPZ/NPY files. The public API is identical to V1;
switching pipelines requires **one import line change** in the notebook.

---

## H5 Data Source

**Location:** `set_load_path(sys) + '/Taris/Data_ML_V1_h5'`
(GPU: `/home/aaverin/RZ-Dienste/hpc-user/aaverin/2025/2025-11-04-Av-ZIKA-Mirko-Taris-Hologen-2kw-measurements/Taris/Data_ML_V1_h5`)

**Three files:**
| File | Content | Key pattern |
|---|---|---|
| `dataset_masks_v1.h5` | Binary masks (H×W) | `/{power_mode}/{sample}/{location}/BinMask_0/data` |
| `dataset_raw_v1.h5` | Raw temperature data (H×W×280) | `/{power_mode}/{sample}/{location}/temperature/data` |
| `dataset_features_v1.h5` | PCA/ICA/PPT components (H×W each) | `/{power_mode}/{sample}/{location}/PCA\|ICA\|PPT/a=N_width=M/[Phase\|Amp/]component=K/data` |

**Internal structure confirmed:**
- **Location names** (string keys): `bottom_left`, `bottom_right`, `mid_left`, `mid_right`, `top_left`, `top_right`
- **Power modes**: `2kw_left`, `2kw_right`, `4kw_both`
- **Samples per power mode**: `s0`–`s6` (7 samples)
- **PCA variants**: `a=10_width=110`, `a=10_width=280` (6 components each)
- **PPT variants**: `a=0_width=110`, `a=0_width=280` (Phase + Amp, many components each)
- **ICA variants**: `a=0_width=280` (6 components)
- **mask_type** is ignored in V2 — only `BinMask_0` exists (no alternatives)

---

## Files to Create (5 total)

```
NNsTorchV2/
├── core/
│   ├── data_discovery_V2.py     ← NEW (replaces data_discovery.py)
│   └── data_loading_V2.py       ← NEW (replaces data_loading.py)
└── HybridTrainV2/
    ├── hybrid_manager_V2.py     ← NEW (replaces hybrid_manager.py)
    └── components/
        ├── hybrid_utils_V2.py   ← NEW (replaces hybrid_utils.py)
        └── infrastructure_V2.py ← NEW (replaces infrastructure.py)
```

No existing files are modified.

---

## File 1: `NNsTorchV2/core/data_discovery_V2.py`

### Public API (identical names to V1, different signatures for V2 internals)

```python
def discover_data_files_for_location(
    h5_base_dir: str,       # directory containing the 3 H5 files
    power_mode: str,        # e.g. '4kw_both'
    sample_name: str,       # e.g. 's0'
    location_name: str,     # e.g. 'bottom_left'  ← str not int (V2 difference)
    mask_type: str = 'normal',    # accepted but IGNORED in V2 (only BinMask_0)
    data_regime: str = 'postprocessed'
) -> Optional[Dict[str, Any]]:

def discover_samples(
    h5_base_dir: str,
    power_mode: str,
    dirs: List[int],
    mask_type: str = 'normal',
    data_regime: str = 'postprocessed'
) -> List[Tuple[str, str]]:   # (sample_name, location_name) — both strings
```

### V2 files_info dict format

```python
# postprocessed regime:
{
    'mask_h5':     '/path/dataset_masks_v1.h5',
    'mask_key':    '/4kw_both/s0/bottom_left/BinMask_0/data',
    'features_h5': '/path/dataset_features_v1.h5',
    'base_key':    '/4kw_both/s0/bottom_left',
    'PCA': [(10, 110, 'PCA/a=10_width=110'), (10, 280, 'PCA/a=10_width=280')],
    'PPT': [(0, 110, 'PPT/a=0_width=110'), (0, 280, 'PPT/a=0_width=280')],
    'ICA': [(0, 280, 'ICA/a=0_width=280')],
    'data_regime': 'postprocessed',
    'shape': (H, W),    # cached here so infrastructure avoids re-opening H5
}

# raw regime:
{
    'mask_h5':  '/path/dataset_masks_v1.h5',
    'mask_key': '/4kw_both/s0/bottom_left/BinMask_0/data',
    'raw_h5':   '/path/dataset_raw_v1.h5',
    'raw_key':  '/4kw_both/s0/bottom_left/temperature/data',
    'data_regime': 'raw',
    'shape': (H, W),
}
```

### Implementation sketch

```python
import h5py, os, re
from typing import Dict, List, Optional, Any, Tuple

def discover_data_files_for_location(h5_base_dir, power_mode, sample_name,
                                     location_name, mask_type='normal',
                                     data_regime='postprocessed'):
    masks_path    = os.path.join(h5_base_dir, 'dataset_masks_v1.h5')
    features_path = os.path.join(h5_base_dir, 'dataset_features_v1.h5')
    raw_path      = os.path.join(h5_base_dir, 'dataset_raw_v1.h5')

    base_key = f'/{power_mode}/{sample_name}/{location_name}'
    mask_key = f'{base_key}/BinMask_0/data'

    with h5py.File(masks_path, 'r') as f:
        if mask_key not in f:
            return None
        H, W = f[mask_key].shape

    if data_regime == 'raw':
        raw_key = f'{base_key}/temperature/data'
        with h5py.File(raw_path, 'r') as f:
            if raw_key not in f:
                return None
        return {'mask_h5': masks_path, 'mask_key': mask_key,
                'raw_h5': raw_path, 'raw_key': raw_key,
                'data_regime': 'raw', 'shape': (H, W)}

    # postprocessed: enumerate feature groups
    pca, ppt, ica = [], [], []
    with h5py.File(features_path, 'r') as f:
        if base_key not in f:
            return None
        grp = f[base_key]
        for feat_type, container in [('PCA', pca), ('PPT', ppt), ('ICA', ica)]:
            if feat_type in grp:
                for variant in grp[feat_type]:
                    m = re.match(r'a=(\d+)_width=(\d+)', variant)
                    if m:
                        container.append((int(m.group(1)), int(m.group(2)),
                                          f'{feat_type}/{variant}'))
    pca.sort(); ppt.sort(); ica.sort()

    if not pca or not ppt:
        return None

    return {'mask_h5': masks_path, 'mask_key': mask_key,
            'features_h5': features_path, 'base_key': base_key,
            'PCA': pca, 'PPT': ppt, 'ICA': ica,
            'data_regime': 'postprocessed', 'shape': (H, W)}


def discover_samples(h5_base_dir, power_mode, dirs,
                     mask_type='normal', data_regime='postprocessed'):
    masks_path = os.path.join(h5_base_dir, 'dataset_masks_v1.h5')
    samples = []
    with h5py.File(masks_path, 'r') as f:
        if power_mode not in f:
            return []
        pm = f[power_mode]
        all_samples = sorted(pm.keys())
        if dirs:
            # handles multi-digit: s0, s1, ..., s10, s11, ...
            all_samples = [s for s in all_samples
                           if s.startswith('s') and s[1:].isdigit()
                           and int(s[1:]) in dirs]
        for sname in all_samples:
            for loc_name in sorted(pm[sname].keys()):
                fi = discover_data_files_for_location(
                    h5_base_dir, power_mode, sname, loc_name,
                    mask_type=mask_type, data_regime=data_regime)
                if fi is not None:
                    samples.append((sname, loc_name))
    return samples
```

---

## File 2: `NNsTorchV2/core/data_loading_V2.py`

### Public API (identical signatures to V1)

```python
def load_and_aggregate_location(
    files_info: Dict[str, Any],
    ppt_phases: Union[str, int] = 'all',
    ppt_amps: int = 6,
    invert_mask: bool = False,
    mask_only: bool = False,
    data_regime: str = 'postprocessed'
) -> Tuple[np.ndarray, np.ndarray]:   # (H×W×C, H×W)

def calculate_total_channels(
    files_info: Dict[str, Any],
    ppt_phases: Union[str, int] = 'all',
    ppt_amps: int = 6,
    data_regime: str = 'postprocessed'
) -> int:
```

### Key helpers

```python
def _sort_component_keys(keys) -> List[str]:
    """Sort 'component=N' strings by integer N, NOT lexicographically."""
    return sorted(keys, key=lambda k: int(re.match(r'component=(\d+)', k).group(1)))

def _read_components(h5file, group_path: str) -> np.ndarray:
    """Read all component=N/data datasets; stack to (H, W, N)."""
    grp = h5file[group_path]
    comp_keys = _sort_component_keys(
        [k for k in grp.keys() if k.startswith('component=')])
    arrays = [grp[k]['data'][...].astype(np.float32) for k in comp_keys]
    return np.stack(arrays, axis=2)
```

### load_and_aggregate_location logic

```python
# 1. Load mask
with h5py.File(files_info['mask_h5'], 'r') as f:
    mask = f[files_info['mask_key']][...].astype(np.float32)
if invert_mask: mask = 1.0 - mask
if mask_only: return mask, mask

# 2a. Raw regime
if data_regime == 'raw':
    with h5py.File(files_info['raw_h5'], 'r') as f:
        return f[files_info['raw_key']][...].astype(np.float32), mask

# 2b. Postprocessed: open features H5 ONCE, read all inside
parts = []
base = files_info['base_key']
with h5py.File(files_info['features_h5'], 'r') as f:
    for a, w, rel in files_info['PCA']:
        parts.append(_read_components(f, f'{base}/{rel}'))           # (H,W,n)

    for a, w, rel in files_info['PPT']:
        phase = _read_components(f, f'{base}/{rel}/Phase')           # (H,W,n_phases)
        amp   = _read_components(f, f'{base}/{rel}/Amp')             # (H,W,n_amps)
        n_ph  = phase.shape[2] if ppt_phases == 'all' else int(ppt_phases)
        parts.append(phase[:, :, :n_ph])
        parts.append(amp[:, :, :ppt_amps])

    for a, w, rel in files_info.get('ICA', []):
        parts.append(_read_components(f, f'{base}/{rel}'))

return np.concatenate(parts, axis=2), mask
```

### calculate_total_channels logic

```python
if data_regime == 'raw':
    with h5py.File(files_info['raw_h5'], 'r') as f:
        return f[files_info['raw_key']].shape[2]

total = 0
base = files_info['base_key']
with h5py.File(files_info['features_h5'], 'r') as f:
    for a, w, rel in files_info['PCA']:
        grp = f[f'{base}/{rel}']
        total += len([k for k in grp if k.startswith('component=')])

    for a, w, rel in files_info['PPT']:
        ph_grp = f[f'{base}/{rel}/Phase']
        am_grp = f[f'{base}/{rel}/Amp']
        n_ph   = len([k for k in ph_grp if k.startswith('component=')])
        n_am   = len([k for k in am_grp if k.startswith('component=')])
        total += (n_ph if ppt_phases == 'all' else min(int(ppt_phases), n_ph))
        total += min(ppt_amps, n_am)   # ← min() to match actual loaded slice

    for a, w, rel in files_info.get('ICA', []):
        grp = f[f'{base}/{rel}']
        total += len([k for k in grp if k.startswith('component=')])

return total
```

---

## File 3: `NNsTorchV2/HybridTrainV2/components/hybrid_utils_V2.py`

Copy of `hybrid_utils.py`. **Only changes:**

```python
# Changed imports:
from ...core.data_discovery_V2 import discover_data_files_for_location
from ...core.data_loading_V2 import load_and_aggregate_location

# Changed discovery call inside HybridPatchDataset.__init__ loop:
# V1: sample_dir = os.path.join(load_path, power_mode, sample_name)
#     fi = discover_data_files_for_location(sample_dir, location_idx, ...)
# V2:
fi = discover_data_files_for_location(
    load_path, power_mode, sample_name, location_name,  # ← location is str
    mask_type=mask_type, data_regime=data_regime)
```

Everything else (XGB inference, patch extraction, transforms, `__getitem__`) is unchanged.
Note: `load_path` parameter now semantically means `h5_base_dir` but the name stays the same.

---

## File 4: `NNsTorchV2/HybridTrainV2/components/infrastructure_V2.py`

Copy of `infrastructure.py`. **Only changes:**

```python
# Changed imports:
from ...core.data_discovery_V2 import discover_samples, discover_data_files_for_location
from ...core.data_loading_V2 import calculate_total_channels

# discover_samples() return type is now List[Tuple[str, str]] — no code change needed

# determine_input_shape: discovery call changes to V2 signature:
fi = discover_data_files_for_location(
    self.load_path, self.power_mode, sample, loc_name,
    self.mask_type, self.data_regime)

# In full_padding loop: use cached shape — avoids re-opening H5 per sample:
# V1: mask = np.load(fi2['mask'])  →  h, w = mask.shape
# V2:
fi2 = discover_data_files_for_location(...)
h, w = fi2['shape']   # cached during discovery
```

---

## File 5: `NNsTorchV2/HybridTrainV2/hybrid_manager_V2.py`

Copy of `hybrid_manager.py`. **Only changes:**

### Module docstring (new):
```
HybridTrainingManager V2 — identical training logic, H5 data source.
Switch with one import line:
    from NNsTorchV2.HybridTrainV2.hybrid_manager_V2 import HybridTrainingManager
```

### 4 import lines changed:
```python
from ..core.data_discovery_V2 import discover_data_files_for_location
from ..core.data_loading_V2 import load_and_aggregate_location
from .components.hybrid_utils_V2 import create_hybrid_dataloader
from .components.infrastructure_V2 import TrainingInfrastructure
```

### Default parameter changed:
```python
subfolder_name: str = 'Taris/Data_ML_V1_h5',   # was 'Taris/Data_ML_V3'
```

### compute_weights body change:
```python
# V1: mask = np.load(fi['mask'])
# V2:
with h5py.File(fi['mask_h5'], 'r') as f:
    mask = f[fi['mask_key']][...]
```

---

## Notebook Change (single import line)

```python
# Before (V1):
from NNsTorchV2.HybridTrainV2.hybrid_manager import HybridTrainingManager

# After (V2):
from NNsTorchV2.HybridTrainV2.hybrid_manager_V2 import HybridTrainingManager
```

Also set `subfolder_name='Taris/Data_ML_V1_h5'` in the constructor.

---

## Edge Cases

| Issue | Location | Fix |
|---|---|---|
| Lexicographic component sort breaks for N≥10 | `data_loading_V2` | `_sort_component_keys` parses `int(N)`, never uses string sort |
| `ppt_phases='all'` vs int | loading + channel count | `if ppt_phases == 'all'` branch everywhere |
| `ppt_amps` > available components | `calculate_total_channels` | `min(ppt_amps, n_am)` — must match actual loaded slice |
| Multi-digit sample names (`s10`, `s11`) | `discover_samples` | `int(s[1:]) in dirs`, not `int(d[-1])` as in V1 |
| `mask_type` has no effect | `discover_data_files_for_location` | Accept param (API compat), silently ignore, always use `BinMask_0` |
| H5 + `num_workers > 0` | `HybridPatchDataset` | All H5 reads happen in `__init__` → data stored in `self.patches` as numpy; workers never touch H5 |

---

## Verification Steps

```python
# 1. Import and construct
from NNsTorchV2.HybridTrainV2.hybrid_manager_V2 import HybridTrainingManager
mgr = HybridTrainingManager(
    model_name='test_v2', sys='GPU', mode='nn_only',
    subfolder_name='Taris/Data_ML_V1_h5',
    xgb_model_path=None, power_mode='4kw_both')

# 2. Check discovery
print(len(mgr.all_samples))   # should be > 0
print(mgr.input_shape)        # should print (C, H, W)

# 3. Single batch test
from NNsTorchV2.HybridTrainV2.components.hybrid_utils_V2 import create_hybrid_dataloader
loader = create_hybrid_dataloader(
    mgr.all_samples[:2], xgb_model=None,
    load_path=mgr.load_path, power_mode='4kw_both', batch_size=1)
data, xgb, mask = next(iter(loader))
print(data.shape, xgb.shape, mask.shape)

# 4. Full smoke test
mgr.run_kfold(model_fn, n_splits=2, epochs=2)
```
