---
dataset_info:
  features:
  - name: caption
    dtype: string
  - name: motion
    sequence:
      sequence: float32
  - name: meta_data
    struct:
    - name: duration
      dtype: float64
    - name: name
      dtype: string
    - name: num_frames
      dtype: int64
  splits:
  - name: train
    num_bytes: 3492188041
    num_examples: 23384
  - name: val
    num_bytes: 219480031
    num_examples: 1460
  - name: test
    num_bytes: 654286564
    num_examples: 4384
  download_size: 4361988681
  dataset_size: 4365954636
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: val
    path: data/val-*
  - split: test
    path: data/test-*
---
