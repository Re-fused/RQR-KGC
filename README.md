# RQR-KGC implementation

This directory is a standalone copy of the implementation used for the paper.
The original experimental workspace is not modified or required at runtime.

## Environment

Python 3.10 and a CUDA-enabled PyTorch installation are recommended.

```bash
python -m pip install -r requirements.txt
```

## Data layout

Each dataset directory must contain the standard KGE files:

```text
DATA_PATH/
  entities.dict
  relations.dict
  train.txt
  valid.txt
  test.txt
```

Dictionary lines use `id<TAB>name`; triple lines use
`head<TAB>relation<TAB>tail`.

## Training and evaluation

Both scripts train from scratch, select the checkpoint with the highest
validation MRR, and run filtered test evaluation on that checkpoint.

```bash
# Run from the repository root with default relative paths:
# data/fb15k-237 -> outputs/fb15k237 on GPU 0
bash code/train_fb15k237.sh

# data/wn18rr -> outputs/wn18rr on GPU 0
bash code/train_wn18rr.sh

# Optional custom relative paths and GPU index
bash code/train_fb15k237.sh data/fb15k-237 outputs/fb-run 2
bash code/train_wn18rr.sh data/wn18rr outputs/wn-run 7
```

All paths are interpreted relative to the repository root. The scripts first
change to that directory, so they work the same way even when invoked from a
different current directory.

The output directory must not already contain a checkpoint, configuration, or
training log. This guard prevents accidental overwrite of existing results.

## Main implementation

- `model.py` contains only RQR-KGC: Hamilton products, relative frame decoding,
  analytic reciprocal relations, bounded directional residuals, endpoint
  modulation, and the final score.
- `data.py` implements filtered negative sampling and evaluation datasets.
- `curriculum.py` implements relation-pool hard-negative scheduling.
- `train.py` implements optimization, validation checkpoint selection, and
  filtered test evaluation.

No baseline model or abandoned experimental model is included.
