EEG-DINO

This repository contains the code for pretraining and evaluating a Self Supervised learning model with EEG-DINO.

Project layout

- `train.py`, `evaluate.py`, `evaluate_bci.py`, and `run_significance_test_sleep.py` are the launch scripts.
- `model/` contains the core implementation files: `channel_aware_sampling.py`, `dpe_module.py`, `eeg_dino_model.py`, `losses.py`, and `tfe_module.py`.
- `experiments/` contains experimental analysis scripts
- `configs.py`, `datasets.py`, `utils.py`, and `requirements.txt` stay at the root.
- `checkpoints/` stores training outputs.
- The raw dataset files stay on the server, as in the original setup.

Project tree

```text
eeg-dino/
  README.md
  requirements.txt
  train.py
  evaluate.py
  evaluate_bci.py
  run_significance_test_sleep.py
  configs.py
  datasets.py
  utils.py
  model/
    __init__.py
    channel_aware_sampling.py
    dpe_module.py
    eeg_dino_model.py
    losses.py
    tfe_module.py
  experiments/
    analyze_representation_quality.py
  checkpoints/
```

Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dataset paths

Set a `.env` file at the project root to:

```bash
SLEEP_EDF_PATH="/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/sleep_edf/"
BCI_2A_PATH="/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/Raw_data/BCICIV_2a_gdf/"
BCI_2B_PATH="/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/Raw_data/BCICIV_2b_gdf/"
```

Run training (for theta masking for example)

```bash
python train.py --preset tiny --dataset sleep_edf --mask_strategy theta --n_epochs 30 --save_dir checkpoints/myrun
```

This will save the run under `checkpoints/myrun/theta_seed42/` by default, so the best model ends up at `checkpoints/myrun/theta_seed42/best_model.pth`.

Run the significance test

```bash
python run_significance_test_sleep.py --strategies spatiotemporal theta --seeds 5 --n_epochs 30
```

You can point `--checkpoint_root` to a different directory if your saved runs live elsewhere.
If the checkpoints already exist, add `--skip_training` to evaluate them directly without retraining.

Analyze representation quality

```bash
python experiments/analyze_representation_quality.py \
  --strategies none theta random \ 
  --seeds 42 43 44 45 46 \
  --checkpoint_dir checkpoints/significance_sleep_model \
  --preset tiny
```


Notes

- Use `checkpoints/` for all saved runs and results.
- For quick local tests, reduce `--n_epochs` and `--seeds`.
