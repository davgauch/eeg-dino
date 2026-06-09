EEG-DINO

Overview

EEG-DINO provides code to pretrain and evaluate a self-supervised EEG representation model (EEG-DINO) and the analysis pipelines used in the project. The code implements state-of-the-art self-supervised EEG representation learning (DINO-style) and includes utilities for downstream evaluation and significance testing.

Quick structure

- **Launch scripts**: `train.py`, `evaluate.py`, `evaluate_bci.py`, `run_significance_test_sleep.py`.
- **Model code**: [model/](model/) contains the core implementation.
- **Analysis**: [experiments/](experiments/) contains downstream and visualization utilities.
- **Checkpoints**: `checkpoints/` stores saved runs and best models.

Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Datasets

Set dataset paths as environment variables (or in a `.env` file) before running scripts. Example (exact lab server paths used for experiments):

```bash
SLEEP_EDF_PATH="/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/sleep_edf/"
BCI_2A_PATH="/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/Raw_data/BCICIV_2a_gdf/"
BCI_2B_PATH="/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/Raw_data/BCICIV_2b_gdf/"
```

Reproducing results (minimal)

1. Train a small test run (use `--preset tiny` for fast runs):

```bash
python train.py --preset tiny --dataset sleep_edf --mask_strategy theta --n_epochs 30 --save_dir checkpoints/myrun
```

2. Evaluate a saved model:

```bash
python evaluate.py --checkpoint checkpoints/myrun/theta_seed42/best_model.pth --preset tiny
```

3. Run a significance test pipeline (uses `checkpoints/` by default):

```bash
python run_significance_test_sleep.py --strategies none theta random --seeds 42 43 44 --n_epochs 30
```

Where to look next

- Model implementation: [model/eeg_dino_model.py](model/eeg_dino_model.py)
- Training loop & config flags: `train.py` and `configs.py`
- Analysis scripts and outputs: [experiments/](experiments/) and `experiments/results/`

Notes

- Use `checkpoints/` for saved runs. For quick local checks, reduce `--n_epochs` and number of `--seeds`.

