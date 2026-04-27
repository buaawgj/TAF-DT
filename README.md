# TAF-DT: Target-Aligned Fusion for Decision-Sequence Learning under Dynamics Shift

This repository contains the research code for **Target-Aligned Fusion with a Decision Transformer backbone (TAF-DT)**. TAF-DT studies how a target-domain decision-sequence learner should absorb externally sourced trajectories when the source and target dynamics differ.

The implementation builds on the Q-value Regularized Transformer (QT) codebase and extends it to cross-domain offline reinforcement learning with:

- **MMD-based fragment selection** for target-aligned state-structure filtering;
- **OT-based feasibility weighting** for source transitions that are locally compatible with the target domain;
- **Advantage-conditioned sequence tokens** as a more stable alternative to raw return-to-go tokens under source-target stitching;
- **Weighted Q-regularized Decision Transformer training** under the same fused source-target distribution.

> **Status.** This is a research codebase for the paper *Target-Aligned Fusion for Decision-Sequence Learning under Dynamics Shift*. The code still keeps some historical file names such as `QT`, `qt`, and `Q-value regularized Transformer`, because TAF-DT was developed by extending the QT implementation.

## Contents

- [Method Overview](#method-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Running Full Experiments](#running-full-experiments)
- [Configuration](#configuration)
- [Output and Analysis](#output-and-analysis)
- [Main Results](#main-results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Method Overview

TAF-DT follows a gate-then-weight fusion pipeline.

1. **Target-aligned fragment gate.** Source trajectory fragments are ranked by a latent-space MMD score against target fragments. Fragments with smaller MMD are retained because their state structure is closer to the target domain.
2. **Feasibility-aware transition weighting.** Retained source transitions are compared with target transitions through an OT cost. Lower-cost transitions receive larger Gibbs-style sampling weights.
3. **Fused training distribution.** Target data and weighted source data are mixed into one empirical training distribution.
4. **Advantage-conditioned tokenization.** Auxiliary value and Q functions are trained on the fused data. Their difference is used as an advantage token, replacing fragile return-to-go tokens in stitched source-target sequences.
5. **Weighted Q-regularized Transformer training.** The final Decision Transformer policy is trained with behavior-cloning, Q-regularization, and optional policy regularization, all under the same target-aligned fused sampler.

In the paper, the two data-side quantities controlled by this pipeline are the retained-fragment state-structure radius and the weighted source-to-target transport radius. In this codebase, these correspond operationally to the MMD filtering scores and OT feasibility scores computed before training.

## Repository Structure

```text
.
├── config/                         # YAML configs for environment and shift pairs
├── data/
│   └── target_dataset/             # Target-domain HDF5 datasets
├── decision_transformer/           # DT/QT/TAF-DT model and trainer modules
│   ├── evaluation/                 # Evaluation utilities
│   ├── misc/                       # Dataset, command, and helper utilities
│   ├── models/                     # Transformer, QL-DT, command network modules
│   └── training/                   # Sequence and Q-learning trainers
├── envs/                           # Modified MuJoCo environments and XML assets
├── filter_data.py                  # MMD and OT cost computation
├── filter_data_sas.py              # Alternative OT/MMD computation variant
├── experiment.py                   # Single TAF-DT training entry point
├── run_experiments.py              # Config-driven experiment launcher
├── run_ablation_experiments.py     # Ablation experiment launcher
├── analyze_training_progress.py    # Log parser and result summarizer
├── requirements.txt                # Python dependencies
└── README.md
```

## Installation

The code uses the D4RL / Gym / MuJoCo offline RL stack. A Linux machine with an NVIDIA GPU is recommended.

```bash
conda create -n tafdt python=3.8 -y
conda activate tafdt

pip install --upgrade pip
pip install -r requirements.txt
```

The main dependencies include `torch`, `gym==0.23.1`, `d4rl`, `mujoco_py`, `transformers<=4.8.1`, `pot`, `h5py`, `pyyaml`, `scikit-learn`, and `torch-geometric`.

If `mujoco_py` fails to build, please first make sure that MuJoCo, OpenGL, and compiler dependencies are correctly installed on your machine. The exact setup can vary across CUDA, driver, and Linux versions.

## Data Preparation

### Source datasets

Source datasets use the standard D4RL naming convention and are loaded through D4RL, for example:

- `halfcheetah-medium-v2`
- `hopper-medium-replay-v2`
- `walker2d-medium-expert-v2`
- `ant-medium-v2`

Supported source dataset types are:

```text
medium, medium-replay, medium-expert
```

### Target datasets

Target-domain datasets should be placed under:

```text
data/target_dataset/
```

The expected target dataset types are:

```text
medium, medium_expert, expert
```

For command-line arguments, use `medium_expert` with an underscore for target data. The runner converts it into target dataset names such as:

```text
hopper_gravity_0.5_medium_expert.hdf5
hopper_kinematic_medium_expert.hdf5
hopper_morph_medium_expert.hdf5
```

The code supports three dynamics-shift families:

| Shift type | Description | Example target name |
|---|---|---|
| `gravity` | modified gravity coefficient | `hopper_gravity_0.5_medium` |
| `kinematic` | constrained joint ranges | `hopper_kinematic_medium` |
| `morph` | modified body geometry / morphology | `hopper_morph_medium` |

> **GitHub note.** HDF5 datasets and saved model checkpoints can be large. For a public repository, consider storing them through Git LFS, a release asset, or an external download link instead of committing all generated files directly.

## Quick Start

### 1. Run a short smoke test

```bash
python run_experiments.py \
  --mode single \
  --env hopper \
  --variation gravity \
  --srctype medium \
  --tartype medium \
  --short \
  --cost-strategy isolate
```

The `--short` flag reduces the number of training iterations and is intended only for checking whether data loading, cost computation, and training can run end to end.

### 2. Run a single full experiment

```bash
python run_experiments.py \
  --mode single \
  --env hopper \
  --variation gravity \
  --srctype medium \
  --tartype medium \
  --cost-strategy isolate
```

This command first runs `filter_data.py` to compute MMD/OT costs, then runs `experiment.py` to train the TAF-DT policy.

### 3. Reuse existing cost files

```bash
python run_experiments.py \
  --mode single \
  --env hopper \
  --variation gravity \
  --srctype medium \
  --tartype medium \
  --cost-strategy reuse
```

The runner will reuse compatible cost files from `data/costs/` when available. If a compatible file is missing, it will recompute the costs.

## Running Full Experiments

The unified runner supports four modes.

### Single mode

Run one `(environment, shift, source type, target type)` setting.

```bash
python run_experiments.py \
  --mode single \
  --env ant \
  --variation morph \
  --srctype medium-expert \
  --tartype expert
```

### Env9 mode

Run all 9 source-target dataset combinations for one fixed environment and one fixed shift type.

```bash
python run_experiments.py \
  --mode env9 \
  --env hopper \
  --variation kinematic \
  --num-workers 3 \
  --cost-strategy reuse
```

This mode sweeps:

```text
source: medium, medium-replay, medium-expert
target: medium, medium_expert, expert
```

### Diff12 mode

Run the 12 environment-shift pairs using one fixed source type and one fixed target type.

```bash
python run_experiments.py \
  --mode diff12 \
  --srctype medium \
  --tartype medium \
  --num-workers 4 \
  --cost-strategy reuse
```

The 12 pairs are:

```text
halfcheetah: gravity, kinematic, morph
hopper:      gravity, kinematic, morph
walker2d:    gravity, kinematic, morph
ant:         gravity, kinematic, morph
```

### Full108 mode

Run the full grid of 108 experiments: 4 environments × 3 shifts × 3 source types × 3 target types.

```bash
python run_experiments.py \
  --mode full108 \
  --num-workers 4 \
  --cost-strategy reuse
```

When multiple workers are used, `filter_data.py` is protected by a process lock because OT/MMD preprocessing can be GPU-memory intensive. Training jobs can still run in parallel after cost computation.

## Configuration

Each environment-shift pair has a YAML file in `config/`, for example:

```text
config/hopper_gravity.yaml
config/hopper_kinematic.yaml
config/hopper_morph.yaml
```

A minimal relevant configuration block looks like this:

```yaml
runner:
  filter_data:
    seq_len: 20
    dir: data/costs_20

  experiment:
    seed: 123
    K: 20
    eta: 3.5
    max_iters: 500
    num_steps_per_iter: 1000
    lr_decay: true
    early_stop: true

    # TAF-DT fusion and training options
    proportion: 0.5
    ot_filter: true
    ot_proportion: 0.5
    use_weighted_qloss: true
    relabel_adv: true
    rtg_no_q: true
    pi_reg: true
    pi_reg_weight: 0.5
    training_normalization: true
    command_state_normalization: true
    expectile: 0.55
```

The runner checks that `runner.filter_data.seq_len` and `runner.experiment.K` are consistent, because the cost files and Transformer context length must match.

### Override parameters from the command line

Any config value can be overridden with dot notation:

```bash
python run_experiments.py \
  --mode single \
  --env hopper \
  --variation gravity \
  --srctype medium \
  --tartype medium \
  --override experiment.eta=5.0 \
  --override experiment.expectile=0.98 \
  --override experiment.ot_proportion=0.5
```

For matching MMD/OT sequence length and Transformer context length, override both fields together:

```bash
python run_experiments.py \
  --mode single \
  --env hopper \
  --variation morph \
  --srctype medium \
  --tartype medium \
  --override filter_data.seq_len=10 \
  --override experiment.K=10
```

### Cost strategies

| Strategy | Behavior |
|---|---|
| `isolate` | Default. Save newly computed costs under the current result directory. |
| `reuse` | Reuse compatible costs from `data/costs/`; recompute if missing or incompatible. |
| `regenerate` | Always recompute costs under `data/costs/`. |

## Running Individual Components

### Compute MMD/OT costs only

```bash
python filter_data.py \
  --env hopper \
  --srctype medium \
  --tartype gravity_0.5_medium \
  --seq_len 20 \
  --batch_size 10000 \
  --dir ./data/costs
```

Useful arguments include:

- `--ot-metric cosine`
- `--mmd-metric rbf`
- `--ot_method sinkhorn`
- `--ot_reg 0.1`
- `--mmd_step_type state`
- `--ot_step_type all`

### Train from precomputed costs

```bash
python experiment.py \
  --env hopper \
  --dataset medium \
  --tar_dataset gravity_0.5_medium \
  --K 20 \
  --eta 3.5 \
  --max_iters 500 \
  --num_steps_per_iter 1000 \
  --lr_decay \
  --early_stop \
  --k_rewards \
  --use_discount \
  --v_target \
  --use_mean_reduce \
  --relabel_adv \
  --rtg_no_q \
  --adv_scale 2.0 \
  --command_state_normalization \
  --adv_mean_reduce \
  --proportion 0.5 \
  --ot_filter \
  --ot_proportion 0.5 \
  --pi_reg \
  --pi_reg_weight 0.5 \
  --training_normalization \
  --rescale_reward \
  --use_weighted_qloss \
  --cost_dir ./data/costs
```

## Output and Analysis

By default, the runner creates a timestamped directory such as:

```text
save/results_YYYYMMDD_HHMMSS/
```

A run directory may contain:

- `run_experiments.log`: global launcher log;
- `run_experiments_args_*.yaml`: command-line argument snapshot;
- `config_snapshot_*.yaml`: exact config used by each experiment;
- individual experiment logs when running in parallel;
- `results_*.json`: success/failure summary and run duration;
- `costs/`: MMD/OT cost files when `--cost-strategy isolate` is used;
- model checkpoints and training outputs saved by `experiment.py`.

### Analyze training logs

```bash
python analyze_training_progress.py \
  --log-dir save/results_YYYYMMDD_HHMMSS \
  --output analysis_results.md
```

Analyze a single log file:

```bash
python analyze_training_progress.py \
  --file save/results_YYYYMMDD_HHMMSS/hopper-gravity-medium-to-medium-*.log
```

The analysis script extracts training duration, maximum return, maximum normalized score, and last-10-iteration statistics.

## Main Results

The paper evaluates TAF-DT on D4RL-style MuJoCo control tasks under morphology, kinematic, and gravity shifts. The compact summary is:

| Shift type | TAF-DT total | Best competing total | TAF-DT wins |
|---|---:|---:|---:|
| Morphology | 2078.2 | 1274.3 (OTDF) | 31 / 36 |
| Kinematic | 1900.6 | 1547.6 (OTDF) | 24 / 36 |
| Gravity | 1347.3 | 1160.7 (OTDF) | 19 / 36 |

These results are reported over source qualities `{medium, medium-replay, medium-expert}` and target qualities `{medium, medium-expert, expert}` across HalfCheetah, Hopper, Walker2d, and Ant. The paper also reports stitch-junction diagnostics showing that TAF-DT reduces action jumps, Q-value jumps, and local TD residuals around source-target stitch boundaries.

## Common Issues

### Cost file shape mismatch

If you see an error about cost file shape or `seq_len`, make sure these two config values match:

```yaml
runner:
  filter_data:
    seq_len: 20
  experiment:
    K: 20
```

If they do not match, regenerate the cost files.

### Target dataset not found

Check that the HDF5 file is under `data/target_dataset/` and follows the expected naming pattern, for example:

```text
hopper_gravity_0.5_medium.hdf5
hopper_kinematic_medium.hdf5
hopper_morph_medium.hdf5
```

### MuJoCo or D4RL import errors

Most environment issues come from `mujoco_py`, OpenGL, or old Gym/D4RL dependencies. Recheck your MuJoCo installation, GPU driver, and Python version. A clean conda environment is recommended.

## Citation

If this repository is useful for your research, please cite the paper once it is publicly available. A provisional BibTeX entry is:

```bibtex
@article{wang2026targetalignedfusion,
  title   = {Target-Aligned Fusion for Decision-Sequence Learning under Dynamics Shift},
  author  = {Wang, Guojian and Hon, Quinson and Chen, Xuyang and Zhao, Lin},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  note    = {Under review},
  year    = {2026}
}
```

This codebase also builds on QT:

```bibtex
@inproceedings{hu2024qvalueregularizedtransformer,
  title     = {Q-value Regularized Transformer for Offline Reinforcement Learning},
  author    = {Hu, Shengchao and Fan, Ziqing and Huang, Chaoqin and Shen, Li and Zhang, Ya and Wang, Yanfeng and Tao, Dacheng},
  booktitle = {International Conference on Machine Learning},
  year      = {2024}
}
```

## Acknowledgements

This repository benefits from prior implementations of Decision Transformer, Q-value Regularized Transformer, D4RL, and optimal-transport-based cross-domain offline RL. We thank the authors of these open-source projects for making their code available.

## License

This project is released under the Apache 2.0 License. See `LICENSE` for details.
