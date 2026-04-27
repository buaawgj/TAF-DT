# QT-off-dynamics: Q-value Regularized Transformer for Offline Dynamics Transfer

A research implementation that extends the Q-value regularized Transformer (QT) for offline reinforcement learning to handle **dynamics transfer** scenarios, where an agent trained on one environment must adapt to environmental variations with limited target data.

## Overview

This project adapts the QT method to address offline dynamics transfer by:

1. **Computing domain similarity costs** between source and target datasets using Optimal Transport (OT) and Maximum Mean Discrepancy (MMD)
2. **Training with weighted sampling** where source data points more similar to the target domain receive higher sampling weights
3. **Testing across multiple environments** and dynamics variations including gravity changes, kinematic modifications, and morphological alterations

## Quick Start

### Installation

```bash
# Install all dependencies (including data processing and visualization packages)
pip install -r requirements.txt
```

### Basic Usage

The project provides three execution modes through a unified runner:

```bash
# Run a single experiment
python run_experiments.py --mode single --env hopper --variation gravity --srctype medium --tartype medium

# Run core 9 experiments (all env-variation pairs with medium datasets)
python run_experiments.py --mode core9 --num-workers 3

# Run all 81 experiments (full grid)
python run_experiments.py --mode full81 --num-workers 3
```

### Short Testing Mode

Add `--short` flag for quick testing with reduced iterations:

```bash
python run_experiments.py --mode core9 --short --num-workers 3
```

## Project Structure

### Core Scripts

- **`run_experiments.py`** - Unified experiment runner with config-driven execution
- **`filter_data.py`** - Computes OT/MMD costs between source and target datasets  
- **`experiment.py`** - Trains the QT model with domain adaptation

### Configuration

- **`config/`** - YAML configurations for each environment-variation pair
  - Format: `{env}_{variation}.yaml` (e.g., `hopper_gravity.yaml`)
  - Contains parameters for both data filtering and model training

### Key Directories

- **`data/target_dataset/`** - Target environment datasets
- **`data/costs_*/`** - Computed OT/MMD cost files
- **`save/`** - Model checkpoints and experiment results

## Supported Environments

| Environment | Variations |
|-------------|------------|
| **HalfCheetah** | gravity, kinematic, morph |
| **Hopper** | gravity, kinematic, morph |
| **Walker2D** | gravity, kinematic, morph |

**Source Dataset Types (D4RL)**: `medium`, `medium-replay`, `medium-expert`  
**Target Dataset Types**: `medium`, `expert`, `medium_expert`

> **Note**: Source datasets come from D4RL and use the standard naming convention. Target datasets are custom variations with modified environment dynamics and use different naming (e.g., `medium_expert` instead of `medium-expert`).

## Configuration System

### Config File Structure

Each environment-variation pair has a YAML configuration file in `config/` with the following structure:

```yaml
# Environment-specific parameters (legacy SAC parameters, mostly unused in QT)
alpha: 0.2
batch_size: 128
actor_lr: 0.0003
critic_lr: 0.0003
gamma: 0.99
state_dim: 17
action_dim: 3
device: cuda
env: "Hopper-morph"

# Core QT configuration
runner:
  filter_data:
    seq_len: 5                   # Sequence length for cost computation (5 or 20)
    dir: "data/costs_5"          # Directory to save/load cost files
    
  experiment:
    # Core training parameters
    seed: 123                    # Random seed
    eta: 3.5                     # Q-value regularization weight
    max_iters: 500              # Maximum training iterations  
    num_steps_per_iter: 1000    # Steps per training iteration
    K: 5                        # Context length (matches seq_len)
    
    # Advanced training options
    lr_decay: true              # Enable learning rate decay
    early_stop: true            # Enable early stopping
    grad_norm: 5.0              # Gradient clipping norm
    adv_scale: 2.0              # Advantage scaling factor
    
    # Domain adaptation parameters
    proportion: 0.5             # Proportion of target data to use
    ot_filter: true             # Enable optimal transport filtering
    ot_proportion: 0.5          # Proportion for OT filtering
    pi_reg: true                # Enable policy regularization
    pi_reg_weight: 1.0          # Policy regularization weight (varies by env)
    
    # Training features
    use_discount: true          # Use discount factor
    v_target: true              # Use value targets
    k_rewards: true             # Use k-step rewards
    rtg_no_q: true              # Return-to-go without Q-values
    relabel_adv: true           # Relabel advantages
    command_state_normalization: true
    training_normalization: true
    rescale_reward: true
    use_mean_reduce: true
    use_full_pi_reg: true
    adv_mean_reduce: true
    
    # Paths
    exp_name: "qt"
    save_path: "./save/"

# Target environment configuration
tar_env_config:
  env_name: "Hopper-morph"
  # Optional environment-specific parameters
  env_targets: [3600, 1800]     # Performance targets (kinematic only)
  param:                        # Environment modifications (kinematic only)
    torso jnt lower limit: [0.001]
    foot jnt lower limit: [0.4] 
    foot jnt upper limit: [0.4]
```

### Parameter Overrides

Override any config parameter from command line using dot notation:

```bash
# Override core experiment parameters
python run_experiments.py --mode single --env hopper --variation gravity \
    --override experiment.eta=2.0 \
    --override experiment.max_iters=1000 \
    --override experiment.pi_reg_weight=0.8

# Override sequence length and context
python run_experiments.py --mode single --env hopper --variation morph \
    --override filter_data.seq_len=10 \
    --override experiment.K=10

# Multiple overrides for batch experiments
python run_experiments.py --mode core9 \
    --override experiment.proportion=0.3 \
    --override experiment.ot_proportion=0.3 \
    --override experiment.adv_scale=1.5
```

## Advanced Usage

### Running Individual Components

**Compute costs only:**
```bash
python filter_data.py --env hopper --srctype medium --tartype gravity_0.5_medium \
    --seq_len 20 --batch_size 10000
```

**Train model only (costs must exist):**
```bash
python experiment.py --env hopper --dataset medium --tar_dataset gravity_0.5_medium \
    --eta 3.5 --max_iters 500 --cost_dir ./data/costs_20
```

## Execution Modes

### Single Mode
```bash
python run_experiments.py --mode single --env hopper --variation gravity --srctype medium --tartype medium
```

### Core9 Mode  
Runs 9 core experiments (3 environments × 3 variations) with specified dataset types:
```bash
# Default: medium to medium
python run_experiments.py --mode core9

# Custom dataset types
python run_experiments.py --mode core9 --srctype medium-replay --tartype expert
```

### Full81 Mode
Runs complete grid (9 env-variation pairs × 3 source types × 3 target types):
```bash
python run_experiments.py --mode full81
```

## Command Line Options

The `run_experiments.py` script supports the following options:

- **`--mode`**: Execution mode (`single`, `core9`, `full81`) - **Required**
- **`--env`**: Environment name (e.g., `hopper`) - **Required for single mode**
- **`--variation`**: Environment variation (e.g., `gravity`) - **Required for single mode**  
- **`--srctype`**: Source dataset type (default: `medium`) - **Applies to all modes**
- **`--tartype`**: Target dataset type (default: `medium`) - **Applies to all modes**
- **`--short`**: Use short mode with reduced iterations for testing
- **`--override`**: Override config parameters (format: `section.key=value`)
- **`--output-dir`**: Specify directory for saving results (default: timestamped `save/results_*`)
- **`--num-workers`**: Number of parallel workers for `core9` and `full81` modes (default: 1)

## Output Management

Each experiment run generates a timestamped directory containing:
- **Configuration snapshots**: YAML files with the exact config used for each experiment
- **Individual log files**: Detailed logs for each experiment (when run in parallel)
- **Model checkpoints**: Saved model weights and training state
- **Results summary**: JSON file with experiment outcomes and statistics
- **Cost files**: Computed OT/MMD costs in `costs/` subdirectory

## Key Features

- **Config-driven execution** - All parameters managed through YAML configs
- **Parallel processing** - Multi-worker support for large experiment grids  
- **Cost computation** - OT/MMD-based domain similarity measurement
- **Weighted training** - Source data sampling based on target similarity
- **Comprehensive logging** - Detailed experiment tracking and results
- **Automated analysis** - Training progress extraction and statistical summarization

## Results Analysis

### Training Progress Analysis

The project includes a comprehensive analysis script for extracting and summarizing training progress from experiment logs:

**`analyze_training_progress.py`** - Automated training progress analysis

#### Usage

**Analyze all experiments in default results directory:**
```bash
python analyze_training_progress.py
```

**Analyze specific log file:**
```bash
python analyze_training_progress.py --file save/results_20250822_224227/hopper-morph-medium-to-medium-20250823_064807.log
```

**Analyze specific results directory:**
```bash
python analyze_training_progress.py --log-dir save/results_20250822_224227/
```

**Save output to markdown file:**
```bash
python analyze_training_progress.py --log-dir save/results_20250822_224227/ --output analysis_results.md
```

#### Output

The script generates a markdown table with comprehensive statistics for each experiment:

- **Experiment**: Name and configuration
- **Status**: Success/failure with completion info
- **Duration**: Total training time  
- **Start Time**: Experiment start timestamp
- **Max Return**: Highest achieved return value
- **Max Score**: Highest normalized D4RL score
- **Last 10 Iterations**: Statistics from final 10 training iterations
  - Return (Max/Avg/Std): Return value statistics
  - Score (Max/Avg/Std): Normalized score statistics

#### Example Output

```markdown
| Experiment | Start | Duration(h) | Status | Iters | Return_Max | Return_L10Max | Return_L10Avg | Return_L10Std | Score_Max | Score_L10Max | Score_L10Avg | Score_L10Std |
|------------|-------|-------------|---------|-------|------------|---------------|---------------|---------------|-----------|--------------|--------------|--------------|
| hc-grav-m2m | 08-22 22:42 | 8.1 | ✓ | 101 | 331.4 | -215.0 | -230.9 | 12.2 | 6.25 | 0.67 | 0.50 | 0.124 |
| hc-kin-m2m | 08-22 22:42 | 8.1 | ✓ | 101 | 2831.2 | 2810.1 | 2772.5 | 22.1 | 42.36 | 42.07 | 41.56 | 0.301 |
| hc-morph-m2m | 08-22 22:42 | 8.1 | ✓ | 101 | 4189.9 | 4101.5 | 3950.2 | 116.8 | 44.73 | 43.84 | 42.33 | 1.169 |
| hop-grav-m2m | 08-23 06:47 | 3.6 | ✓ | 101 | 3163.6 | 2143.9 | 1670.6 | 556.4 | 97.83 | 66.56 | 52.04 | 17.065 |
| hop-kin-m2m | 08-23 06:47 | 3.9 | ✓ | 101 | 1889.1 | 1855.3 | 1393.2 | 256.0 | 66.76 | 65.58 | 49.48 | 8.923 |
| hop-morph-m2m | 08-23 06:48 | 3.6 | ✓ | 101 | 2183.1 | 1896.1 | 1618.9 | 225.3 | 69.50 | 60.47 | 51.75 | 7.086 |
| walker2d-gravity-102706 | 08-23 10:27 | N/A | ✗ | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 0.00 | 0.000 |
| w2d-kin-m2m | 08-23 10:23 | N/A | ✗ | 75 | 2264.4 | 2199.4 | 1686.7 | 521.4 | 69.42 | 67.42 | 51.63 | 16.055 |
| w2d-morph-m2m | 08-23 10:43 | N/A | ✗ | 69 | 2621.8 | 2442.1 | 2004.9 | 295.9 | 59.51 | 55.42 | 45.46 | 6.742 |

```

## Algorithm Details

The method works in two stages:

1. **Cost Computation** (`filter_data.py`):
   - Computes Optimal Transport costs using Sinkhorn algorithm
   - Calculates Maximum Mean Discrepancy with RBF kernels
   - Generates cost files for weighted sampling during training

2. **Model Training** (`experiment.py`):  
   - Trains Decision Transformer with Q-value regularization
   - Uses computed costs to weight source data sampling
   - Adapts to target domain through similarity-based data filtering

## Citation

This work builds upon the Q-value Regularized Transformer:

```bibtex
@inproceedings{QT,
    title={Q-value Regularized Transformer for Offline Reinforcement Learning},
    author={Hu, Shengchao and Fan, Ziqing and Huang, Chaoqin and Shen, Li and Zhang, Ya and Wang, Yanfeng and Tao, Dacheng},
    booktitle={International Conference on Machine Learning},
    year={2024},
}
```

## License

Apache 2.0 License
