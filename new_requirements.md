# Streamlined Requirements: Config-Driven Experiment Runner

## Objective

Create a simple, config-driven experiment runner to enable flexible and reproducible tests across different environment and variation pairs. This runner will streamline the process of launching experiments by leveraging a unified configuration system that mirrors the arguments and APIs used in `filter_data.py` and `experiment.py`.

## Key Features

- **Unified Configuration**: A single YAML configuration file for each `(environment, variation)` pair, containing all parameters for both data filtering and model training.
- **Flexible Execution Modes**: The runner script will support multiple execution modes:
  - **Single Run**: Execute a single experiment for a specified environment and variation.
  - **Core 9 Runs**: Execute the 9 core experiments, representing all `(environment, variation)` pairs with `medium` source and `medium` target datasets.
  - **Full 81 Runs**: Execute the full grid of 81 experiments, covering all combinations of environments, variations, source datasets, and target datasets.
- **Short Run Option**: A `--short` flag can be applied to any of the above modes to run them with a reduced number of iterations (`max_iters`) and steps per iteration (`num_steps_per_iter`), allowing for rapid integration testing.
- **Parameter Overrides**: Allow for overriding specific configuration parameters via the command line for quick iteration, while maintaining the config-first approach.
- **Concurrency**: Support for running multiple experiments in parallel to speed up large-scale tests for `core9` and `full81` modes.

## Configuration

- **Config Files**: All configuration files will be located in the `config/` directory. The naming convention will be `config/<env_name>_<variation>.yaml` (e.g., `config/hopper_gravity.yaml`). If there is no variation, the file will be named `config/<env_name>.yaml`.
- **Config Structure**: Each config file will contain a top-level `runner` key, with `filter_data` and `experiment` sections nested inside. This isolates runner-specific parameters and avoids conflicts with other scripts.

Example `config/hopper_gravity.yaml`:
```yaml
runner:
  filter_data:
    ot_metric: 'cosine'
    mmd_metric: 'rbf'
    steps: 100000
    proportion: 1.0
    seq_len: 20
    ot_method: 'sinkhorn'
    ot_reg: 0.1
    batch_size: 10000

  experiment:
    K: 20
    batch_size: 128
    embed_dim: 256
    n_layer: 4
    n_head: 4
    learning_rate: 3e-4
    weight_decay: 1e-4
    max_iters: 500
    num_steps_per_iter: 1000
    # ... other experiment parameters
```

## Execution

- **Runner Script**: A single Python script, e.g., `run_experiments.py`, will be the entry point for all experiments.
- **Cost File Naming**: The script will adhere to the existing cost file naming convention: `<env>-srcdatatype-<srctype>-tardatatype-<tartype>.hdf5`.
- **Output Management**: Each run will generate a directory containing:
  - A snapshot of the configuration used for the run.
  - Saved model checkpoints.
  - Logs and evaluation metrics.

## Script Interface

The runner script will have a command-line interface similar to the following:

```bash
python run_experiments.py --mode <single|core9|full81> [options]
```

- **`--mode`**: Specifies the execution mode.
  - `single`: Requires `--env`, `--variation`, `--srctype`, and `--tartype`.
  - `core9`: Runs the 9 core experiments.
  - `full81`: Runs all 81 experiments.
- **`--short`**: If specified, runs the selected mode with reduced iterations for quick testing.
- **`--env`**: The base environment name (e.g., `hopper`).
- **`--variation`**: The environment variation (e.g., `gravity`).
- **`--srctype`**: The source dataset type (e.g., `medium`).
- **`--tartype`**: The target dataset type (e.g., `medium`).
- **`--override`**: Override a config parameter. This is useful for quick experiments without modifying config files. For example: `--override runner.experiment.learning_rate=1e-4` or `--override runner.filter_data.proportion=0.5`.
- **`--output-dir`**: Specifies the directory for saving results.
- **`--num-workers`**: Number of parallel processes for concurrent execution (optional).

## Example Usage

- **Single Run**:
  ```bash
  python run_experiments.py --mode single --env hopper --variation gravity --srctype medium --tartype medium
  ```
- **Core 9 Runs with Concurrency**:
  ```bash
  python run_experiments.py --mode core9 --num-workers 3
  ```
- **Full 81 Runs with Concurrency**:
  ```bash
  python run_experiments.py --mode full81 --num-workers 3
  ```
- **Short Run (Single)**:
  ```bash
  python run_experiments.py --mode single --short --env hopper --variation gravity --srctype medium --tartype medium
  ```
- **Short Run (Core 9)**:
  ```bash
  python run_experiments.py --mode core9 --short --num-workers 3
  ```
- **Overriding Proportion**:
  ```bash
  python run_experiments.py --mode single --env walker2d --srctype medium --tartype medium --override runner.filter_data.proportion=0.1
  ```
