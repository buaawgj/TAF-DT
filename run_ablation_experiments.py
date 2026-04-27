#!/usr/bin/env python3
"""
Config-driven experiment runner for QT off-dynamics experiments.

This script provides a unified interface for running filter_data.py and experiment.py
with different execution modes: single, core12, full108, with optional short mode and
parameter overrides.
"""

import argparse
import json
import logging
import multiprocessing
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
import h5py
import numpy as np

from decision_transformer.misc.utils import _deep_update, _env_key_from_id, _join_middle


# Global lock for filter_data.py execution (multiprocessing-safe)
filter_data_lock = multiprocessing.Lock()


# Configure basic logging initially
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Main class for running QT experiments."""
    
    def __init__(self, output_dir: str = None):
        # Add timestamp to output directory if it's the default (None)
        if output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = f"./save/results_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler to also log to the results directory
        log_file = self.output_dir / 'run_experiments.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Define the 12 core experiments (4 environments x 3 variations)
        self.core_experiments = [
            ('halfcheetah', 'kinematic'),
            ('halfcheetah', 'gravity'),
            ('halfcheetah', 'morph'),
            ('hopper', 'kinematic'),
            ('hopper', 'gravity'),
            ('hopper', 'morph'),
            ('walker2d', 'kinematic'),
            ('walker2d', 'gravity'),
            ('walker2d', 'morph'),
            ('ant', 'kinematic'),
            ('ant', 'gravity'),
            ('ant', 'morph'),
        ]
        
        # Define source dataset types (from D4RL) and target dataset types (from repo)
        self.source_dataset_types = ['medium', 'medium-replay', 'medium-expert']
        self.target_dataset_types = ['medium', 'expert', 'medium_expert']
        
        # The advantage token ablation configurations
        self.ablation = [
            ('hopper', 'morph'),
            # ('walker2d', 'kinematic'),
            ('ant', 'gravity'),
            # ('halfcheetah', 'morph'),
        ]
        
        self.source_dataset_types = ['medium',]
        self.target_dataset_types = ['medium', 'medium_expert']
    
    def _run_with_streaming_output(self, cmd, logger, process_name, cwd):
        """Run subprocess with real-time output streaming to log file."""
        
        def stream_reader(pipe, q, stream_name):
            """Read from pipe and put lines in queue."""
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        q.put((stream_name, line.rstrip()))
                pipe.close()
            except Exception as e:
                q.put((stream_name, f"Error reading {stream_name}: {e}"))
        
        # Start process with pipes
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            cwd=cwd,
            bufsize=1,  # Line buffering
            universal_newlines=True
        )
        
        # Create queue for collecting output
        output_queue = queue.Queue()
        
        # Start threads to read stdout and stderr
        stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, output_queue, 'stdout'))
        stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, output_queue, 'stderr'))
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Process output in real-time
        while process.poll() is None or not output_queue.empty():
            try:
                stream_name, line = output_queue.get(timeout=0.1)
                if stream_name == 'stdout':
                    logger.info(f"[{process_name}] {line}")
                else:  # stderr
                    # Allow tqdm output to be printed
                    if line.strip() == '':
                        continue  # Skip blank lines only
                    logger.info(f"[{process_name}] {line}")
                
                # Force flush the log handlers
                for handler in logger.handlers:
                    handler.flush()
                    
            except queue.Empty:
                continue
        
        # Wait for threads to complete
        stdout_thread.join(timeout=1.0)
        stderr_thread.join(timeout=1.0)
        
        # Wait for process to complete and return exit code
        return process.wait()
    
    def load_config(self, env: str, variation: str) -> Dict[str, Any]:
        """Load configuration for a specific environment and variation."""
        config_file = f"config/{env}_{variation}.yaml"
        
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'runner' not in config:
            raise ValueError(f"Config file {config_file} missing 'runner' section")
        
        return config['runner']
    
    def load_config_for_env(self, env: str, variation: str, src_type: str, tar_type: str) -> dict:
        """Load configuration for a specific environment and variation."""
        src_type_key = _env_key_from_id(src_type)
        tar_type_key = _env_key_from_id(tar_type)
        
        base_path = f"config/{env}_{variation}.yaml"
        env_path = f"config/{env}_{variation}_{'_'.join(src_type_key)}_{'_'.join(tar_type_key)}.yaml"
        
        def _read_yaml(p):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Config file not found: {p}")
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

        base_cfg = _read_yaml(base_path)
        env_cfg = {}
        if os.path.exists(env_path):
            env_cfg  = _read_yaml(env_path)
        else:
            logger.debug(f"Environment-specific config file not found: {env_path}")

        cfg = _deep_update(base_cfg, env_cfg)
        logger.debug(f"Merged config: {cfg}")
        
        if 'runner' not in cfg:
            raise ValueError(f"Config file {base_path} or {env_path} missing 'runner' section")
        return cfg['runner']
    
    def apply_overrides(self, config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
        """Apply command-line parameter overrides to config."""
        for override in overrides:
            if '=' not in override:
                logger.warning(f"Invalid override format: {override}. Expected key=value")
                continue
            
            key_path, value = override.split('=', 1)
            
            # User provides section.key format (e.g., "experiment.learning_rate")
            # Since we already extracted config['runner'], we can use it directly
            keys = key_path.split('.')
            
            # Navigate to the target location in config
            target = config
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            
            # Convert value to appropriate type
            final_key = keys[-1]
            try:
                # Try to parse as JSON first (handles booleans, numbers, etc.)
                parsed_value = json.loads(value.lower() if value.lower() in ['true', 'false'] else value)
            except json.JSONDecodeError:
                # If JSON parsing fails, use as string
                parsed_value = value
            
            target[final_key] = parsed_value
            logger.info(f"Applied override: {key_path} = {parsed_value}")
        
        return config
    
    def apply_short_mode(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply short mode settings (reduced iterations for testing)."""
        if 'experiment' in config:
            config['experiment']['max_iters'] = 10
            config['experiment']['num_steps_per_iter'] = 100
            config['experiment']['early_stop'] = True
            config['experiment']['early_epoch'] = 2
            logger.info("Applied short mode: max_iters=10, num_steps_per_iter=100, " \
            "early_stop=True, early_epoch=2")
        return config
    
    def build_filter_data_command(self, env: str, variation: str, srctype: str, tartype: str, 
                                 config: Dict[str, Any], cost_directory: str) -> List[str]:
        """Build the command to run filter_data.py."""
        filter_config = config.get('filter_data', {})
        
        # Determine target dataset name with proper naming convention
        if variation == 'gravity':
            tar_dataset = f"{variation}_0.5_{tartype}"
        else:
            tar_dataset = f"{variation}_{tartype}"
        
        cmd = [
            'python', 'filter_data.py',
            '--env', env,
            '--srctype', srctype,
            '--tartype', tar_dataset,
        ]
        
        # Add filter_data specific parameters
        for key, value in filter_config.items():
            if key == 'dir':
                # Override the dir parameter to use our cost strategy directory
                cmd.extend(['--dir', cost_directory])
            elif key in ['seq_len', 'steps', 'batch_size']:
                cmd.extend([f'--{key}', str(value)])
            elif key in ['ot_metric', 'mmd_metric', 'ot_method']:
                cmd.extend([f'--{key}', str(value)])
            elif key in ['proportion', 'ot_reg']:
                cmd.extend([f'--{key}', str(value)])
        
        # If no 'dir' was specified in config, add our cost strategy directory
        if 'dir' not in filter_config:
            cmd.extend(['--dir', cost_directory])
        
        return cmd
    
    def build_experiment_command(self, env: str, variation: str, srctype: str, tartype: str, 
                               config: Dict[str, Any], cost_directory: str, save_dir: Path) -> List[str]:
        """Build the command to run experiment.py."""
        exp_config = config.get('experiment', {})
        
        # Determine target dataset name with proper naming convention
        if variation == 'gravity':
            tar_dataset = f"{variation}_0.5_{tartype}"
        else:
            tar_dataset = f"{variation}_{tartype}"
        
        cmd = [
            'python', 'experiment.py',
            '--env', env,
            '--dataset', srctype,
            '--tar_dataset', tar_dataset,
        ]
        
        # Add experiment specific parameters
        for key, value in exp_config.items():
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    cmd.append(f'--{key}')
            else:
                cmd.extend([f'--{key}', str(value)])
        
        # Add cost_dir parameter using provided cost_directory
        cmd.extend(['--cost_dir', cost_directory])
        
        # Add save_path parameter - let experiment.py create its own subdirectory within our output dir
        save_path_str = str(save_dir)
        if not save_path_str.endswith('/'):
            save_path_str += '/'
        cmd.extend(['--save_path', save_path_str])
        
        return cmd
    
    def check_cost_file_compatibility(self, cost_file_path: str, src_env: str, tar_env: str, 
                                    seq_len: int, max_len_param: int = 1000) -> Tuple[bool, str]:
        """
        Fast compatibility check for cost files without loading datasets.
        
        Uses the max_len formula: max(max_len_param, expected_max_traj_len + seq_len - 1)
        For standard D4RL datasets:
        - halfcheetah-*: max trajectory length ~1000
        - hopper-*: max trajectory length ~1000  
        - walker2d-*: max trajectory length ~1000
        
        Args:
            cost_file_path: Path to the HDF5 cost file
            src_env: Source environment name (e.g., 'halfcheetah-medium-expert-v2')
            tar_env: Target environment name (e.g., 'halfcheetah-gravity-0.5-expert-v2')
            seq_len: Sequence length parameter used in buffer creation
            max_len_param: Base max_len parameter (default 1000)
            
        Returns:
            (is_compatible, info_message)
        """
        try:
            # Check file existence
            if not os.path.exists(cost_file_path):
                return False, f"Cost file does not exist: {cost_file_path}"
            
            with h5py.File(cost_file_path, 'r') as f:
                # Check required datasets exist
                required_keys = ['ot_cost', 'mmd_cost']
                missing_keys = [key for key in required_keys if key not in f.keys()]
                if missing_keys:
                    return False, f"Missing required datasets: {missing_keys}"
                
                ot_cost = f['ot_cost']
                mmd_cost = f['mmd_cost']
                
                # Check shapes match between ot_cost and mmd_cost
                if ot_cost.shape != mmd_cost.shape:
                    return False, f"Shape mismatch: ot_cost {ot_cost.shape} != mmd_cost {mmd_cost.shape}"
                
                # Calculate expected max_len using TrajectoryBuffer formula:
                # max_len = max(max_len_param, traj_lens.max() + seq_len - 1)
                # For most D4RL datasets, max trajectory length is ~1000
                expected_max_traj_len = 1000
                expected_max_len = max(max_len_param, expected_max_traj_len + seq_len - 1)
                
                actual_shape = ot_cost.shape
                
                # Check max_len dimension (skip trajectory count check as it varies by dataset)
                if actual_shape[1] != expected_max_len:
                    return False, (f"Buffer max_len mismatch: got {actual_shape[1]}, "
                                 f"expected {expected_max_len} (max_len_param={max_len_param}, "
                                 f"seq_len={seq_len})")
                
                # Check data types
                if ot_cost.dtype != np.float32 or mmd_cost.dtype != np.float32:
                    return False, f"Wrong data types: ot_cost {ot_cost.dtype}, mmd_cost {mmd_cost.dtype} (expected float32)"
                
                # Quick sanity check: sample the end of trajectories where masking/padding effects are visible
                sample_size = min(100, actual_shape[0])
                # Check the last portion of trajectories where padding would occur
                last_cols = max(10, actual_shape[1] - 100)  # Last 100 columns or at least last 10
                ot_sample = ot_cost[:sample_size, last_cols:]
                mmd_sample = mmd_cost[:sample_size, last_cols:]
                
                if not np.isfinite(ot_sample).any() or not np.isfinite(mmd_sample).any():
                    return False, "Cost file appears to contain only NaN/infinite values"
                
                return True, (f"Compatible: shape {actual_shape}, expected max_len {expected_max_len}, "
                            f"seq_len={seq_len}, max_len_param={max_len_param}")
                
        except Exception as e:
            return False, f"Error reading cost file: {str(e)}"
    
    def get_cost_directory(self, cost_strategy: str) -> str:
        """Get the cost directory based on strategy."""
        if cost_strategy in ['reuse', 'regenerate']:
            return "data/new_costs"
        elif cost_strategy == 'isolate':
            return str(self.output_dir / "costs")
        else:
            raise ValueError(f"Unknown cost strategy: {cost_strategy}")
    
    def get_cost_file_path(self, env: str, variation: str, srctype: str, tartype: str, 
                          config: Dict[str, Any], cost_strategy: str) -> Optional[Path]:
        """
        Get the expected path for cost file based on experiment parameters.
        
        Returns None if the path cannot be determined.
        """
        try:
            cost_dir = self.get_cost_directory(cost_strategy)
            logger.info(f"Determined cost directory: {cost_dir}")
            if not cost_dir:
                return None
                
            # Build cost file name using the actual pattern from filter_data.py output
            # Pattern: {env}-srcdatatype-{srctype}-tardatatype-{variation}_{tartype}.hdf5
            variation = variation.replace('gravity', 'gravity_0.5')  # Adjust for gravity naming
            cost_file_name = f"{env}-srcdatatype-{srctype}-tardatatype-{variation}_{tartype}.hdf5"
            cost_file_path = Path(cost_dir) / cost_file_name
            logger.info(f"Determined cost file path: {cost_file_path}")
            return cost_file_path if cost_file_path.exists() else None
        except Exception:
            return None
    
    def run_single_experiment(self, env: str, variation: str, srctype: str, tartype: str, 
                            short_mode: bool = False, overrides: List[str] = None, 
                            individual_log: bool = False, cost_strategy: str = 'isolate') -> None:
        """Run a single experiment (filter_data + experiment)."""
        overrides = overrides or []
        
        # Create individual logger if requested (for parallel runs)
        if individual_log:
            datetime_str = time.strftime("%Y%m%d_%H%M%S")
            exp_name = f"{env}-{variation}-{srctype}-to-{tartype}-{datetime_str}"
            
            # Create individual logger for this experiment
            exp_logger = logging.getLogger(f"experiment_{exp_name}")
            exp_logger.setLevel(logging.INFO)
            
            # Remove any existing handlers to avoid duplicates
            for handler in exp_logger.handlers[:]:
                exp_logger.removeHandler(handler)
            
            # Add file handler for individual log
            log_file = self.output_dir / f'{exp_name}.log'
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            exp_logger.addHandler(file_handler)
            
            # Add console handler for parallel runs to enable streaming output
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(f'%(asctime)s - [{exp_name}] - %(levelname)s - %(message)s'))
            exp_logger.addHandler(console_handler)
            
            # Prevent propagation to parent logger to avoid duplicate logs
            exp_logger.propagate = False
        else:
            exp_logger = logger
            datetime_str = time.strftime("%Y%m%d_%H%M%S")
            exp_name = f"{env}-{variation}-{srctype}-to-{tartype}-{datetime_str}"
            
            # For single runs, temporarily update the console handler format to include experiment name
            for handler in exp_logger.handlers:
                if isinstance(handler, logging.StreamHandler) and handler.stream.name == '<stderr>':
                    handler.setFormatter(logging.Formatter(f'%(asctime)s - [{exp_name}] - %(levelname)s - %(message)s'))
        
        try:
            # Load and process config
            config = self.load_config_for_env(env, variation, srctype, tartype)
            if short_mode:
                config = self.apply_short_mode(config)
            config = self.apply_overrides(config, overrides)
            
            # Validate parameter consistency between filter_data and experiment
            filter_seq_len = config.get('filter_data', {}).get('seq_len', 20)  # Default from filter_data.py
            experiment_K = config.get('experiment', {}).get('K', 20)  # Default from experiment.py
            
            if filter_seq_len != experiment_K:
                error_msg = (f"Parameter mismatch: filter_data.seq_len={filter_seq_len} != "
                           f"experiment.K={experiment_K}. These must be equal for compatibility.")
                exp_logger.error(error_msg)
                raise ValueError(error_msg)
            
            exp_logger.info(f"Parameter validation passed: seq_len=K={filter_seq_len}")
            
            # Save config snapshot directly in the main results directory
            config_snapshot_file = self.output_dir / f'config_snapshot_{exp_name}.yaml'
            with open(config_snapshot_file, 'w') as f:
                yaml.dump({'runner': config}, f, default_flow_style=False)
            
            exp_logger.info(f"Starting experiment: {exp_name}")
            start_time = time.time()
            
            # Cost strategy handling - determine if we need to run filter_data
            # Use the cost_strategy parameter passed to the method
            cost_directory = self.get_cost_directory(cost_strategy)  # Compute once for consistency
            skip_filter_data = False
            
            exp_logger.info(f"Using cost strategy: {cost_strategy}")
            exp_logger.info(f"Cost directory: {cost_directory}")
            
            if cost_strategy == 'regenerate':
                exp_logger.info("REGENERATE strategy: Will always run filter_data.py")
                
            elif cost_strategy == 'isolate':
                exp_logger.info("ISOLATE strategy: Will generate costs in experiment-specific directory (always runs filter_data.py)")
                
            elif cost_strategy == 'reuse':
                # Check if compatible cost file exists for reuse
                cost_file_path = self.get_cost_file_path(env, variation, srctype, tartype, config, cost_strategy)
                # exp_logger.info(f"REUSE strategy: Checking for existing cost file at {cost_file_path}")
                if not cost_file_path or not cost_file_path.exists():
                    exp_logger.info("REUSE strategy: No existing cost file found, will generate costs")
                else:
                    exp_logger.info(f"REUSE strategy: Found existing cost file: {cost_file_path}")
                    
                    # Check compatibility
                    # NOTE: experiment.py uses variant['K'] as seq_len, so we must use the same for compatibility check
                    seq_len = config.get('experiment', {}).get('K', 20)  # Use experiment.K, which is the actual seq_len used in experiment.py
                    src_env = f"{env}-{srctype}-v2"
                    tar_env = f"{env}-{variation}-{tartype}-v2"
                    
                    is_compatible, message = self.check_cost_file_compatibility(
                        str(cost_file_path), src_env, tar_env, seq_len
                    )
                    
                    if is_compatible:
                        exp_logger.info(f"REUSE strategy: Cost file compatible - {message}")
                        exp_logger.info("REUSE strategy: Skipping filter_data.py execution")
                        skip_filter_data = True
                    else:
                        exp_logger.warning(f"REUSE strategy: Cost file incompatible - {message}")
                        exp_logger.info("REUSE strategy: Will regenerate costs")
                        
            else:
                exp_logger.warning(f"Unknown cost_strategy '{cost_strategy}', defaulting to regenerate")
            
            # Execute filter_data.py if needed
            if not skip_filter_data:
                # CRITICAL SECTION: Only one filter_data.py can run at a time due to GPU memory limitations
                exp_logger.info(f"Waiting for filter_data lock...")
                with filter_data_lock:
                    exp_logger.info(f"Acquired filter_data lock for {exp_name}")
                    
                    # Run filter_data.py
                    filter_cmd = self.build_filter_data_command(env, variation, srctype, tartype, config, cost_directory)
                    exp_logger.info(f"Running filter_data: {' '.join(filter_cmd)}")
                    
                    # Run filter_data with real-time output streaming
                    filter_result = self._run_with_streaming_output(
                        filter_cmd, exp_logger, f"filter_data.py", str(Path.cwd())
                    )
                    
                    exp_logger.info(f"Released filter_data lock for {exp_name}")
                
                # Lock is released - experiment.py can run in parallel
                
                if filter_result != 0:
                    exp_logger.error(f"filter_data.py failed for {exp_name} with return code {filter_result}")
                    
                    # Restore original formatter for single runs
                    if not individual_log:
                        for handler in exp_logger.handlers:
                            if isinstance(handler, logging.StreamHandler) and handler.stream.name == '<stderr>':
                                handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                    
                    return {
                        'experiment': exp_name,
                        'status': 'failed',
                        'stage': 'filter_data',
                        'error': f'filter_data.py failed with return code {filter_result}',
                        'duration': time.time() - start_time
                    }
            else:
                exp_logger.info("Skipped filter_data.py execution - using existing compatible cost file")
            
            # Run experiment.py
            exp_cmd = self.build_experiment_command(env, variation, srctype, tartype, config, cost_directory, self.output_dir)
            exp_logger.info(f"Running experiment: {' '.join(exp_cmd)}")
            
            # Run experiment with real-time output streaming
            exp_result = self._run_with_streaming_output(
                exp_cmd, exp_logger, f"experiment.py", str(Path.cwd())
            )
            
            if exp_result != 0:
                exp_logger.error(f"experiment.py failed for {exp_name} with return code {exp_result}")
                
                # Restore original formatter for single runs
                if not individual_log:
                    for handler in exp_logger.handlers:
                        if isinstance(handler, logging.StreamHandler) and handler.stream.name == '<stderr>':
                            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                
                return {
                    'experiment': exp_name,
                    'status': 'failed',
                    'stage': 'experiment',
                    'error': f'experiment.py failed with return code {exp_result}',
                    'duration': time.time() - start_time
                }
            
            duration = time.time() - start_time
            exp_logger.info(f"Completed experiment: {exp_name} in {duration:.2f}s")
            
            # Restore original formatter for single runs
            if not individual_log:
                for handler in exp_logger.handlers:
                    if isinstance(handler, logging.StreamHandler) and handler.stream.name == '<stderr>':
                        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            
            return {
                'experiment': exp_name,
                'status': 'success',
                'duration': duration
            }
            
        except Exception as e:
            exp_logger.error(f"Exception in experiment {exp_name}: {str(e)}")
            
            # Restore original formatter for single runs in case of exception
            if not individual_log:
                for handler in exp_logger.handlers:
                    if isinstance(handler, logging.StreamHandler) and handler.stream.name == '<stderr>':
                        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            
            return {
                'experiment': exp_name,
                'status': 'failed',
                'stage': 'setup',
                'error': str(e),
                'duration': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def run_experiments_parallel(self, experiments: List[Tuple[str, str, str, str]], 
                                num_workers: int, short_mode: bool = False, 
                                overrides: List[str] = None, cost_strategy: str = 'isolate') -> List[Dict[str, Any]]:
        """Run multiple experiments in parallel."""
        results = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all experiments with individual logging enabled
            future_to_exp = {
                executor.submit(
                    self.run_single_experiment, 
                    env, variation, srctype, tartype, short_mode, overrides, True, cost_strategy
                ): (env, variation, srctype, tartype)
                for env, variation, srctype, tartype in experiments
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_exp):
                env, variation, srctype, tartype = future_to_exp[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    status = result['status']
                    duration = result['duration']
                    logger.info(f"Experiment {env}-{variation}-{srctype}-to-{tartype}: {status} ({duration:.2f}s)")
                    
                except Exception as e:
                    logger.error(f"Exception in parallel experiment {env}-{variation}-{srctype}-to-{tartype}: {str(e)}")
                    results.append({
                        'experiment': f"{env}-{variation}-{srctype}-to-{tartype}",
                        'status': 'failed',
                        'stage': 'parallel_execution',
                        'error': str(e),
                        'duration': 0
                    })
        
        return results
    
    def generate_experiments(self, mode, env=None, variation=None, srctype=None, tartype=None,) -> List[Tuple[str, str, str, str]]:
        if mode == 'single' or mode == 'single_ablation':
            return [(env, variation, srctype, tartype)]
        elif mode == 'env9':
            return [
                (env, variation, srctype, tartype)
                for srctype in self.source_dataset_types
                for tartype in self.target_dataset_types
            ]
        elif mode == 'diff12':
            return [
                (env, variation, srctype, tartype)
                for env, variation in self.core_experiments
            ]
        elif mode == 'full108':
            return [
                (env, variation, srctype, tartype)
                for env, variation in self.core_experiments
                for srctype in self.source_dataset_types
                for tartype in self.target_dataset_types
            ]
        elif mode == 'ablation':
            return [
                (env, variation, srctype, tartype)
                for env, variation in self.ablation
                for srctype in self.source_dataset_types
                for tartype in self.target_dataset_types
            ]
            
    def save_results(self, results: List[Dict[str, Any]], mode: str):
        """Save experiment results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"results_{mode}_{timestamp}.json"
        
        summary = {
            'mode': mode,
            'timestamp': timestamp,
            'total_experiments': len(results),
            'successful': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'failed']),
            'total_duration': sum(r['duration'] for r in results),
            'results': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Summary: {summary['successful']}/{summary['total_experiments']} successful, "
                   f"total duration: {summary['total_duration']:.2f}s")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Run QT off-dynamics experiments")
    parser.add_argument('--mode', required=True, choices=['single', 'env9', 'diff12', 'full108'],
                       help='Execution mode')
    parser.add_argument('--env', help='Environment name (required for single mode)')
    parser.add_argument('--variation', help='Environment variation (required for single mode)')
    parser.add_argument('--srctype', help='Source dataset type')
    parser.add_argument('--tartype', help='Target dataset type')
    parser.add_argument('--short', action='store_true', help='Use short mode (reduced iterations)')
    parser.add_argument('--override', action='append', default=[], 
                       help='Override config parameters (format: section.key=value, e.g., experiment.learning_rate=1e-4)')
    parser.add_argument('--output-dir', default=None, help='Output directory for results')
    parser.add_argument('--num-workers', type=int, default=1, 
                       help='Number of parallel workers (for core12 and full108 modes)')
    parser.add_argument('--cost-strategy', choices=['reuse', 'regenerate', 'isolate'], default='isolate',
                       help='Cost strategy for experiments')

    args = parser.parse_args()
    
    # Create runner first to set up file logging
    runner = ExperimentRunner(args.output_dir)
    
    # Log the cost strategy (now that file handler is set up)
    logger.info(f"Cost strategy: {args.cost_strategy}")
    
    # Validate arguments based on mode
    if args.mode == 'single' or 'single_ablation':
        if not args.env or not args.variation or not args.srctype or not args.tartype:
            logger.error("--env, --variation, --srctype and --tartype are required for single mode")
            sys.exit(1)
    elif args.mode == 'env9':
        if not args.env or not args.variation:
            logger.error("--env and --variation are required for env9 mode")
            sys.exit(1)
        if args.srctype != 'medium' or args.tartype != 'medium':
            logger.warning("Source and target types are ignored in env9 mode, using all combinations")
    elif args.mode == 'diff12':
        if not args.srctype or not args.tartype:
            logger.error("--srctype and --tartype are required for diff12 mode")
            sys.exit(1)  
        if args.env or args.variation:
            logger.warning("Environment and variation are ignored in diff12 mode")
    elif args.mode == 'full108':
        if args.env or args.variation or args.srctype != 'medium' or args.tartype != 'medium':
            logger.warning("All parameters except mode are ignored in full108 mode")
    
    # Save run_experiments.py arguments to a separate YAML file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_args = {
        'mode': args.mode,
        'env': args.env,
        'variation': args.variation,
        'srctype': args.srctype,
        'tartype': args.tartype,
        'short_mode': args.short,
        'overrides': args.override,
        'output_dir': args.output_dir,
        'num_workers': args.num_workers,
        'cost_strategy': args.cost_strategy,
        'timestamp': timestamp
    }
    run_args_file = runner.output_dir / f'run_experiments_args_{timestamp}.yaml'
    with open(run_args_file, 'w') as f:
        yaml.dump(run_args, f, default_flow_style=False)
    logger.info(f"Run arguments saved to: {run_args_file}")
    
    experiments = runner.generate_experiments(args.mode, args.env, args.variation, args.srctype, args.tartype)

    logger.info(f"Running {args.mode} experiments with {len(experiments)} variations")


    try:

        results = runner.run_experiments_parallel(
            experiments, 
            num_workers=args.num_workers if args.mode != 'single' else 1, 
            short_mode=args.short, 
            overrides=args.override,
            cost_strategy=args.cost_strategy
        )

        # Save results
        summary = runner.save_results(results, args.mode)
        
        # Exit with appropriate code
        if summary['failed'] > 0:
            logger.error(f"{summary['failed']} experiments failed")
            sys.exit(1)
        else:
            logger.info("All experiments completed successfully")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
        
def ablation_main():
    parser = argparse.ArgumentParser(description="Run QT off-dynamics experiments")
    parser.add_argument('--mode', required=True, choices=['single', 'env9', 'diff12', 'full108', 'ablation', 'single_ablation'], help='Execution mode')
    parser.add_argument('--env', help='Environment name (required for single mode)')
    parser.add_argument('--variation', help='Environment variation (required for single mode)')
    parser.add_argument('--srctype', help='Source dataset type',)
    parser.add_argument('--tartype', help='Target dataset type',)
    parser.add_argument('--short', action='store_true', help='Use short mode (reduced iterations)')
    parser.add_argument('--override', action='append', default=[], 
                       help='Override config parameters (format: section.key=value, e.g., experiment.learning_rate=1e-4)')
    parser.add_argument('--output-dir', default=None, help='Output directory for results')
    parser.add_argument('--num-workers', type=int, default=1, 
                       help='Number of parallel workers (for core12 and full108 modes)')
    parser.add_argument('--cost-strategy', choices=['reuse', 'regenerate', 'isolate'], default='isolate', help='Cost strategy for experiments')

    args = parser.parse_args()
    
    # Create runner first to set up file logging
    runner = ExperimentRunner(args.output_dir)
    
    # Log the cost strategy (now that file handler is set up)
    logger.info(f"Cost strategy: {args.cost_strategy}")
    
    # Validate arguments based on mode
    if args.mode == 'single':
        if not args.env or not args.variation or not args.srctype or not args.tartype:
            logger.error("--env, --variation, --srctype and --tartype are required for single mode")
            sys.exit(1)
    elif args.mode == 'env9':
        if not args.env or not args.variation:
            logger.error("--env and --variation are required for env9 mode")
            sys.exit(1)
        if args.srctype != 'medium' or args.tartype != 'medium':
            logger.warning("Source and target types are ignored in env9 mode, using all combinations")
    elif args.mode == 'diff12':
        if not args.srctype or not args.tartype:
            logger.error("--srctype and --tartype are required for diff12 mode")
            sys.exit(1)  
        if args.env or args.variation:
            logger.warning("Environment and variation are ignored in diff12 mode")
    elif args.mode == 'full108':
        if args.env or args.variation or args.srctype != 'medium' or args.tartype != 'medium':
            logger.warning("All parameters except mode are ignored in full108 mode")
    elif args.mode == 'ablation':
        if args.env or args.variation or args.srctype != None or args.tartype != None:
            logger.warning("All parameters except mode are ignored in ablation mode")
    
    # Save run_experiments.py arguments to a separate YAML file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_args = {
        'mode': args.mode,
        'env': args.env,
        'variation': args.variation,
        'srctype': args.srctype,
        'tartype': args.tartype,
        'short_mode': args.short,
        'overrides': args.override,
        'output_dir': args.output_dir,
        'num_workers': args.num_workers,
        'cost_strategy': args.cost_strategy,
        'timestamp': timestamp
    }
    run_args_file = runner.output_dir / f'run_experiments_args_{timestamp}.yaml'
    with open(run_args_file, 'w') as f:
        yaml.dump(run_args, f, default_flow_style=False)
    logger.info(f"Run arguments saved to: {run_args_file}")
    
    experiments = runner.generate_experiments(args.mode, args.env, args.variation, args.srctype, args.tartype)
    print('experiments:', experiments)

    logger.info(f"Running {args.mode} experiments with {len(experiments)} variations")

    try:
        results = runner.run_experiments_parallel(
            experiments, 
            num_workers=args.num_workers if args.mode != 'single' else 1, 
            short_mode=args.short, 
            overrides=args.override,
            cost_strategy=args.cost_strategy
        )

        # Save results
        summary = runner.save_results(results, args.mode)
        
        # Exit with appropriate code
        if summary['failed'] > 0:
            logger.error(f"{summary['failed']} experiments failed")
            sys.exit(1)
        else:
            logger.info("All experiments completed successfully")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    ablation_main()
