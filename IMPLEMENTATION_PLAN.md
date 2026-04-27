# QT Experiment Runner Enhancement Plan

## Overview

This plan implements intelligent cost file management for the QT experiment runner, providing three distinct strategies for handling cost generation and reuse.

## Problem Statement

Currently, the experiment runner always generates costs in the output directory and lacks compatibility checking. This leads to:
- Unnecessary cost regeneration when compatible files exist
- No reuse of expensive cost computations across experiments
- No flexibility in cost file management strategies

## Solution: Cost Strategy System

Implement a single `--cost-strategy` argument with three mutually exclusive modes:

1. **`reuse`** (default): Smart reuse with compatibility checking
2. **`regenerate`**: Force regeneration in global directory
3. **`isolate`**: Generate in experiment-specific directory (current behavior)

## Implementation Tasks

### Task 1: Add Cost Strategy Argument ✅ COMPLETE

**File**: `run_experiments.py`
**Location**: `main()` function argument parser

**Changes**:
- Add cost strategy argument to parser:
```python
parser.add_argument('--cost-strategy', 
                   choices=['reuse', 'regenerate', 'isolate'], 
                   default='reuse',
                   help='Cost generation strategy: reuse (default, use existing if compatible), regenerate (always regenerate in data/costs), isolate (generate in output directory)')
```

**Logging**: 
- `"Cost strategy: {args.cost_strategy}"`

**Status**: ✅ **COMPLETE** - Already implemented in current code

### Task 2: Update Method Signatures and Parameter Passing

**File**: `run_experiments.py`

**Priority**: **HIGH** - Must be done first to establish the foundation

**Add cost_strategy parameter to methods**:

1. **Update `build_filter_data_command()` signature**:
   ```python
   def build_filter_data_command(self, env: str, variation: str, srctype: str, tartype: str, 
                                config: Dict[str, Any], cost_strategy: str = 'isolate') -> List[str]:
   ```

2. **Update `build_experiment_command()` signature**:
   ```python  
   def build_experiment_command(self, env: str, variation: str, srctype: str, tartype: str, 
                               config: Dict[str, Any], save_dir: Path, cost_strategy: str = 'isolate') -> List[str]:
   ```

3. **Update `run_single_experiment()` signature**:
   ```python
   def run_single_experiment(self, env: str, variation: str, srctype: str, tartype: str, 
                            short_mode: bool = False, overrides: List[str] = None, 
                            individual_log: bool = False, cost_strategy: str = 'isolate') -> Dict[str, Any]:
   ```

4. **Update `run_experiments_parallel()` signature**:
   ```python
   def run_experiments_parallel(self, experiments: List[Tuple[str, str, str, str]], 
                               num_workers: int, short_mode: bool = False, 
                               overrides: List[str] = None, cost_strategy: str = 'isolate') -> List[Dict[str, Any]]:
   ```

**Update all call sites**:
- In `run_single_experiment()`: Pass `cost_strategy` to `build_filter_data_command()` and `build_experiment_command()`
- In `run_experiments_parallel()`: Pass `cost_strategy` to `run_single_experiment()` in executor.submit()
- In `main()`: Pass `args.cost_strategy` to `run_experiments_parallel()`

**Add missing validation in main()**:
```python
# Validate arguments based on mode (move validation from generate_experiments to main)
if args.mode == 'single':
    if not args.env or not args.variation:
        logger.error("--env and --variation are required for single mode")
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
```

### Task 3: Add Helper Method and Basic Cost Directory Logic

**File**: `run_experiments.py`
**Location**: Add new method to `ExperimentRunner` class

**Add helper method**:
```python
def get_cost_directory(self, cost_strategy: str) -> Path:
    """Get the cost directory based on strategy."""
    if cost_strategy in ['reuse', 'regenerate']:
        return Path("data/costs")
    elif cost_strategy == 'isolate':
        return self.output_dir / "costs"
    else:
        raise ValueError(f"Unknown cost strategy: {cost_strategy}")
```

**Update cost directory logic in build methods**:
- **In `build_filter_data_command()`**: Replace hardcoded cost directory logic with `self.get_cost_directory(cost_strategy)`
- **In `build_experiment_command()`**: Replace hardcoded cost directory logic with `self.get_cost_directory(cost_strategy)`

**Add logging**:
- `"Using cost directory for filter_data: {cost_dir} (strategy: {strategy})"`
- `"Setting experiment cost_dir to: {cost_dir} (strategy: {strategy})"`

### Task 4: Implement Cost Compatibility Checking

**File**: `run_experiments.py`
**Location**: Add new method to `ExperimentRunner` class

**Method**: `check_cost_file_compatibility(self, cost_path, env, variation, srctype, tartype, seq_len)`

**Compatibility Checks**:
1. File exists at expected path
2. Valid HDF5 format (can load with h5py)
3. Contains required keys: `ot_cost` and `mmd_cost`
4. Arrays have valid 2D shapes with positive dimensions
5. Can be reshaped to expected buffer dimensions `(num_trajectories, max_len, 1)`

**Return**: `(is_compatible: bool, reason: str)`

### Task 4: Implement Cost Compatibility Checking

**File**: `run_experiments.py`
**Location**: Add new method to `ExperimentRunner` class

**Add required import**:
```python
import h5py
```

**Method**: `check_cost_file_compatibility(self, cost_path, env, variation, srctype, tartype, seq_len)`

**Compatibility Checks**:
1. File exists at expected path
2. Valid HDF5 format (can load with h5py)
3. Contains required keys: `ot_cost` and `mmd_cost`
4. Arrays have valid 2D shapes with positive dimensions
5. Can be reshaped to expected buffer dimensions `(num_trajectories, max_len, 1)`

**Return**: `(is_compatible: bool, reason: str)`

**Logging for each check**:
```
"Checking cost file compatibility: {cost_path}"
"✓ Cost file exists" / "✗ Cost file does not exist"
"✓ Cost file is valid HDF5" / "✗ Cost file is corrupted HDF5: {error}"
"✓ Cost file has required keys: ot_cost, mmd_cost" / "✗ Missing keys: {missing_keys}"
"✓ Cost arrays have valid shapes: ot_cost{shape1}, mmd_cost{shape2}" / "✗ Invalid shapes: {details}"
"✓ Cost arrays can be reshaped to buffer dimensions" / "✗ Reshape failed: {error}"
"Cost file compatibility: COMPATIBLE" / "Cost file compatibility: INCOMPATIBLE - {reason}"
```

### Task 5: Integrate Strategy Logic into Experiment Execution

**File**: `run_experiments.py`
**Method**: `run_single_experiment()`

**Current Flow**:
1. Load config and apply overrides
2. **Always** run filter_data.py (within critical section)
3. Run experiment.py

**New Flow**:
1. Load config and apply overrides
2. **NEW**: Determine cost strategy and check compatibility
3. **Conditionally** run filter_data.py based on strategy decision
4. Run experiment.py

**Integration Point**: 
Insert cost strategy logic **after config processing** but **before the filter_data critical section**

**Specific Location in `run_single_experiment()`**:
Insert new logic after:
```python
# Save config snapshot directly in the main results directory
config_snapshot_file = self.output_dir / f'config_snapshot_{exp_name}.yaml'
with open(config_snapshot_file, 'w') as f:
    yaml.dump({'runner': config}, f, default_flow_style=False)

exp_logger.info(f"Starting experiment: {exp_name}")
start_time = time.time()
```

And before:
```python
# CRITICAL SECTION: Only one filter_data.py can run at a time due to GPU memory limitations
exp_logger.info(f"Waiting for filter_data lock...")
with filter_data_lock:
```

**New Code Block to Insert**:
```python
# Determine cost strategy and check if filter_data should be skipped
exp_logger.info("Checking if cost generation can be skipped...")

# Get cost directory and file path based on strategy
cost_dir = self.get_cost_directory(cost_strategy)
cost_filename = f"{env}-srcdatatype-{srctype}-tardatatype-{tartype}.hdf5"
cost_path = cost_dir / cost_filename

# Strategy-specific decision logic
skip_filter_data = False

if cost_strategy == 'reuse':
    is_compatible, reason = self.check_cost_file_compatibility(
        str(cost_path), env, variation, srctype, tartype, config.get('filter_data', {}).get('seq_len', 20)
    )
    
    if is_compatible:
        exp_logger.info("✓ Skipping filter_data.py - compatible cost file found")
        skip_filter_data = True
    else:
        exp_logger.info(f"→ Running filter_data.py - {reason}")
        skip_filter_data = False

elif cost_strategy == 'regenerate':
    exp_logger.info("Regenerating costs in global directory (strategy: regenerate)")
    skip_filter_data = False

elif cost_strategy == 'isolate':
    exp_logger.info("Generating costs in output directory (strategy: isolate)")
    skip_filter_data = False
```

**Modified Critical Section**:
```python
# CRITICAL SECTION: Only run if filter_data is needed
if not skip_filter_data:
    exp_logger.info(f"Waiting for filter_data lock...")
    with filter_data_lock:
        exp_logger.info(f"Acquired filter_data lock for {exp_name}")
        
        # Run filter_data.py
        filter_cmd = self.build_filter_data_command(env, variation, srctype, tartype, config, cost_strategy)
        exp_logger.info(f"Running filter_data: {' '.join(filter_cmd)}")
        
        # Run filter_data with real-time output streaming
        filter_result = self._run_with_streaming_output(
            filter_cmd, exp_logger, f"filter_data.py", str(Path.cwd())
        )
        
        exp_logger.info(f"Released filter_data lock for {exp_name}")
        
        if filter_result != 0:
            exp_logger.error(f"filter_data.py failed for {exp_name} with return code {filter_result}")
            # ... existing error handling ...
### Task 5: Integrate Strategy Logic into Experiment Execution

**File**: `run_experiments.py`
**Method**: `run_single_experiment()`

**Prerequisites**: Tasks 2, 3, and 4 must be complete

**Current Flow**:
1. Load config and apply overrides
2. **Always** run filter_data.py (within critical section)
3. Run experiment.py

**New Flow**:
1. Load config and apply overrides
2. **NEW**: Determine cost strategy and check compatibility
3. **Conditionally** run filter_data.py based on strategy decision
4. Run experiment.py

**Integration Point**: 
Insert cost strategy logic **after config processing** but **before the filter_data critical section**

**Specific Location in `run_single_experiment()`**:
Insert new logic after:
```python
# Save config snapshot directly in the main results directory
config_snapshot_file = self.output_dir / f'config_snapshot_{exp_name}.yaml'
with open(config_snapshot_file, 'w') as f:
    yaml.dump({'runner': config}, f, default_flow_style=False)

exp_logger.info(f"Starting experiment: {exp_name}")
start_time = time.time()
```

And before:
```python
# CRITICAL SECTION: Only one filter_data.py can run at a time due to GPU memory limitations
exp_logger.info(f"Waiting for filter_data lock...")
with filter_data_lock:
```

**New Code Block to Insert**:
```python
# Determine cost strategy and check if filter_data should be skipped
exp_logger.info("Checking if cost generation can be skipped...")

# Get cost directory and file path based on strategy
cost_dir = self.get_cost_directory(cost_strategy)
cost_filename = f"{env}-srcdatatype-{srctype}-tardatatype-{tartype}.hdf5"
cost_path = cost_dir / cost_filename

# Strategy-specific decision logic
skip_filter_data = False

if cost_strategy == 'reuse':
    is_compatible, reason = self.check_cost_file_compatibility(
        str(cost_path), env, variation, srctype, tartype, config.get('filter_data', {}).get('seq_len', 20)
    )
    
    if is_compatible:
        exp_logger.info("✓ Skipping filter_data.py - compatible cost file found")
        skip_filter_data = True
    else:
        exp_logger.info(f"→ Running filter_data.py - {reason}")
        skip_filter_data = False

elif cost_strategy == 'regenerate':
    exp_logger.info("Regenerating costs in global directory (strategy: regenerate)")
    skip_filter_data = False

elif cost_strategy == 'isolate':
    exp_logger.info("Generating costs in output directory (strategy: isolate)")
    skip_filter_data = False
```

**Modified Critical Section**:
```python
# CRITICAL SECTION: Only run if filter_data is needed
if not skip_filter_data:
    exp_logger.info(f"Waiting for filter_data lock...")
    with filter_data_lock:
        exp_logger.info(f"Acquired filter_data lock for {exp_name}")
        
        # Run filter_data.py
        filter_cmd = self.build_filter_data_command(env, variation, srctype, tartype, config, cost_strategy)
        exp_logger.info(f"Running filter_data: {' '.join(filter_cmd)}")
        
        # Run filter_data with real-time output streaming
        filter_result = self._run_with_streaming_output(
            filter_cmd, exp_logger, f"filter_data.py", str(Path.cwd())
        )
        
        exp_logger.info(f"Released filter_data lock for {exp_name}")
        
        if filter_result != 0:
            exp_logger.error(f"filter_data.py failed for {exp_name} with return code {filter_result}")
            # ... existing error handling ...
else:
    exp_logger.info("Skipped filter_data.py execution")
```

## Task Execution Order

The tasks have been reordered based on dependencies and logical implementation flow:

1. **Task 1**: ✅ **COMPLETE** - Cost strategy argument already implemented
2. **Task 2**: **NEXT** - Update method signatures and parameter passing (foundation)
3. **Task 3**: Add helper method and basic cost directory logic (enables basic functionality)
4. **Task 4**: Implement cost compatibility checking (enables smart reuse)
5. **Task 5**: Integrate strategy logic into experiment execution (completes implementation)

**Rationale for New Order**:
- Task 2 establishes the parameter plumbing required for all other tasks
- Task 3 enables basic cost directory selection without breaking existing functionality  
- Task 4 adds the intelligence for compatibility checking
- Task 5 ties everything together with the full strategy logic

This order ensures each task builds on the previous ones and maintains functionality throughout the implementation process.

### Test Commands (with --short mode for faster testing)

```bash
# Test 1: Default reuse strategy - should check compatibility and skip if compatible
python run_experiments.py --mode single --env hopper --variation kinematic --short

# Test 2: Regenerate strategy - should always run filter_data in global directory
python run_experiments.py --mode single --env hopper --variation kinematic --short --cost-strategy regenerate

# Test 3: Isolate strategy - should use output directory (current behavior)
python run_experiments.py --mode single --env hopper --variation kinematic --short --cost-strategy isolate

# Test 4: Reuse with non-existent cost file - should run filter_data
python run_experiments.py --mode single --env walker2d --variation morph --srctype medium-expert --tartype expert --short --cost-strategy reuse

# Test 5: Sequence length compatibility test - dimension mismatch detection
# Step 5a: Generate cost file with seq_len=20 (default)
python run_experiments.py --mode single --env hopper --variation kinematic --short --cost-strategy regenerate --override filter_data.seq_len=20

# Step 5b: Try to reuse with seq_len=5 - should detect incompatibility and regenerate
python run_experiments.py --mode single --env hopper --variation kinematic --short --cost-strategy reuse --override filter_data.seq_len=5

# Test 6: Reuse with existing but corrupted file - should run filter_data
# (Would need to manually corrupt a cost file to test this)
```

### Expected Log Patterns

**Successful reuse strategy (compatible file found)**:
```
INFO - Cost strategy: reuse
INFO - Using cost directory for filter_data: data/costs (strategy: reuse)
INFO - Setting experiment cost_dir to: data/costs (strategy: reuse)
INFO - Checking if cost generation can be skipped...
INFO - Checking cost file compatibility: data/costs/hopper-srcdatatype-medium-tardatatype-kinematic_medium.hdf5
INFO - ✓ Cost file exists
INFO - ✓ Cost file is valid HDF5
INFO - ✓ Cost file has required keys: ot_cost, mmd_cost
INFO - ✓ Cost arrays have valid shapes: ot_cost(2187, 1004), mmd_cost(2187, 1004)
INFO - ✓ Cost arrays can be reshaped to buffer dimensions
INFO - Cost file compatibility: COMPATIBLE
INFO - ✓ Skipping filter_data.py - compatible cost file found
INFO - Running experiment: python experiment.py --cost_dir data/costs ...
```

**Regenerate strategy**:
```
INFO - Cost strategy: regenerate
INFO - Using cost directory for filter_data: data/costs (strategy: regenerate)
INFO - Setting experiment cost_dir to: data/costs (strategy: regenerate)
INFO - Checking if cost generation can be skipped...
INFO - Regenerating costs in global directory (strategy: regenerate)
INFO - Running filter_data: python filter_data.py --dir data/costs ...
INFO - Running experiment: python experiment.py --cost_dir data/costs ...
```

**Isolate strategy**:
```
INFO - Cost strategy: isolate
INFO - Using cost directory for filter_data: ./save/results_20250902_143022/costs (strategy: isolate)
INFO - Setting experiment cost_dir to: ./save/results_20250902_143022/costs (strategy: isolate)
INFO - Checking if cost generation can be skipped...
INFO - Generating costs in output directory (strategy: isolate)
INFO - Running filter_data: python filter_data.py --dir ./save/results_20250902_143022/costs ...
INFO - Running experiment: python experiment.py --cost_dir ./save/results_20250902_143022/costs ...
```

**Reuse strategy (incompatible/missing file)**:
```
INFO - Cost strategy: reuse
INFO - Using cost directory for filter_data: data/costs (strategy: reuse)
INFO - Setting experiment cost_dir to: data/costs (strategy: reuse)
INFO - Checking if cost generation can be skipped...
INFO - Checking cost file compatibility: data/costs/walker2d-srcdatatype-medium-expert-tardatatype-morph_expert.hdf5
INFO - ✗ Cost file does not exist
INFO - Cost file compatibility: INCOMPATIBLE - cost file not found
INFO - → Running filter_data.py - cost file not found
INFO - Waiting for filter_data lock...
INFO - Acquired filter_data lock for walker2d-morph-medium-expert-to-morph_expert-TIMESTAMP
INFO - Running filter_data: python filter_data.py --dir data/costs ...
INFO - Released filter_data lock for walker2d-morph-medium-expert-to-morph_expert-TIMESTAMP
INFO - Running experiment: python experiment.py --cost_dir data/costs ...
```

**Sequence Length Compatibility Test (Test 5b - seq_len mismatch)**:
```
INFO - Cost strategy: reuse
INFO - Using cost directory for filter_data: data/costs (strategy: reuse)
INFO - Setting experiment cost_dir to: data/costs (strategy: reuse)
INFO - Checking if cost generation can be skipped...
INFO - Checking cost file compatibility: data/costs/hopper-srcdatatype-medium-tardatatype-kinematic_medium.hdf5
INFO - ✓ Cost file exists
INFO - ✓ Cost file is valid HDF5
INFO - ✓ Cost file has required keys: ot_cost, mmd_cost
INFO - ✓ Cost arrays have valid shapes: ot_cost(2187, 1004), mmd_cost(2187, 1004)
INFO - ✗ Cost arrays cannot be reshaped to buffer dimensions: expected shape for seq_len=5 but found seq_len=20 data
INFO - Cost file compatibility: INCOMPATIBLE - dimension mismatch: cost file has seq_len=20, experiment needs seq_len=5
INFO - → Running filter_data.py - dimension mismatch: cost file has seq_len=20, experiment needs seq_len=5
INFO - Waiting for filter_data lock...
INFO - Acquired filter_data lock for hopper-kinematic-medium-to-kinematic_medium-TIMESTAMP
INFO - Running filter_data: python filter_data.py --dir data/costs --seq_len 5 ...
INFO - Released filter_data lock for hopper-kinematic-medium-to-kinematic_medium-TIMESTAMP
INFO - Running experiment: python experiment.py --cost_dir data/costs ...
```

## Benefits of This Implementation

1. **Efficiency**: Automatic reuse of compatible cost files saves computation time
2. **Flexibility**: Three strategies cover different workflow needs
3. **Safety**: Compatibility checking prevents silent errors from mismatched cost files
4. **Clarity**: Single argument with clear semantics, no confusing flag combinations
5. **Backward Compatibility**: `isolate` strategy preserves current behavior
6. **Extensibility**: Easy to add new strategies in the future
7. **Comprehensive Logging**: Detailed logging for debugging and verification

## Default Behavior Change

**Current**: Always generate costs in output directory
**New**: Use existing compatible costs from global directory by default, generate only when needed

This change improves efficiency while maintaining all existing functionality through the `isolate` strategy.
