#!/usr/bin/env python3
"""
Script to analyze training progress from experiment log files.
Extracts "Current return is ..., normalized score is ..., Iteration ..." lines
and computes statistics using numpy for efficient numerical operations.
"""

import re
import sys
import glob
import math
from pathlib import Path
import argparse
import numpy as np
import yaml


# Dataset type abbreviations
DATASET_TYPE_ABBREVS = {
    'medium': 'm',
    'medium-expert': 'me',
    'medium_expert': 'me',
    'medium-replay': 'mr',
    'medium_replay': 'mr',
    'expert': 'e'
}


def abbreviate_dataset_types(exp_name):
    """
    Convert dataset type patterns in experiment names using abbreviations.
    Handles patterns like 'medium-to-expert' -> 'm2e', 'medium_expert-to-medium' -> 'me2m', etc.
    """
    # Create a pattern that matches any of the known dataset types
    dataset_types = list(DATASET_TYPE_ABBREVS.keys())
    # Sort by length (longest first) to match compound types first
    dataset_types.sort(key=len, reverse=True)
    
    # Create alternation pattern for dataset types
    dataset_pattern = '|'.join(re.escape(dt) for dt in dataset_types)
    
    # Pattern to match dataset transitions: (dataset_type)-to-(dataset_type)
    pattern = f'({dataset_pattern})-to-({dataset_pattern})'
    
    def replace_transition(match):
        source = match.group(1)
        target = match.group(2)
        
        # Get abbreviations
        source_abbrev = DATASET_TYPE_ABBREVS.get(source, source)
        target_abbrev = DATASET_TYPE_ABBREVS.get(target, target)
        
        return f"{source_abbrev}2{target_abbrev}"
    
    return re.sub(pattern, replace_transition, exp_name)




def extract_experiment_metadata(log_file_path):
    """
    Extract experiment metadata from log file.
    
    Returns:
        dict with start_time, end_time, duration, and other metadata
    """
    metadata = {
        'start_time': None,
        'end_time': None,
        'duration_seconds': None,
        'duration_hours': None,
        'experiment_name': None,
        'status': 'unknown'
    }
    
    # Patterns to match
    start_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - Starting experiment: (.+)'
    end_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - Completed experiment: (.+) in ([\d.]+)s'
    final_result_pattern = r'The final best return mean is ([\d.-]+), normalized score is ([\d.-]+)'
    
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
            
            # Find start time
            start_match = re.search(start_pattern, content)
            if start_match:
                metadata['start_time'] = start_match.group(1)
                metadata['experiment_name'] = start_match.group(2)
            
            # Find end time and duration
            end_match = re.search(end_pattern, content)
            if end_match:
                metadata['end_time'] = end_match.group(1)
                metadata['duration_seconds'] = float(end_match.group(3))
                metadata['duration_hours'] = metadata['duration_seconds'] / 3600
                metadata['status'] = 'completed'
            
            # Check for final results to confirm success
            if re.search(final_result_pattern, content):
                metadata['status'] = 'success'
            elif 'ERROR' in content or 'failed' in content:
                metadata['status'] = 'failed'
            elif metadata['end_time'] is None:
                metadata['status'] = 'incomplete'
                
    except Exception as e:
        print(f"Error extracting metadata from {log_file_path}: {e}")
    
    return metadata


def extract_experiment_config(log_dir):
    """
    Extract experiment configuration from run arguments and config files.
    
    Returns:
        dict with configuration information
    """
    config_info = {
        'run_args': None,
        'config_files': [],
        'summary': {}
    }
    
    try:
        # Find run arguments file
        args_files = glob.glob(f"{log_dir}/run_experiments_args_*.yaml")
        if args_files:
            args_file = args_files[0]
            with open(args_file, 'r') as f:
                config_info['run_args'] = yaml.safe_load(f)
                
                # Extract key summary information
                args = config_info['run_args']
                config_info['summary'] = {
                    'environment': args.get('env', 'unknown'),
                    'variation': args.get('variation', 'unknown'),
                    'mode': args.get('mode', 'unknown'),
                    'overrides': args.get('overrides', []),
                    'timestamp': args.get('timestamp', 'unknown')
                }
        
        # Find config snapshot files
        config_files = glob.glob(f"{log_dir}/config_snapshot_*.yaml")
        if config_files:
            # Just store the filenames for reference, loading all would be too much
            config_info['config_files'] = [Path(f).name for f in config_files[:3]]  # Limit to first 3
            
            # Load one example config for key parameters
            try:
                with open(config_files[0], 'r') as f:
                    sample_config = yaml.safe_load(f)
                    if 'runner' in sample_config and 'experiment' in sample_config['runner']:
                        exp_config = sample_config['runner']['experiment']
                        config_info['summary'].update({
                            'max_iters': exp_config.get('max_iters', 'unknown'),
                            'seed': exp_config.get('seed', 'unknown'),
                            'lr_decay': exp_config.get('lr_decay', 'unknown'),
                            'cql_weight': exp_config.get('cql_weight', 'unknown'),
                            'eta': exp_config.get('eta', 'unknown')
                        })
            except Exception as e:
                pass  # Skip if config parsing fails
                
    except Exception as e:
        print(f"Warning: Could not extract config from {log_dir}: {e}")
    
    return config_info


def extract_progress_data(log_file_path):
    """
    Extract training progress data from a log file.
    
    Returns:
        list of tuples: (iteration, return_mean, normalized_score)
    """
    progress_data = []
    
    # Pattern to match lines like:
    # "Current return mean is 1553.48..., normalized score is 49.694..., Iteration 101"
    pattern = r'Current return mean is ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), normalized score is ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), Iteration (\d+)'
    
    try:
        with open(log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                match = re.search(pattern, line)
                if match:
                    return_mean = float(match.group(1))
                    normalized_score = float(match.group(2))
                    iteration = int(match.group(3))
                    progress_data.append((iteration, return_mean, normalized_score))
    except FileNotFoundError:
        print(f"Error: File {log_file_path} not found")
        return []
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return []
    
    return progress_data

# update this function
def compute_stats(values):
    """
    Compute statistics for a list of values using numpy.
    """
    stats = {
        "max": 0,
        "max_last10": 0,
        "avg_last10": 0,
        "std_last10": 0,
        "max_last5": 0,
        "avg_last5": 0,
        "std_last5": 0,
        "max_last3": 0,
        "avg_last3": 0,
        "std_last3": 0,
        "last": 0
    }
    
    if len(values) == 0:
        return stats
    
    # Convert to numpy array for easier manipulation
    values_array = np.array(values)
    
    # Overall max
    max_val = np.max(values_array)
    
    # Last N values statistics
    def get_stats_for_last_n(n):
        last_n = values_array[-n:] if len(values_array) >= n else values_array
        return {
            'max': np.max(last_n),
            'avg': np.mean(last_n),
            'std': np.std(last_n, ddof=1) if len(last_n) > 1 else 0.0
        }
    
    last10_stats = get_stats_for_last_n(10)
    last5_stats = get_stats_for_last_n(5)
    last3_stats = get_stats_for_last_n(3)
    last = values_array[-1]
    
    return {
        "max": max_val,
        "max_last10": last10_stats['max'],
        "avg_last10": last10_stats['avg'],
        "std_last10": last10_stats['std'],
        "max_last5": last5_stats['max'],
        "avg_last5": last5_stats['avg'],
        "std_last5": last5_stats['std'],
        "max_last3": last3_stats['max'],
        "avg_last3": last3_stats['avg'],
        "std_last3": last3_stats['std'],
        "last": last
    }


def analyze_all_experiments(log_dir):
    """
    Analyze all experiment log files and output results in table format.
    """
    log_pattern = f"{log_dir}/*.log"
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        print(f"No log files found in: {log_dir}")
        return
    
    # Extract experiment configuration first
    config_info = extract_experiment_config(log_dir)
    
    # Skip the main run_experiments.log file
    if len(log_files) > 1:
        log_files = [f for f in log_files if 'run_experiments.log' not in f]
        log_files.sort()
        
    results = []
    
    for log_file in log_files:
        # Extract both progress and metadata
        progress_data = extract_progress_data(log_file)
        metadata = extract_experiment_metadata(log_file)
        
        if not progress_data:
            # Add empty row for failed experiments - use metadata name if available
            if metadata['experiment_name']:
                exp_name = metadata['experiment_name']
            else:
                exp_name = Path(log_file).stem
            
            # Clean up dataset type patterns using abbreviations
            exp_name = abbreviate_dataset_types(exp_name)
            results.append({
                'experiment': exp_name,
                'iterations': 0,
                'return_stats': compute_stats([]),
                'score_stats': compute_stats([]),
                'metadata': metadata
            })
            continue
        
        # Sort by iteration
        progress_data.sort(key=lambda x: x[0])
        
        # Convert to numpy arrays for efficient processing
        progress_array = np.array(progress_data)
        iterations = progress_array[:, 0].astype(int)
        returns = progress_array[:, 1].astype(float)
        normalized_scores = progress_array[:, 2].astype(float)
        
        # Clean up experiment name - use metadata name if available, otherwise file stem
        if metadata['experiment_name']:
            exp_name = metadata['experiment_name']
        else:
            exp_name = Path(log_file).stem
        
        # Clean up dataset type patterns using abbreviations
        exp_name = abbreviate_dataset_types(exp_name)
        
        # Shorten environment names
        exp_name = exp_name.replace('halfcheetah', 'hc').replace('hopper', 'hop').replace('walker2d', 'w2d')
        exp_name = exp_name.replace('gravity', 'grav').replace('kinematic', 'kin').replace('morph', 'morph')
        
        results.append({
            'experiment': exp_name,
            'iterations': len(progress_data),
            'return_stats': compute_stats(returns),
            'score_stats': compute_stats(normalized_scores),
            'metadata': metadata
        })
    
    # Print markdown table
    print("\n# Training Progress Analysis - Experiment Statistics\n")
    
    # Print experiment configuration if available
    if config_info['summary']:
        print("## Experiment Configuration\n")
        summary = config_info['summary']
        print(f"**Environment:** {summary.get('environment', 'N/A')}")
        print(f"**Variation:** {summary.get('variation', 'N/A')}")
        print(f"**Mode:** {summary.get('mode', 'N/A')}")
        print(f"**Timestamp:** {summary.get('timestamp', 'N/A')}")
        
        if summary.get('overrides'):
            print(f"**Overrides:** {', '.join(summary['overrides'])}")
        
        if 'max_iters' in summary:
            print(f"**Max Iterations:** {summary.get('max_iters', 'N/A')}")
            print(f"**Seed:** {summary.get('seed', 'N/A')}")
            print(f"**CQL Weight:** {summary.get('cql_weight', 'N/A')}")
            print(f"**Eta:** {summary.get('eta', 'N/A')}")
        
        if config_info['config_files']:
            print(f"**Config Files:** {', '.join(config_info['config_files'])}")
        
        print("\n---\n")
    
    print("## Results Table\n")
    
    # Markdown table header with timing info

    headers = [
        "Experiment", "Start", "Duration(h)", "Status", "Iters", "Return_Max", 
        "Return_Max(L10/L5/L3)", "Return_Avg(L10/L5/L3)", "Return_Std(L10/L5/L3)", 
        "Score_Max", "Score_Max(L10/L5/L3)", "Score_Avg(L10/L5/L3)", "Score_Std(L10/L5/L3)",
        "Last_Return", "Last_Score"
    ]

    header_str = "| " + " | ".join(headers) + " |"
    separator_str = "| " + " | ".join(['---'] * len(headers)) + " |"

    print(header_str)
    print(separator_str)
    
    # Print data rows
    for result in results:
        r_stats = result['return_stats']
        s_stats = result['score_stats']
        meta = result['metadata']
        
        # Format start time (just date and hour)
        start_str = "N/A"
        if meta['start_time']:
            start_str = meta['start_time'][5:16]  # MM-DD HH:MM
        
        # Format duration
        duration_str = "N/A"
        if meta['duration_hours'] is not None:
            duration_str = f"{meta['duration_hours']:.1f}"
        
        # Format status
        status_map = {
            'success': '✓',
            'completed': '✓',
            'failed': '✗',
            'incomplete': '⚠',
            'unknown': '?'
        }
        status_str = status_map.get(meta['status'], '?')
        
        row = f"| {result['experiment']} | {start_str} | {duration_str} | {status_str} | {result['iterations']} | "
        row += f"{r_stats['max']:.1f} | {r_stats['max_last10']:.1f}/{r_stats['max_last5']:.1f}/{r_stats['max_last3']:.1f} | {r_stats['avg_last10']:.1f}/{r_stats['avg_last5']:.1f}/{r_stats['avg_last3']:.1f} | {r_stats['std_last10']:.1f}/{r_stats['std_last5']:.1f}/{r_stats['std_last3']:.1f} | {r_stats['last']:.1f} | "
        row += f"{s_stats['max']:.2f} | {s_stats['max_last10']:.2f}/{s_stats['max_last5']:.2f}/{s_stats['max_last3']:.2f} | {s_stats['avg_last10']:.2f}/{s_stats['avg_last5']:.2f}/{s_stats['avg_last3']:.2f} | {s_stats['std_last10']:.3f}/{s_stats['std_last5']:.3f}/{s_stats['std_last3']:.3f} | {s_stats['last']:.2f} |"

        
        print(row)
    
    print("\n**Legend:** L10/L5/L3=Last10/Last5/Last3, Max=Maximum, Avg=Average, Std=Standard Deviation")
    print("**Status:** ✓=Success, ✗=Failed, ⚠=Incomplete, ?=Unknown\n")
    
    # Find best performers
    valid_results = [r for r in results if r['iterations'] > 0]
    if valid_results:
        print("## Top Performers\n")
        # Sort by max normalized score
        top_by_score = sorted(valid_results, key=lambda x: x['score_stats']['max'], reverse=True)[:3]
        for i, result in enumerate(top_by_score):
            duration_str = f"{result['metadata']['duration_hours']:.1f}h" if result['metadata']['duration_hours'] else "N/A"
            print(f"{i+1}. **{result['experiment']}**: {result['score_stats']['max']:.2f}% norm score, {result['return_stats']['max']:.1f} return ({duration_str})")
        
        print("\n## Experiment Timeline\n")
        # Sort by start time
        timed_results = [r for r in valid_results if r['metadata']['start_time']]
        if timed_results:
            timed_results.sort(key=lambda x: x['metadata']['start_time'])
            for result in timed_results:
                start_time = result['metadata']['start_time'][5:16] if result['metadata']['start_time'] else "N/A"
                duration = f"{result['metadata']['duration_hours']:.1f}h" if result['metadata']['duration_hours'] else "N/A"
                status = result['metadata']['status']
                print(f"- **{start_time}**: {result['experiment']} ({duration}, {status})")
        
        print()  # Add extra newline


def main():
    parser = argparse.ArgumentParser(description="Analyze training progress from experiment logs")
    parser.add_argument('--log-dir', type=str, default='save/results_20250822_224227',
                       help='Directory containing log files')
    parser.add_argument('--file', type=str, 
                       help='Analyze a specific log file')
    parser.add_argument('--output', type=str,
                       help='Save markdown output to file')
    
    args = parser.parse_args()
    
    # Capture output if saving to file
    if args.output:
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
    
    if args.file:
        # Single file analysis
        progress_data = extract_progress_data(args.file)
        if progress_data:
            # Convert to numpy arrays for efficient processing
            progress_array = np.array(progress_data)
            returns = progress_array[:, 1].astype(float)
            scores = progress_array[:, 2].astype(float)
            r_stats = compute_stats(returns)
            s_stats = compute_stats(scores)
            
            print(f"\n# Analysis of {Path(args.file).name}\n")
            print(f"**Iterations:** {len(progress_data)}")
            print(f"**Return** - Max: {r_stats['max']:.2f}, Last10/5/3 Max: {r_stats['max_last10']:.2f}/{r_stats['max_last5']:.2f}/{r_stats['max_last3']:.2f}, Last10/5/3 Avg: {r_stats['avg_last10']:.2f}/{r_stats['avg_last5']:.2f}/{r_stats['avg_last3']:.2f}")
            print(f"**Score** - Max: {s_stats['max']:.2f}%, Last10/5/3 Max: {s_stats['max_last10']:.2f}%/{s_stats['max_last5']:.2f}%/{s_stats['max_last3']:.2f}%, Last10/5/3 Avg: {s_stats['avg_last10']:.2f}%/{s_stats['avg_last5']:.2f}%/{s_stats['avg_last3']:.2f}%")
        else:
            print("No progress data found in file.")
    else:
        # Analyze all files in directory
        analyze_all_experiments(args.log_dir)
    
    # Save to file if requested
    if args.output:
        sys.stdout = old_stdout
        output_content = captured_output.getvalue()
        with open(args.output, 'w') as f:
            f.write(output_content)
        print(f"Markdown output saved to: {args.output}")
        print(output_content)  # Also print to console


if __name__ == "__main__":
    main()
