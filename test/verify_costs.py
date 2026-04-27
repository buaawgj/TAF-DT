from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import sys

import sklearn.metrics
sys.path.append(str(Path(__file__).resolve().parent.parent))

import h5py
import sklearn
import ot
from tqdm import tqdm

from decision_transformer.misc.utils import call_d4rl_dataset, call_tar_dataset, get_keys

def get_trajectories(dataset):
    """
    Extract trajectories from the dataset.
    
    :param dataset: The dataset from which to extract trajectories.
    :return: List of trajectories.
    """
    trajectories = []
    curr_traj = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': [],
        'terminals': [],
        'masks': [],
        'timesteps': []
    }
    curr_timestep = 0
    for i in range(len(dataset['observations'])):
        curr_traj['observations'].append(dataset['observations'][i])
        curr_traj['actions'].append(dataset['actions'][i])
        curr_traj['rewards'].append(dataset['rewards'][i])
        curr_traj['next_observations'].append(dataset['next_observations'][i])
        curr_traj['terminals'].append(dataset['terminals'][i] if 'terminals' in dataset else dataset['dones'][i])
        curr_traj['masks'].append(1.0 - dataset['terminals'][i] if 'terminals' in dataset else 1.0 - dataset['dones'][i])
        curr_traj['timesteps'].append(curr_timestep)
        curr_timestep += 1

        if dataset['terminals'][i] == 1 or i == len(dataset['observations']) - 1:
            # end of trajectory, save current trajectory
            for k in curr_traj:
                curr_traj[k] = np.array(curr_traj[k])
            trajectories.append(curr_traj)
            curr_traj = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': [],
                'terminals': [],
                'masks': [],
                'timesteps': []
            }
            curr_timestep = 0

def get_features(dataset):
    """
    Extract features from the dataset.
    
    :param dataset: The dataset from which to extract features.
    :return: Dictionary of features.
    """
    features = np.hstack([
        dataset['observations'],
        dataset['actions'],
        dataset['next_observations'],
        dataset['rewards'].reshape(-1, 1),
    ])
    # remove columns with all zeros
    features = features[:, np.any(features != 0, axis=0)]

    return features

def solve_optimal_transport(source_data, target_data):
    """
    Solve the optimal transport problem between source and target data.
    
    :param source_data: Source dataset containing observations and actions.
    :param target_data: Target dataset containing observations and actions.
    :return: Optimal transport costs.
    """
    # Placeholder for actual OT computation logic
    # This should be replaced with the actual implementation
    distance_matrix = sklearn.metrics.pairwise.paired_cosine_distances(source_data, target_data)
    ot_matrix = ot.sinkhorn(
        np.ones(source_data.shape[0]) / source_data.shape[0],
        np.ones(target_data.shape[0]) / target_data.shape[0],
        distance_matrix,
        reg=0.1
    )
    costs = np.sum(ot_matrix * distance_matrix, axis=1)
    
    return costs

def compute_optimal_transport_costs(data):
    """
    Compute optimal transport costs based on the provided data.
    
    :param data: Dictionary containing the necessary data for cost computation.
    :return: Dictionary of computed costs.
    """
    # Placeholder for actual cost computation logic
    # This should be replaced with the actual implementation
    costs = {}
    for key, value in data.items():
        costs[key] = value * 2  # Example computation
    return costs

def verify_costs(h5_file_path, expected_costs):
    """
    Verify that the costs in the HDF5 file match the expected costs.

    :param h5_file_path: Path to the HDF5 file.
    :param expected_costs: Dictionary of expected costs.
    :return: True if all costs match, False otherwise.
    """
    with h5py.File(h5_file_path, 'r') as f:
        for key, expected_value in expected_costs.items():
            if key not in f:
                print(f"Key '{key}' not found in the file.")
                return False
            actual_value = f[key][()]
            if actual_value != expected_value:
                print(f"Cost mismatch for '{key}': expected {expected_value}, got {actual_value}.")
                return False
    return True

if __name__ == "__main__":
    # Example usage
    parser = ArgumentParser(description="Verify costs in an HDF5 file.")
    parser.add_argument("--src-env", type=str, required=True, help="Source environment for cost computation. Should be an environment in D4RL.")
    parser.add_argument("--tar-dir-name", type=str, default='data/target_dataset', help="Directory name for the target dataset.")
    parser.add_argument("--tar-env-name", type=str, required=True, help="Target environment for cost computation. Should be an environment where a D4RL formatted dataset is available.")
    parser.add_argument("--tar-datatype", type=str, default='data/target_dataset', help="Directory name for the target dataset.")
    parser.add_argument("--cost-path", type=str, help="Path to the HDF5 file containing costs.")

    args = parser.parse_args()
    source_data = call_d4rl_dataset(args.src_env)
    target_data = call_tar_dataset(args.tar_dir_name, args.tar_env_name, args.tar_datatype)

    data_dict = {}
    with h5py.File(args.cost_path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    costs = data_dict
        
    # print(source_data)
    # print(target_data)
    for k in source_data:
        print(f"Source data {k}: {source_data[k].shape}")
    for k in target_data:
        print(f"Target data {k}: {target_data[k].shape}")
    for k in costs:
        print(f"Cost {k}: {costs[k].shape}")
    print("Data loaded successfully.")

    src_trajectories = get_trajectories(source_data)

    tar_features = np.hstack([
        target_data['observations'],
        target_data['actions'],
        target_data['next_observations'],
        target_data['rewards'].reshape(-1, 1),
    ])

    for i in range(len(src_trajectories)):
        src_features = np.hstack([
            src_trajectories[i]['observations'],
            src_trajectories[i]['actions'],
            src_trajectories[i]['next_observations'],
            src_trajectories[i]['rewards'].reshape(-1, 1),
        ])
        # Solve the optimal transport problem
        ot_costs = solve_optimal_transport(src_features, tar_features)
        # pad on left to match the target dataset length
        ot_costs = np.concatenate([
            np.zeros((tar_features.shape[0] - ot_costs.shape[0],1)),
            ot_costs
        ])

# usage example: hopper-medium-v2 as source, hopper-kinematic-medium as target
# python test/verify_costs.py --src-env hopper-medium-v2 --tar-dir-name data/target_dataset --tar-env-name hopper-kinematic --tar-datatype medium --cost-path data/costs/hopper-srcdatatype-medium-tardatatype-kinematic_medium.hdf5

