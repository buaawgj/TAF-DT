# based on https://github.com/dmksjfl/OTDF/blob/master/make_otdf_cost.py

import numpy as np
import torch
import gym
import argparse
import os
import random
import math
import time
import copy
from pathlib import Path
import yaml
import h5py
import matplotlib.pyplot as plt

import d4rl
from tqdm import tqdm, trange

import ot
import numpy as np

# import algo.utils as utils
# import decision_transformer.utils_ref as utils_ref
# from decision_transformer.misc.utils import SequenceReplayBuffer, call_tar_dataset
from decision_transformer.misc.utils import TrajectoryBuffer, call_d4rl_dataset, call_tar_dataset, pad_along_axis
from envs.env_utils import call_terminal_func
from envs.common import call_env

@torch.no_grad()
def compute_kernel(src_tensor, tar_tensor=None, kernel_type='rbf'):
    """
    Computes the kernel between source and target data using PyTorch tensors.
    :param src_tensor: Source data (numpy array).
    :param tar_tensor: Target data (numpy array).
    :param kernel_type: Type of kernel to compute ('rbf' or 'linear').
    :return: Computed kernel matrix.
    """
    if tar_tensor is None:
        tar_tensor = src_tensor  # If no target data is provided, use source data for self-kernel
    if kernel_type == 'rbf':
        # Compute RBF kernel
        gamma = 1.0 / src_tensor.shape[1]
        src_norm = torch.sum(src_tensor ** 2, dim=1, keepdim=True)
        tar_norm = torch.sum(tar_tensor ** 2, dim=1, keepdim=True)
        cross_term = torch.mm(src_tensor, tar_tensor.t())
        distances = src_norm + tar_norm.t() - 2 * cross_term # (a - b)^2 = a^2 + b^2 - 2ab
        kernel_matrix = torch.exp(-gamma * distances)  # RBF kernel
    elif kernel_type == 'linear':
        # Compute linear kernel
        kernel_matrix = torch.mm(src_tensor, tar_tensor.t())
    else:
        raise ValueError("Unsupported kernel type. Use 'rbf' or 'linear'.")
    return kernel_matrix

# @torch.no_grad()
# def compute_kernel_window_means(data, window_size, kernel_type='rbf'):
#     """
#     Computes the kernel mean for each sliding window of size `window_size` in the data.
#     """
#     window_kernel_means = np.zeros(len(data))
#     for i in range(len(data)):
#         window = data[i:i + window_size]
#         window_kernel_means[i] = compute_kernel(window, kernel_type=kernel_type).mean()
#     return window_kernel_means

@torch.no_grad()
def compute_kernel_window_means(src_tensor, tar_tensor, window_size, kernel_type='rbf'):
    """
    Computes the cross kernel mean for each sliding window of size `window_size` between source and target data.
    """
    cross_kernel = compute_kernel(src_tensor, tar_tensor, kernel_type=kernel_type)
    cross_kernel = cross_kernel.unsqueeze(0).unsqueeze(0)
    padded_kernel = torch.nn.functional.pad(cross_kernel, (0, window_size - 1, 0, window_size - 1), mode='constant', value=0)
    cross_kernel_means = torch.nn.functional.avg_pool2d(padded_kernel, kernel_size=window_size, stride=1)
    cross_kernel_means = cross_kernel_means.squeeze(0).squeeze(0)  # Remove the added dimensions

    # Normalize by the number of elements in the window
    src_len = src_tensor.shape[0]
    tar_len = tar_tensor.shape[0]
    mean_range = torch.full((src_len, tar_len), window_size ** 2)
    mean_range[-window_size:, :] = mean_range[-window_size:, :] // window_size * torch.arange(min(window_size, src_len), 0, -1).reshape(-1, 1)
    mean_range[:, -window_size:] = mean_range[:, -window_size:] // window_size * torch.arange(min(window_size, tar_len), 0, -1).reshape(1, -1)
    mean_range = mean_range.to(cross_kernel_means.device)  # Ensure the mean_range is on the same device as src_tensor
    cross_kernel_means = cross_kernel_means * window_size ** 2 / mean_range  
    return cross_kernel_means

@torch.no_grad()
def solve_mmd(src_tensor, tar_tensor, window_size, src_kernel_window_means=None, tar_kernel_window_means=None, kernel_type='rbf', device='cpu'):
    if src_kernel_window_means is None:
        src_kernel_window_means = compute_kernel_window_means(src_tensor, src_tensor, window_size, kernel_type=kernel_type)

    if tar_kernel_window_means is None:
        tar_kernel_window_means = compute_kernel_window_means(tar_tensor, tar_tensor, window_size=window_size, kernel_type=kernel_type)

    cross_kernel_window_means = compute_kernel_window_means(src_tensor, tar_tensor,  window_size, kernel_type=kernel_type)

    src_kernel_window_means = src_kernel_window_means.diagonal().view(-1, 1)
    tar_kernel_window_means = tar_kernel_window_means.diagonal().view(1, -1)  # Convert to row vector
    cross_kernel_window_means = cross_kernel_window_means
    mmd_costs = src_kernel_window_means + tar_kernel_window_means - 2 * cross_kernel_window_means

    mmd_costs = mmd_costs.min(dim=1).values  # Reduce to source samples
    return mmd_costs  # Return negative MMD costs for minimization

@torch.no_grad()
def solve_ot(
    src_tensor, tar_tensor, cost_type='sqeuclidean', steps=1e5, ot_method='sinkhorn', ot_reg=0.1
):
    device = src_tensor.device

    src_B = src_tensor.shape[0]
    tar_B = tar_tensor.shape[0]

    # convert to float64 for convergence in sinkhorn
    src_probs = torch.ones((src_B,), dtype=torch.float64, device=device)
    tar_probs = torch.ones((tar_B,), dtype=torch.float64, device=device)
    src_probs /= src_B
    tar_probs /= tar_B

    if cost_type != 'sqeuclidean' and cost_type != 'euclidean':
        src_arr = src_tensor.cpu().numpy()
        tar_arr = tar_tensor.cpu().numpy()
    print("src_tensor shape:", src_tensor.shape)
    print("tar_tensor shape:", tar_tensor.shape)
    cost_matrix = ot.dist(src_arr, tar_arr, metric=cost_type)

    # convert to float64 for convergence in sinkhorn
    cost_matrix = torch.tensor(cost_matrix, dtype=torch.float64, device=device)
    cost_matrix_scaled = cost_matrix / torch.max(cost_matrix)  # bound costs to avoid numerical issues

    ot_matrix = ot.sinkhorn(src_probs, tar_probs, cost_matrix_scaled, reg=ot_reg, numItermax=int(steps), method=ot_method)

    ot_costs = torch.einsum('ij,ij->i', ot_matrix, cost_matrix)

    ot_costs = ot_costs.to(torch.float32)  # Convert back to float32 for consistency

    return ot_costs

def make_tensor_list_from_buffer(buffer: TrajectoryBuffer, step_type: str = "all"):
    """
    Create tensors from the trajectory buffer for source and target datasets.
    """
    data_list = []

    for i in range(len(buffer.observations)):
        seq_len = buffer.traj_lens[i]
        if buffer.padding == "right":
            slice_ = slice(0, seq_len)
        elif buffer.padding == "left":
            slice_ = slice(-seq_len, None)
        if step_type == "state":
            traj_data = np.hstack([buffer.observations[i][slice_]])
        elif step_type == "transition":
            traj_data = np.hstack([
                buffer.observations[i][slice_],
                buffer.actions[i][slice_],
                buffer.next_observations[i][slice_],
            ])
        else:
            traj_data = np.hstack([
                buffer.observations[i][slice_],
                buffer.actions[i][slice_],
                buffer.next_observations[i][slice_],
                buffer.rewards[i][slice_],
            ])
        traj_data = torch.tensor(traj_data, dtype=torch.float32)
        data_list.append(traj_data)
    return data_list


def filter_dataset(src_buffer: TrajectoryBuffer, tar_buffer: TrajectoryBuffer, 
                   seq_len: int, ot_metric='cosine', mmd_metric='rbf', batch_size=10000, 
                   num_trajectories=None, steps=1e5, ot_method='sinkhorn', ot_reg=0.1, mmd_step_type="state", ot_step_type="all"):
    """
    Filter the source replay buffer based on the target replay buffer using optimal transport and MMD.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_num = src_buffer.rewards.shape[0]
    tar_num = tar_buffer.rewards.shape[0]

    if num_trajectories is None:
        num_trajectories = src_num

    print(f"Creating tensor representations from buffers...")
    src_tensor_list = make_tensor_list_from_buffer(src_buffer, ot_step_type) # (src_num, seq_len[i], combined_dim)
    mmd_src_tensor_list = make_tensor_list_from_buffer(src_buffer, mmd_step_type)
    src_tensor_full = torch.cat(src_tensor_list, dim=0)  # (num_src_steps, combined_dim)
    mmd_src_tensor_full = torch.cat(mmd_src_tensor_list, dim=0) 
    tar_tensor_list = make_tensor_list_from_buffer(tar_buffer, ot_step_type)
    mmd_tar_tensor_list = make_tensor_list_from_buffer(tar_buffer, mmd_step_type)
    tar_tensor_full = torch.cat(tar_tensor_list, dim=0)  # (num_tar_steps, combined_dim)
    mmd_tar_tensor_full = torch.cat(mmd_tar_tensor_list, dim=0)
   
    # compute OT costs
    src_cum_traj_lens = np.cumsum(src_buffer.traj_lens[:num_trajectories]).astype(np.int32)  # Cumulative lengths of trajectories
    src_cum_traj_lens = np.concatenate(([0], src_cum_traj_lens), axis=0)  # Add zero at the start
    tar_cum_traj_lens = np.cumsum(tar_buffer.traj_lens).astype(np.int32)  # Cumulative lengths of trajectories
    tar_cum_traj_lens = np.concatenate(([0], tar_cum_traj_lens), axis=0)  # Add zero at the start

    valid_indices = [(i, j) for i in range(num_trajectories) for j in range(src_buffer.traj_lens[i])]    # assert all rows are 
    # randomize order
    np.random.shuffle(valid_indices)
    # divide into splits
    num_splits = math.ceil(len(valid_indices) / batch_size)
    split_indices = np.array_split(valid_indices, num_splits)
    ot_result = np.zeros((num_trajectories, src_buffer.max_len), dtype=np.float32)  # (num_src_samples, max_len)

    print(f"Starting OT cost calculation for {num_splits} batches...")
    for i in trange(num_splits, desc="Calculating OT costs", mininterval=5.0, maxinterval=30.0):
        batch_indices = split_indices[i]
        batch_steps = src_cum_traj_lens[batch_indices[:, 0]] + batch_indices[:, 1]  # Convert to absolute step indices
        batch_steps = batch_steps.tolist()  # Convert to list for indexing
        assert len(batch_steps) == len(set(batch_steps)), "There are duplicate steps in batch_steps."
        src_tensor = src_tensor_full[batch_steps]  # (num_steps_in_batch, combined_dim)
        src_tensor = src_tensor.to(device)  # Move to device
        if i == 0:
            print(f"OT src_tensor shape = {src_tensor.shape}")
        ot_costs = solve_ot(
            src_tensor, tar_tensor_full, cost_type=ot_metric, steps=steps, ot_method=ot_method, ot_reg=ot_reg
        )
        # normalize by batch size
        ot_costs = ot_costs / batch_indices.shape[0] * batch_size
        ot_costs = ot_costs.cpu().numpy()  # Move to CPU and convert to numpy
        if src_buffer.padding == 'left':
            batch_indices[:, 1] += src_buffer.max_len - src_buffer.traj_lens[batch_indices[:, 0]]  # Adjust for left padding
        ot_result[batch_indices[:, 0], batch_indices[:, 1]] = ot_costs
    
    assert np.all(ot_result[src_buffer.masks[:num_trajectories, :] == 1]) != 0, "OT Result contains zero values where mask is 1, indicating a mismatch in expected costs."

    assert np.all(ot_result[src_buffer.masks[:num_trajectories, :] == 0]) == 0, "OT Result contains non-zero values where mask is 0, indicating a mismatch in expected costs."

    print("OT cost calculation completed successfully.")
    print("Starting MMD cost calculation...")
    # Compute MMD costs

    src_kernel_window_means = [
        compute_kernel_window_means(src_tensor, src_tensor, seq_len, kernel_type=mmd_metric).to(device)  # (num_src_steps, num_src_steps)
        for src_tensor in mmd_src_tensor_list[:num_trajectories]
    ]

    tar_kernel_window_means = [
        compute_kernel_window_means(tar_tensor, tar_tensor, seq_len, kernel_type=mmd_metric).to(device)  # (num_tar_steps, num_tar_steps)
        for tar_tensor in mmd_tar_tensor_list
    ]

    print(f"Starting MMD cost calculation for {num_trajectories} trajectories...")
    mmd_result = []

    for i in trange(num_trajectories, desc="Calculating MMD costs", mininterval=5.0, maxinterval=30.0):
        mmd_costs = np.full(src_buffer.max_len, np.inf, dtype=np.float32)  # Initialize with inf
        for j in range(len(mmd_tar_tensor_list)):
            src_tensor = mmd_src_tensor_list[i].to(device)  # Move to device
            tar_tensor = mmd_tar_tensor_list[j].to(device)  # Move to device

            if i == 50 and j == 0:
                print(f"MMD src_tensor shape = {src_tensor.shape}")
                print(f"MMD tar_tensor shape = {tar_tensor.shape}")

            # print(type(src_tensor), type(tar_tensor), src_tensor.shape, tar_tensor.shape)
            curr_mmd_costs = solve_mmd(
                src_tensor, tar_tensor, seq_len, src_kernel_window_means[i], tar_kernel_window_means[j], kernel_type=mmd_metric
            )

            max_len = src_buffer.max_len

            curr_mmd_costs = curr_mmd_costs.cpu().numpy()  # Move to CPU and convert to numpy

            curr_mmd_costs = pad_along_axis(curr_mmd_costs, max_len, axis=0, fill_value=np.inf, padding=src_buffer.padding)
            mmd_costs = np.minimum(mmd_costs, curr_mmd_costs)  # Take the minimum across all target tensors
        mmd_result.append(mmd_costs)

    # print length of each element in ot_result and mmd_result
    # print(f"OT Result Lengths: {[len(x) for x in ot_result]}")
    # print(f"MMD Result Lengths: {[len(x) for x in mmd_result]}")

    ot_result = np.array(ot_result, dtype=np.float32)  # (num_src_samples, max_len)
    mmd_result = np.array(mmd_result, dtype=np.float32)  # (num_src_samples, max_len)

    assert ot_result.shape == (num_trajectories, max_len), f"OT Result shape mismatch: {ot_result.shape} vs {(num_trajectories, max_len)}"

    assert mmd_result.shape == (num_trajectories, max_len), f"MMD Result shape mismatch: {mmd_result.shape} vs {(num_trajectories, max_len)}"

    print("MMD cost calculation completed successfully.")
    return ot_result, mmd_result

# def get_topk_values(cost, k=100):
#     """
#     Get the top-k values and indices from the cost array.
#     """
#     if len(cost) < k:
#         k = len(cost)
#     topk_values, topk_indices = torch.topk(torch.tensor(cost), k=k, largest=True)
#     return topk_values.numpy(), topk_indices.numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./data/costs")
    parser.add_argument("--policy", default="OTDF", help='policy to use, support OTDF')
    parser.add_argument("--env", default="halfcheetah")
    parser.add_argument("--seed", default=0, type=int)            
    parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--ot-metric", default='cosine', type=str)     # metric used in optimal transport
    parser.add_argument("--mmd-metric", default='rbf', type=str)         # metric used in MMD
    parser.add_argument('--srctype', default='medium', type=str)
    parser.add_argument("--tartype", default='medium', type=str)
    parser.add_argument("--steps", default=1e5, type=int)
    parser.add_argument("--proportion", default=1.0, type=float, help='proportion of the source dataset to process, 1.0 means all data')
    parser.add_argument("--seq_len", default=20, type=int, help='window size for MMD calculation')
    parser.add_argument("--ot_method", default='sinkhorn', type=str, help='OT method to use, e.g., sinkhorn_log, sinkhorn')
    parser.add_argument("--ot_reg", default=0.1, type=float, help='regularization parameter for OT')
    parser.add_argument("--batch_size", default=10000, type=int, help='batch size for OT calculation')
    parser.add_argument("--mmd_step_type", default="state", type=str, help='type of step to use in MMD calculations')
    parser.add_argument("--ot_step_type", default="all", type=str, help='type of step to use in OT calculations')    
    args = parser.parse_args()

    with open(f"{str(Path(__file__).parent.absolute())}/config/{args.env.replace('-', '_')}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("------------------------------------------------------------")
    
    outdir = args.dir + '/' + args.env + '-srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype
    if args.proportion != 1.0:
        outdir += f'-proportion-{args.proportion}'
    # outdir = args.dir + '/' + args.env + '-' + args.srctype + '-to-' + args.tartype

    print("Arguments:")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    
    if '_' in args.env:
        args.env = args.env.replace('_', '-')
    
    # train env
    src_env_name = args.env.split('-')[0] + '-' + args.srctype + '-v2'
    src_env = gym.make(src_env_name)
    src_env.reset(seed=args.seed)
    # test env
    tar_env = call_env(config['tar_env_config'])
    tar_env.reset(seed=args.seed)
    # eval env
    src_eval_env = copy.deepcopy(src_env)
    src_eval_env.reset(seed=args.seed + 100)
    tar_eval_env = copy.deepcopy(tar_env)
    tar_eval_env.reset(seed=args.seed + 100)

    # seed all
    src_env.action_space.seed(args.seed)
    tar_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    state_dim = src_env.observation_space.shape[0]
    action_dim = src_env.action_space.shape[0] 
    max_action = float(src_env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config['ot_metric'] = args.ot_metric
    config['mmd_metric'] = args.mmd_metric

    config.update({
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
    })

    # We set the value of seq_len in filter_data.sh
    seq_len = args.seq_len

    # src_replay_buffer = SequenceReplayBuffer(state_dim, action_dim, device, max_size=1000000, seq_len=config['seq_len'])
    # tar_replay_buffer = SequenceReplayBuffer(state_dim, action_dim, device, max_size=1000000, seq_len=config['seq_len'])

    # load offline datasets
    src_dataset = call_d4rl_dataset(src_env_name)
    tar_dataset = call_tar_dataset('data/target_dataset', args.env, args.tartype)

    src_buffer = TrajectoryBuffer(src_dataset, seq_len=seq_len, padding='left')
    tar_buffer = TrajectoryBuffer(tar_dataset, seq_len=seq_len, padding='left')

    num_trajectories = int(args.proportion * len(src_buffer.observations))

    print("=" * 60)
    print(f"Starting data filtering process:")
    print(f"  Source environment: {src_env_name}")
    print(f"  Target environment: {args.env}-{args.tartype}")
    print(f"  Processing {num_trajectories} trajectories ({args.proportion*100:.1f}% of source data)")
    print(f"  Sequence length: {seq_len}")
    print(f"  OT metric: {config['ot_metric']}, MMD metric: {config['mmd_metric']}")
    print("=" * 60)

    ot_cost, mmd_cost = filter_dataset(
        src_buffer, 
        tar_buffer, 
        seq_len, 
        config['ot_metric'], 
        config['mmd_metric'], 
        num_trajectories=num_trajectories, 
        steps=args.steps, 
        ot_method=args.ot_method, 
        ot_reg=args.ot_reg, 
        batch_size=args.batch_size,
        mmd_step_type=args.mmd_step_type,
        ot_step_type=args.ot_step_type
    )

    # flip signs of ot_cost and mmd_cost so that higher values indicate better matches
    ot_cost = -ot_cost
    mmd_cost = -mmd_cost

    ot_cost_masked = ot_cost[src_buffer.masks[:num_trajectories, :] == 1]
    mmd_cost_masked = mmd_cost[src_buffer.masks[:num_trajectories, :] == 1]

    print("=" * 60)
    print("Data filtering completed successfully!")
    print(f"  OT cost range: [{ot_cost_masked.min():.4f}, {ot_cost_masked.max():.4f}]")
    print(f"  MMD cost range: [{mmd_cost_masked.min():.4f}, {mmd_cost_masked.max():.4f}]")
    print("=" * 60)


    print('done')
    replay_dataset = {
        'ot_cost': ot_cost,
        'mmd_cost': mmd_cost,
    }

    # top_k = 100

    # top_k_ot_cost, top_k_ot_indices = get_topk_values(ot_cost, k=top_k)
    # top_k_mmd_cost, top_k_mmd_indices = get_topk_values(mmd_cost, k=top_k)

    # for i in range(top_k):
    #     traj_len = src_buffer.traj_lens[top_k_ot_indices[i]]
    #     traj_idx, step_idx = top_k_ot_indices[i]
    #     if src_buffer.padding == 'left':
    #         step_idx = step_idx + traj_len - src_buffer.seq_len
    #     print(f"Top {i+1} OT Cost: {top_k_ot_cost[i]}, Trajectory Index: {traj_idx}, Step Index: {step_idx}/{traj_len}, MMD Cost: {top_k_mmd_cost[i]}")

    print("Plotting costs...")

    # plot ot_cost and mmd_cost to check if they are correlated
    mask = src_buffer.masks[:num_trajectories, :]
    ot_cost_valid = ot_cost[mask == 1].flatten()
    mmd_cost_valid = mmd_cost[mask == 1].flatten()
    print(f"OT Cost Valid Shape: {ot_cost_valid.shape}, MMD Cost Valid Shape: {mmd_cost_valid.shape}")

    plt.figure(figsize=(9, 16))

    plt.subplot(3, 1, 1)
    plt.scatter(ot_cost_valid, mmd_cost_valid, alpha=0.5, s=0.5)
    plt.title('OT Cost vs MMD Cost')
    plt.xlabel('OT Cost')
    plt.ylabel('MMD Cost')

    # exclude top and bottom 1% of costs for better visualization
    ot_cost_valid = ot_cost_valid[(ot_cost_valid > np.percentile(ot_cost_valid, 1)) & (ot_cost_valid < np.percentile(ot_cost_valid, 99))]
    mmd_cost_valid = mmd_cost_valid[(mmd_cost_valid > np.percentile(mmd_cost_valid, 1)) & (mmd_cost_valid < np.percentile(mmd_cost_valid, 99))]

    plt.subplot(3, 1, 2)
    plt.hist(ot_cost_valid, bins=100, alpha=0.5, label='OT Cost')
    plt.title('Distribution of OT Cost')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')

    plt.subplot(3, 1, 3)
    plt.hist(mmd_cost_valid, bins=100, alpha=0.5, label='MMD Cost')
    plt.title('Distribution of MMD Cost')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(outdir + '_costs.png')

    print(f"Saving costs to {outdir}.hdf5")
    with h5py.File(outdir +  ".hdf5", 'w') as hfile:
        for k in replay_dataset:
            hfile.create_dataset(k, data=replay_dataset[k], compression='gzip')
    print(f"Saved costs to {outdir}.hdf5")


# command to run:
# python filter_data.py --env halfcheetah --srctype medium --tartype kinematic_medium
# python filter_data.py --env hopper --srctype medium --tartype kinematic_medium --proportion 0.05
# python filter_data.py --env walker2d --srctype medium --tartype kinematic_medium --proportion 0.05
