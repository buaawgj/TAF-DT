import os
import re
import yaml
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional, Union
import copy

from torch.nn.modules.dropout import Dropout
from pathlib import Path
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence

import gym
from pathlib import Path
import h5py
from tqdm import tqdm
import d4rl


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

def _env_key_from_id(env_id: str) -> str:
    """
    Extract some simplified environment keys from a full environment ID.
    """
    if not env_id:
        return ""
    # 'Name[-something]-v#' -> ['name', 'something']
    m = re.findall(r"([A-Za-z0-9.]+)", env_id)
    # key = m.group(1).lower() if m else env_id.lower()
    if m:
        for iter, obj in enumerate(m):
            m[iter] = obj.lower()
    else:
        return env_id
    return m

def _join_middle(xs, sep="_"):
    return sep.join(xs[1:-1]) if len(xs) >= 3 else ""

def load_config_for_env(src_name: str, tar_name: str) -> dict:
    # extract all alphanumeric parts of the env names
    src_env_key = _env_key_from_id(src_name)
    tar_env_key = _env_key_from_id(tar_name)
    print(f"Source env key: {src_env_key}, target env key: {tar_env_key}")
    
    # get base config path
    if 'gravity' in tar_env_key or 'morph' in tar_env_key:
        term = tar_env_key[1]
        base_path = f'config/{tar_env_key[0]}_{term}.yaml'
    elif 'kinematic' in tar_env_key:
        term = 'kinematic'
        base_path = f'config/{tar_env_key[0]}.yaml'
    else:
        raise ValueError(f"Unknown target environment: {tar_name}")
    
    # get env-specific config path
    if 'gravity' in tar_env_key:
        tar_term = '_'.join(tar_env_key[3:-1]) if len(tar_env_key) > 4 else 'gravity'
    else:
        tar_term = _join_middle(tar_env_key)
    
    if len(src_env_key) >= 3 and len(tar_env_key) >= 3:
        env_path = f'config/{tar_env_key[0]}_{term}_{_join_middle(src_env_key)}_{tar_term}.yaml'

    # load base config and env-specific config, then merge
    def _read_yaml(p):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    base_cfg = _read_yaml(base_path)
    env_cfg = {}
    if os.path.exists(env_path):
        env_cfg  = _read_yaml(env_path)

    # merge two levels of dict
    cfg = _deep_update(base_cfg, env_cfg)
    return cfg

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

@torch.no_grad()
def _broadcast_bounds(low, high, shape, device):
    """Helper: 将动作上下界广播到 [*, action_dim] 形状（不参与梯度）。"""
    if isinstance(low, torch.Tensor):
        low_t  = low.to(device).reshape(1, -1)
        high_t = high.to(device).reshape(1, -1)
    else:
        low_t  = torch.as_tensor(low,  dtype=torch.float32, device=device).reshape(1, -1)
        high_t = torch.as_tensor(high, dtype=torch.float32, device=device).reshape(1, -1)
    # 目标维度为 (N, K, A) 或 (N, A)，这里只做最后一维广播，前面维度由采样时自行扩展
    return low_t, high_t

def concat_tensor_dict(tensor_dicts):
    """
    Concatenate a list of tensor dictionaries along the first dimension.
    """
    if not tensor_dicts:
        return {}

    concatenated = {}
    for key in tensor_dicts[0].keys():
        concatenated[key] = torch.cat([d[key] for d in tensor_dicts], dim=0)
    
    return concatenated

def convert_to_tensor(data, device):
    """
    Convert data to a PyTorch tensor and move it to the specified device.
    """
    if isinstance(data, np.ndarray):
        return torch.tensor(data, device=device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected numpy array or torch tensor.")

def expectile_regression(pred, target, expectile):
    diff = target - pred
    return torch.where(diff > 0, expectile, 1-expectile) * (diff**2)

def make_target(m: nn.Module) -> nn.Module:
    target = copy.deepcopy(m)
    target.requires_grad_(False)
    target.eval()
    return target

def compute_action_weight(distance):
    """
    Compute the weight for the critic loss based on the OT distance.
    The weight is computed as the exponentiation of the normalized distance.
    """
    # need to be implemented
    pass

def discounted_cum_sum(seq, discount, mask=None):
    """Compute the discounted cumulative sum of a sequence.
    If mask is provided, only the masked elements are considered for the cumulative sum.
    The discount factor is applied to the future values."""
    seq = seq.copy()
    if type(mask) is np.ndarray or type(mask) is torch.Tensor:
        for t in reversed(range(len(seq)-1)):
            if mask[t] == True:
                if t < len(seq) - 1 and mask[t+1] == True:
                    seq[t] += discount * seq[t+1]
                else:
                    seq[t] = seq[t]
    else:
        for t in reversed(range(len(seq)-1)):
            seq[t] += discount * seq[t+1]
            
    return seq

def compute_gae(rewards, values, last_v, gamma=0.99, lam=0.97, dim=-1, mask=None, K=None):
    #! this function need to be revised for computing GAE with a mask
    if type(mask) is np.ndarray or type(mask) is torch.Tensor:
        transfered_mask = transfer_mask(mask, K)
    values = np.concatenate([values, last_v], axis=dim)
    seq_len = values.shape[dim]
    deltas = rewards + gamma * np.take(values, np.arange(1, seq_len), dim) - np.take(values, np.arange(0, seq_len-1), dim)
    # 检查计算delta的过程是否会导致reward之类的发生变化
    gae = discounted_cum_sum(deltas, gamma * lam, mask=transfered_mask if mask is not None else None)
    ret = gae + np.take(values, np.arange(0, seq_len-1), dim)
    # gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    return gae, ret

def transfer_mask(mask, K):
    length = mask.shape[0]
    transfer_mask = np.zeros_like(mask, dtype=bool)
    for k in reversed(range(length)):
        if mask[k] == True: 
            transfer_mask[k:k + K] = True 
    
    return transfer_mask

def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0, padding: str = "right"
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr
    # (left, right) padding for each axis
    npad = [(0, 0)] * arr.ndim
    if padding == "right":
        npad[axis] = (0, pad_size)  
    elif padding == "left":
        npad[axis] = (pad_size, 0)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)

def compute_position_ids(K):
    """
    Compute position ids for a sequence of length K.
    The position ids are computed as the cumulative sum of the sequence length.
    """
    position_ids = np.arange(K)
    return position_ids.reshape(1, -1)  # reshape to (1, K) for broadcasting


class TrajectoryBuffer:
    def __init__(
        self,
        dataset,
        seq_len,
        max_len=1000,
        discount=1.0,
        return_scale=1.0,
        padding="right"
    ) -> None:
        converted_dataset = {
            "observations": dataset["observations"].astype(np.float32), 
            "actions": dataset["actions"].astype(np.float32), 
            "rewards": dataset["rewards"].astype(np.float32), 
            "terminals": dataset["terminals"].astype(np.float32), 
            "next_observations": dataset["next_observations"].astype(np.float32)
        }
        traj, traj_lens = [], []
        self.seq_len = seq_len
        self.discount = discount
        self.return_scale = return_scale
        self.padding = padding
        traj_start = 0
        for i in range(dataset["rewards"].shape[0]):
            if dataset["terminals"][i] == 1.0:
                episode_data = {k: v[traj_start:i+1] for k, v in converted_dataset.items()}
                episode_data["returns"] = discounted_cum_sum(episode_data["rewards"], discount=discount) * self.return_scale
                traj.append(episode_data)
                traj_lens.append(i+1-traj_start)
                traj_start = i+1
        # last episode
        if traj_start < dataset["rewards"].shape[0]:
            episode_data = {k: v[traj_start:] for k, v in converted_dataset.items()}
            episode_data["returns"] = discounted_cum_sum(episode_data["rewards"], discount=discount) * self.return_scale
            traj.append(episode_data)
            traj_lens.append(dataset["rewards"].shape[0] - traj_start)
        self.traj_lens = np.array(traj_lens, dtype=np.int32)
        self.size = self.traj_lens.sum()
        self.traj_num = len(self.traj_lens)
        
        # pad trajs to have the same mask len
        self.max_len = max(max_len, self.traj_lens.max() + self.seq_len - 1)  # this is for the convenience of sampling
        for i_traj in range(self.traj_num):
            this_len = self.traj_lens[i_traj]
            for _key, _value in traj[i_traj].items():
                traj[i_traj][_key] = pad_along_axis(_value, pad_to=self.max_len, padding=self.padding)
            if padding == "right":
                traj[i_traj]["masks"] = np.hstack([np.ones(this_len), np.zeros(self.max_len-this_len)])
            elif padding == "left":
                traj[i_traj]["masks"] = np.hstack([np.zeros(self.max_len-this_len), np.ones(this_len)])
        
        # register all entries
        self.observations = np.asarray([t["observations"] for t in traj])
        self.actions = np.asarray([t["actions"] for t in traj])
        self.rewards = np.asarray([t["rewards"] for t in traj])
        self.terminals = np.asarray([t["terminals"] for t in traj])
        self.returns = np.asarray([t["returns"] for t in traj])
        self.next_observations = np.asarray([t["next_observations"] for t in traj])
        self.masks = np.asarray([t["masks"] for t in traj])
        self.sample_mask = self.masks.copy()
        self.values = np.zeros_like(self.rewards)
        self.agent_advs = np.zeros_like(self.rewards)

        timesteps = np.zeros((len(traj_lens), self.max_len), dtype=np.int32)
        if padding == "right":
            timesteps = np.arange(self.max_len)[None, :] * self.masks
        elif padding == "left":
            for i in range(self.traj_num):
                this_len = self.traj_lens[i]
                timesteps[i, -this_len:] = np.arange(this_len)

        self.timesteps = timesteps

        # compute mean and std of states
        self.obs_mean = converted_dataset["observations"].mean(axis=0)
        self.obs_std = converted_dataset["observations"].std(axis=0)
         
    def __len__(self):
        return self.size

    def _prepare_sample(self, traj_idx, start_idx):
        return {
            "observations": self.observations[traj_idx, start_idx:start_idx+self.seq_len], 
            "actions": self.actions[traj_idx, start_idx:start_idx+self.seq_len], 
            "rewards": self.rewards[traj_idx, start_idx:start_idx+self.seq_len], 
            "terminals": self.terminals[traj_idx, start_idx:start_idx+self.seq_len], 
            "next_observations": self.next_observations[traj_idx, start_idx:start_idx+self.seq_len], 
            "returns": self.returns[traj_idx, start_idx:start_idx+self.seq_len], 
            "agent_advs": self.returns[traj_idx, start_idx:start_idx+self.seq_len],
            "masks": self.masks[traj_idx, start_idx:start_idx+self.seq_len], 
            "timesteps": self.timesteps[traj_idx, start_idx:start_idx+self.seq_len],
            "values": self.values[traj_idx, start_idx:start_idx+self.seq_len], 
            "agent_advs": self.agent_advs[traj_idx, start_idx:start_idx+self.seq_len], 
        }
       
    def __iter__(self):
        """
        Randomly sample trajectories from the buffer.
        """
        traj_sample_probs = self.sample_mask.sum(axis=1) / self.sample_mask.sum()
        step_sample_probs = self.sample_mask / self.sample_mask.sum(axis=1, keepdims=True)
        while True:
            traj_idx = np.random.choice(self.traj_num, p=traj_sample_probs)
            start_idx = np.random.choice(self.max_len, p=step_sample_probs[traj_idx])
            sample = self._prepare_sample(traj_idx, start_idx)
            for k, v in sample.items():
                sample[k] = pad_along_axis(
                    v,
                    pad_to=self.seq_len,
                    axis=0,
                    padding=self.padding,
                    fill_value=0.0 if k != "terminals" else 1.0
                )
            yield sample
            
    def get_batch(self, batch_size: int = 1, seed: int=None):
        """
        Randomly sample a bsatch of trajectories from the buffer.
        """
        if seed is not None:
            np.random.seed(seed)
        batch_data = {}
        traj_sample_probs = self.sample_mask.sum(axis=1) / self.sample_mask.sum()
        row_sum = self.sample_mask.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0  # avoid division by zero
        step_sample_probs = self.sample_mask / row_sum
        for i in range(batch_size):
            traj_idx = np.random.choice(self.traj_num, p=traj_sample_probs)
            start_idx = np.random.choice(self.max_len, p=step_sample_probs[traj_idx])
            sample = self._prepare_sample(traj_idx, start_idx)
            for _key, _value in sample.items():
                if not _key in batch_data:
                    batch_data[_key] = []
                fill_value = 0.0
                if _key == "terminals":
                    fill_value = 1.0
                _value = pad_along_axis(
                    _value,
                    pad_to=self.seq_len,
                    axis=0,
                    padding=self.padding,
                    fill_value=fill_value
                )
                _value = np.expand_dims(_value, axis=0)  
                batch_data[_key].append(_value)
        for _key, _value in batch_data.items():
            batch_data[_key] = np.vstack(_value)
        return batch_data
        

class DataFilteringTrajectoryBuffer(TrajectoryBuffer):
    def __init__(
        self, 
        dataset, 
        seq_len: int, 
        discount: float=1.0, 
        return_scale: float=1.0, 
        adv_scale: Optional[float]=None, 
        lambda_: Optional[float]=None, 
        use_mean_reduce: bool=False, 
        delayed_reward: bool=False, 
        device: Union[str, torch.device]="cpu",
        padding: str="right"
    ) -> None:
        super().__init__(
            dataset=dataset, 
            seq_len=seq_len, 
            discount=discount, 
            return_scale=return_scale,
            padding=padding
        )
        self.lambda_ = lambda_
        self.adv_scale = adv_scale
        self.device = device

        self.values = np.zeros_like(self.rewards)
        self.agent_advs = np.zeros_like(self.rewards)
        self.ot_costs = np.zeros_like(self.rewards)
        self.mmd_costs = np.zeros_like(self.rewards)
        # NOTE: set default costs to -10.0 to avoid selecting padding steps
        # Avoiding padding step selection can be done by masks, this is a fallback and for marking padding steps in cost
        self.default_cost = -10.0  # default cost for padding

        self.is_ssl_pretrain = False
        self.use_mean_reduce = use_mean_reduce
        self.delayed_reward = delayed_reward
        
        if self.delayed_reward:
            total_returns = np.sum(self.rewards, axis=1, keepdims=True)
            total_returns = total_returns / (total_returns.max()) * 50
            self.rewards = np.zeros_like(self.rewards)
            # self.returns = total_returns * self.masks[..., None]  # this may not reconcile with the discounted return
            self.rewards[np.arange(self.traj_num), self.traj_lens - 1] = total_returns[:, 0]

    # NOTE: added prepare_costs to TrajectoryBuffer to prepare costs for the buffer
    def prepare_costs(self, ot_costs=None, mmd_costs=None, default_cost=0.0, proportion=1.0, mask_mode='original'):
        """
        Prepare costs for the buffer.
        If ot_costs or mmd_costs are provided, they will be used to set the costs.
        Otherwise, the costs will be set to the default cost.
        """
        self.label_costs(ot_costs=ot_costs, mmd_costs=mmd_costs, default_cost=default_cost)
        print(f"OT cost minimum = {self.ot_costs[self.masks == 1].min()}, maximum = {self.ot_costs[self.masks == 1].max()}")
        print(f"MMD cost minimum = {self.mmd_costs[self.masks == 1].min()}, maximum = {self.mmd_costs[self.masks == 1].max()}")
        self.normalize_costs(default_cost=default_cost)

        # print("OT cost min and max after normalization:", self.ot_costs[self.masks == 1].min(), self.ot_costs[self.masks == 1].max())
        # print("MMD cost min and max after normalization:", self.mmd_costs[self.masks == 1].min(), self.mmd_costs[self.masks == 1].max())
        if mask_mode == 'original' or mask_mode == 'mmd':
            self.set_sample_mask_by_cost(self.mmd_costs, proportion)
        # elif mask_mode == 'ot':
        else:
            self.set_sample_mask_by_cost(self.ot_costs, proportion=1.0)

        return 

    # NOTE: use new default_cost to avoid padding step selection
    @torch.no_grad()
    def label_costs(self, ot_costs=None, mmd_costs=None, default_cost=-10.0):
        # pad costs to the same shape as rewards

        self.default_cost = default_cost

        if ot_costs is not None:
            self.ot_costs = np.asarray(ot_costs, dtype=np.float32).reshape(len(self.traj_lens), self.max_len, 1)
        else:
            self.ot_costs = np.full(self.ot_costs.shape, default_cost, dtype=np.float32)
        
        if mmd_costs is not None:
            self.mmd_costs = np.asarray(mmd_costs, dtype=np.float32).reshape(len(self.traj_lens), self.max_len, 1)
        else:
            self.mmd_costs = np.full(self.mmd_costs.shape, default_cost, dtype=np.float32)
    
    # NOTE: normalize costs to [-1, 0] range, following OTDF
    def normalize_costs(self, default_cost=None):
        if default_cost is None:
            default_cost = self.default_cost
        ot_costs = self.ot_costs[self.masks == 1].flatten()
        mmd_costs = self.mmd_costs[self.masks == 1].flatten()
        ot_range = (ot_costs.max() - ot_costs.min()) if len(ot_costs) > 0 else 0.0
        mmd_range = (mmd_costs.max() - mmd_costs.min()) if len(mmd_costs) > 0 else 0.0
        if len(ot_costs) > 0 and ot_range > 0.0:
            self.ot_costs = (self.ot_costs - ot_costs.max()) / (ot_costs.max() - ot_costs.min())
        else:
            self.ot_costs = np.full_like(self.ot_costs, default_cost)

        if len(mmd_costs) > 0 and mmd_range > 0.0:
            self.mmd_costs = (self.mmd_costs - mmd_costs.max()) / (mmd_costs.max() - mmd_costs.min())
        else:
            self.mmd_costs = np.full_like(self.mmd_costs, default_cost)
        return self.ot_costs, self.mmd_costs
        
    def _prepare_sample(self, traj_idx, start_idx):
        return {
            "observations": self.observations[traj_idx, start_idx:start_idx+self.seq_len], 
            "actions": self.actions[traj_idx, start_idx:start_idx+self.seq_len], 
            "rewards": self.rewards[traj_idx, start_idx:start_idx+self.seq_len], 
            "terminals": self.terminals[traj_idx, start_idx:start_idx+self.seq_len], 
            "next_observations": self.next_observations[traj_idx, start_idx:start_idx+self.seq_len], 
            "masks": self.masks[traj_idx, start_idx:start_idx+self.seq_len], 
            "timesteps": self.timesteps[traj_idx, start_idx:start_idx+self.seq_len],
            "values": self.values[traj_idx, start_idx:start_idx+self.seq_len], 
            "agent_advs": self.agent_advs[traj_idx, start_idx:start_idx+self.seq_len], 
            "ot_costs": self.ot_costs[traj_idx, start_idx:start_idx+self.seq_len],
            "mmd_costs": self.mmd_costs[traj_idx, start_idx:start_idx+self.seq_len],
            "returns": self.returns[traj_idx, start_idx:start_idx+self.seq_len]
        }

    # NOTE: use proportion instead of percentile to select the top proportion of costs
    def set_sample_mask_by_cost(self, costs, proportion=1.0):
        """
        Set the sample mask based on the given costs and proportion.
        This is used to sample trajectories with different probabilities.

        Args:
            costs (np.ndarray): The costs for each step in each trajectory.
            proportion (float): The proportion of highest costs to be selected as valid for sampling.
                If proportion is 1, all valid costs are considered.
                If proportion is 0, no costs are considered.
        """
        costs = costs.reshape(self.traj_num, self.max_len)  # reshape to (traj_num, max_len)
        valid_costs = costs[self.masks == 1].flatten()
        self.sample_mask = self.masks.copy()
        percentile = 100 - (proportion * 100)
        if len(valid_costs) > 0 and percentile > 0:
            threshold = np.percentile(valid_costs, percentile)
            self.sample_mask *= (costs >= threshold).astype(np.float32)
        print(f"Steps sampled: {self.sample_mask.sum()} / {self.masks.sum()} = {self.sample_mask.sum() / self.masks.sum() * 100:.2f}%")
        
    
    def __iter__(self):
        traj_sample_probs = self.sample_mask.sum(axis=1) / self.sample_mask.sum()
        # NOTE: fix division by zero issue for step_sample_probs
        row_sums = self.sample_mask.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid division by zero
        step_sample_probs = self.sample_mask / row_sums
        while True:
            traj_idx = np.random.choice(self.traj_num, p=traj_sample_probs)
            start_idx = np.random.choice(self.max_len, p=step_sample_probs[traj_idx])
            sample = self._prepare_sample(traj_idx, start_idx)
            for k, v in sample.items():
                sample[k] = pad_along_axis(
                    v,
                    pad_to=self.seq_len,
                    axis=0,
                    padding=self.padding,
                    fill_value=0.0 if k != "terminals" else 1.0
                )
            if self.is_ssl_pretrain:            
                mask_p = np.random.uniform()
                input_mask = np.random.choice([0, 1], p=[mask_p, 1-mask_p], size=[self.seq_len, ])
                while (input_mask * sample["masks"]).sum() == 0:
                    input_mask = np.random.choice([0, 1], p=[mask_p, 1-mask_p], size=[self.seq_len, ])
                prediction_mask = 1 - input_mask
                sample.update({
                    "input_masks": input_mask, 
                    "prediction_masks": prediction_mask
                })
            yield sample
            
            

class SequenceReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6), seq_len=20):
        self.max_size = max_size
        self.seq_len = seq_len
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.rtg = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.timestep = np.zeros((max_size, 1))
        self.cost = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.size = min(self.size + 1, self.max_size)
        prev_ptr = (self.ptr + self.size - 1) % self.size
        if self.size == 1 or (self.not_done[prev_ptr] == 0 and self.not_done[self.ptr] == 1):
            self.timestep[self.ptr] = 0
        else:
            self.timestep[self.ptr] = self.timestep[prev_ptr] + 1
        if self.timestep[self.ptr] == 0:
            # update rtgs for previous episode
            while self.ptr > 0 and self.not_done[(self.ptr - 1) % self.size] == 0:
                prev_ptr = (self.ptr - 1) % self.size
                self.rtg[prev_ptr] = self.reward[prev_ptr] + self.rtg[self.ptr] * self.not_done[self.ptr]
                self.ptr = prev_ptr
        self.ptr = (self.ptr + 1) % self.max_size


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.rtg[ind]).to(self.device),
            torch.FloatTensor(self.timestep[ind]).to(self.device),
            torch.FloatTensor(self.cost[ind]).to(self.device)
        )
    
    def get_sequence(self, start, seq_len=None, padding=True):
        if seq_len is None:
            seq_len = self.seq_len
        """
        Get a sequence starting from the given index.
        The sequence will continue until a terminal state is reached or the maximum sequence length is reached.
        """
        states = []
        actions = []
        next_states = []
        rewards = []
        not_dones = []
        rtgs = []
        timesteps = []
        costs = []

        current_index = start
        while True:
            states.append(self.state[current_index])
            actions.append(self.action[current_index])
            next_states.append(self.next_state[current_index])
            rewards.append(self.reward[current_index])
            not_dones.append(self.not_done[current_index])
            rtgs.append(self.rtg[current_index])
            timesteps.append(self.timestep[current_index])
            costs.append(self.cost[current_index])

            if self.not_done[current_index] == 0 or len(states) >= seq_len or current_index >= self.size - 1:
                # if len(states) 
                #     states.append(self.next_state[current_index])  # append the next state as well
                #     actions.append(self.action[current_index])  # append the action as well
                #     next_states.append(self.next_state[current_index])  # append the next state as well
                #     rewards.append(self.reward[current_index])  # append the reward as well
                #     not_dones.append(self.not_done[current_index])  # append the not_done as well
                #     timesteps.append(self.timestep[current_index])  # append the timestep as well
                #     costs.append(self.cost[current_index])  # append the cost as well
                break
            
            current_index = (current_index + 1) % self.size
    
        if padding and len(states) < seq_len:
            # zero pad on left side
            padding_length = seq_len - len(states)
            states = np.concatenate((np.zeros((padding_length, states[0].shape[0])), states), axis=0)
            actions = np.concatenate((np.zeros((padding_length, actions[0].shape[0])), actions), axis=0)
            next_states = np.concatenate((np.zeros((padding_length, next_states[0].shape[0])), next_states), axis=0)
            rewards = np.concatenate((np.zeros((padding_length, rewards[0].shape[0])), rewards), axis=0)
            not_dones = np.concatenate((np.zeros((padding_length, not_dones[0].shape[0])), not_dones), axis=0)
            rtgs = np.concatenate((np.zeros((padding_length, rtgs[0].shape[0])), rtgs), axis=0)
            timesteps = np.concatenate((np.zeros((padding_length, timesteps[0].shape[0])), timesteps), axis=0)
            costs = np.concatenate((np.zeros((padding_length, costs[0].shape[0])), costs), axis=0)

        # return (
        #     np.array(states),
        #     np.array(actions),
        #     np.array(next_states),
        #     np.array(rewards),
        #     np.array(not_dones),
        #     np.array(timesteps),
        #     np.array(costs)
        # )
        return {
            'observations': np.array(states),
            'actions': np.array(actions),
            'next_observations': np.array(next_states),
            'rewards': np.array(rewards),
            'terminals': 1 - np.array(not_dones).flatten(),
            'not_dones': np.array(not_dones).flatten(),
            'rtgs': np.array(rtgs),
            'timesteps': np.array(timesteps).flatten(),
            'costs': np.array(costs)
        }
    
    def sample_sequence(self):
        start = np.random.randint(0, self.size)
        while self.not_done[start] == 0 or self.not_done[(start + 1) % self.size] == 0:
            start = np.random.randint(0, self.size)
        sequence = self.get_sequence(start)
        return {
            'observations': torch.FloatTensor(sequence['observations']).to(self.device),
            'actions': torch.FloatTensor(sequence['actions']).to(self.device),
            'next_observations': torch.FloatTensor(sequence['next_observations']).to(self.device),
            'rewards': torch.FloatTensor(sequence['rewards']).to(self.device),
            'terminals': torch.FloatTensor(sequence['terminals']).to(self.device),
            'not_dones': torch.FloatTensor(sequence['not_dones']).to(self.device),
            'rtgs': torch.FloatTensor(sequence['rtgs']).to(self.device),
            'timesteps': torch.FloatTensor(sequence['timesteps']).to(self.device),
            'costs': torch.FloatTensor(sequence['costs']).to(self.device)
        }
        
    def sample_sequence_batch(self, batch_size, p_sample):
        indices = np.random.choice(np.arange(self.size), size=batch_size, p=p_sample)

        seq_len = self.seq_len

        states = []
        actions = []
        next_states = []
        rewards = []
        not_dones = []
        rtgs = []
        timesteps = []
        costs = []

        for i in range(batch_size):
            sequence = self.get_sequence(indices[i])
            states.append(sequence['observations'])
            actions.append(sequence['actions'])
            next_states.append(sequence['next_observations'])
            rewards.append(sequence['rewards'])
            not_dones.append(sequence['not_dones'])
            rtgs.append(sequence['rtgs'])
            timesteps.append(sequence['timesteps'])
            costs.append(sequence['costs'])

        states = np.array(states).reshape(batch_size, seq_len, -1)
        actions = np.array(actions).reshape(batch_size, seq_len, -1)
        next_states = np.array(next_states).reshape(batch_size, seq_len, -1)
        rewards = np.array(rewards).reshape(batch_size, seq_len, -1)
        not_dones = np.array(not_dones).reshape(batch_size, seq_len)
        rtgs = np.array(rtgs).reshape(batch_size, seq_len, -1)
        terminals = (1 - not_dones).reshape(batch_size, seq_len, 1)
        timesteps = np.array(timesteps).reshape(batch_size, seq_len)
        costs = np.array(costs).reshape(batch_size, seq_len, -1)

        # return {
        #     'observations': torch.FloatTensor(states).to(self.device),
        #     'actions': torch.FloatTensor(actions).to(self.device),
        #     'next_observations': torch.FloatTensor(next_states).to(self.device),
        #     'rewards': torch.FloatTensor(rewards).to(self.device),
        #     'terminals': torch.FloatTensor(terminals).to(self.device),
        #     'not_dones': torch.FloatTensor(not_dones).to(self.device),
        #     'timesteps': torch.FloatTensor(timesteps).to(self.device),
        #     'costs': torch.FloatTensor(costs).to(self.device)
        # }

        return {
            'observations': states,
            'actions': actions,
            'next_observations': next_states,
            'rewards': rewards,
            'terminals': terminals,
            'not_dones': not_dones,
            'rtgs': rtgs,
            'timesteps': timesteps,
            'costs': costs
        }

    def convert_D4RL(self, dataset):
        N = dataset['rewards'].shape[0]
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(N, 1)
        self.rtg = np.zeros((N, 1))
        terminals = dataset['terminals'].reshape(N, 1)
        if 'timeouts' in dataset:
            timeouts = dataset['timeouts'].reshape(N, 1)
            # Combine terminals and timeouts for episode boundaries
            terminals = np.logical_or(terminals, timeouts).reshape(N, 1)
        self.not_done = 1. - terminals
        self.size = N
        self.timestep = np.zeros((N, 1))
        self.cost = np.zeros((N, 1))
        # Calculate timesteps properly considering both terminals and timeouts
        prev_i = 0
        current_timestep = 0
        for i in range(N):
            if i > 0 and self.not_done[i-1] == 0 and self.not_done[i] == 1:
                # calculate rtg for the previous episode
                self.rtg[prev_i:i] = np.cumsum(self.reward[prev_i:i][::-1])[::-1].reshape(-1, 1)
                # print(f"Episode from {prev_i} to {i} with rtg: {self.rtg[prev_i:i].flatten()}")
                prev_i = i
                current_timestep = 0
            self.timestep[i] = current_timestep
            current_timestep += 1
        # Calculate rtg for the last episode
        if prev_i < N:
            self.rtg[prev_i:N] = np.cumsum(self.reward[prev_i:][::-1])[::-1].reshape(-1, 1)
            # print(f"Last episode from {prev_i} to {N} with rtg: {self.rtg[prev_i:N].flatten()}")

    def preprocess(self, filter_num):
        # get filter_num smallest cost indices
        indices = np.argpartition(self.cost[:self.size].flatten(), filter_num)[:filter_num]

        self.state = self.state[indices]
        self.action = self.action[indices]
        self.next_state = self.next_state[indices]
        self.reward = self.reward[indices]
        self.not_done = self.not_done[indices]
        self.cost = self.cost[indices]
        self.timestep = self.timestep[indices]
        self.size = self.state.shape[0]

    def _get_slice(self, arr, start, end):
        if end < start:
            return np.concatenate((arr[start:], arr[:end]), axis=0)
        else:
            return arr[start:end]

def call_d4rl_dataset(env_name):
    # if '-' in env_name:
    #     env_name = env_name.replace('-', '_')

    # if any(name in env_name for name in ['halfcheetah', 'hopper', 'walker2d']) or env_name.split('_')[0] == 'ant':
    #     make_env_name = env_name.split('_')[0]
    #     env = gym.make(make_env_name + '-medium-v2')
    #     _max_episode_steps = env._max_episode_steps
    # elif 'maze2d' in env_name:
    #     env = gym.make(env_name)
    #     _max_episode_steps = env._max_episode_steps
    # else:
    #     raise NotImplementedError

    env = gym.make(env_name)
    _max_episode_steps = env._max_episode_steps
    
    # dataset = d4rl.qlearning_dataset(env)
    dataset = env.get_dataset()
    
    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    # count how many trajectories are included, ensure that the quantity of trajectories do not exceed number_of_trajectories
    counter = 0
    for i in range(N):
        obs = dataset['observations'][i].astype(np.float32)
        # new_obs = dataset['observations'][i].astype(np.float32)
        if i == N - 1:
            new_obs = dataset['observations'][i].astype(np.float32)
        else:
            new_obs = dataset['observations'][i + 1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        try:
            reward = dataset['rewards'][i].astype(np.float32)[0]
        except:
            reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == _max_episode_steps - 1)

        done_bool = done_bool or final_timestep

        if done_bool:
            counter += 1
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_).reshape(N, -1),
        'actions': np.array(action_).reshape(N, -1),
        'next_observations': np.array(next_obs_).reshape(N, -1),
        'rewards': np.array(reward_).reshape(N, -1),
        'terminals': np.array(done_).reshape(N, -1),
    }

def load_from_hdf5(file_path):
    """
    Load data from an h5py file and return it as a dictionary.
    """
    data_dict = {}
    with h5py.File(file_path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:
                # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
    return data_dict

def call_tar_dataset(dir_name, tar_env_name, tar_datatype):
    if '-' in tar_env_name:
        tar_env_name = tar_env_name.replace('-', '_')

    if any(name in tar_env_name for name in ['halfcheetah', 'hopper', 'walker2d']) or tar_env_name.split('_')[0] == 'ant':
        domain = 'mujoco'
        make_env_name = tar_env_name.split('_')[0]
        env = gym.make(make_env_name + '-medium-v2')
        _max_episode_steps = env._max_episode_steps
    else:
        raise NotImplementedError
    
    tar_dataset_path = Path(__file__).parent.parent.parent / dir_name / (tar_env_name + '_' + str(tar_datatype.replace('-', '_')) + '.hdf5')
    
    dataset = load_from_hdf5(tar_dataset_path)
    
    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    # count how many trajectories are included, ensure that the quantity of trajectories do not exceed number_of_trajectories
    counter = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        try:
            reward = dataset['rewards'][i].astype(np.float32)[0]
        except:
            reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == _max_episode_steps - 1)

        done_bool = done_bool or final_timestep

        if done_bool:
            counter +=1
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_).reshape(N-1, -1),
        'actions': np.array(action_).reshape(N-1, -1),
        'next_observations': np.array(next_obs_).reshape(N-1, -1),
        'rewards': np.array(reward_).reshape(N-1, -1),
        'terminals': np.array(done_).reshape(N-1, -1),
    }