import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_transformer.misc.utils import SequenceReplayBuffer, call_d4rl_dataset, call_tar_dataset
import numpy as np

def get_return_stats(dataset):
    """
    Calculate statistics of returns (sum of rewards over each trajectory) in the dataset.
    Returns:
        - mean_return: Mean of returns
        - std_return: Standard deviation of returns
        - max_return: Maximum return
        - min_return: Minimum return
    """
    returns = []
    current_return = 0.0
    for i in range(len(dataset["rewards"])):
        current_return += dataset["rewards"][i][0]
        if dataset["terminals"][i][0] == 1:  # Terminal state
            returns.append(current_return)
            current_return = 0.0
    if current_return > 0:  # If the last trajectory didn't end with a terminal
        returns.append(current_return)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    max_return = np.max(returns)
    min_return = np.min(returns)
    return mean_return, std_return, max_return, min_return
        

def test_call_d4rl_dataset():
    dataset = call_d4rl_dataset("halfcheetah-medium-v2") # returns a dict with 5 keys: "observations", "actions", "rewards", "next_observations", "terminals"
    for k in dataset:
        v = np.array(dataset[k])
        assert v.ndim == 2, f"Expected 2D array for {k}, got {v.ndim}D"
        assert v.shape[0] > 0, f"Expected non-empty array for {k}, got shape {v.shape}"
        print(f"{k}: shape {v.shape}, dtype {v.dtype}")
        print("First 10 elements:", v[:10])
    assert isinstance(dataset, dict)
    assert len(dataset) == 5  # Should have 5 keys: observations, actions
    assert dataset["observations"].shape[1] == 17  # HalfCheetah state dimension
    assert dataset["next_observations"].shape[1] == 17  # HalfCheetah next state dimension
    assert dataset["actions"].shape[1] == 6  # HalfCheetah
    assert dataset["rewards"].shape[1] == 1  # Rewards should be a single value
    assert dataset["terminals"].shape[1] == 1  # Terminals should
    mean_return, std_return, max_return, min_return = get_return_stats(dataset)
    print(f"Mean return: {mean_return}, Std: {std_return}, Max: {max_return}, Min: {min_return}")

def test_src_buffer_convert_D4RL():
    dataset = call_d4rl_dataset("halfcheetah-medium-v2")
    src_buffer = SequenceReplayBuffer(
        state_dim=dataset["observations"].shape[1],
        action_dim=dataset["actions"].shape[1],
        max_size=1000000,
        device="cpu",
        seq_len=5
    )
    src_buffer.convert_D4RL(dataset)
    assert src_buffer.size == len(dataset["observations"]), "Buffer size does not match dataset size"
    assert src_buffer.state.shape[0] == len(dataset["observations"]), "Observations shape mismatch"
    assert src_buffer.action.shape[0] == len(dataset["actions"]), "Actions shape mismatch"
    assert src_buffer.reward.shape[0] == len(dataset["rewards"]), "Rewards shape mismatch"
    assert src_buffer.next_state.shape[0] == len(dataset["next_observations"]), "Next observations shape mismatch"
    assert src_buffer.not_done.shape[0] == len(dataset["terminals"]), "Terminals shape mismatch"


def test_call_tar_dataset():
    dataset = call_tar_dataset(
        dir_name="data/target_dataset", 
        tar_env_name="halfcheetah_kinematic", 
        tar_datatype="medium"
    )
    for k in dataset:
        v = np.array(dataset[k])
        assert v.ndim == 2, f"Expected 2D array for {k}, got {v.ndim}D"
        assert v.shape[0] > 0, f"Expected non-empty array for {k}, got shape {v.shape}"
        print(f"{k}: shape {v.shape}, dtype {v.dtype}")
        print("First 10 elements:", v[:10])
    reward_mean = dataset["rewards"].mean()
    reward_std = dataset["rewards"].std()
    max_reward = dataset["rewards"].max()
    min_reward = dataset["rewards"].min()
    print(f"Reward mean: {reward_mean}, std: {reward_std}, max: {max_reward}, min: {min_reward}")
    assert isinstance(dataset, dict)
    assert len(dataset) == 5  # Should have 5 keys: observations, actions, rewards, next_observations, terminals
    assert dataset["observations"].shape[1] == 17  # HalfCheetah state dimension
    assert dataset["next_observations"].shape[1] == 17  # HalfCheetah next state dimension
    assert dataset["actions"].shape[1] == 6
    assert dataset["rewards"].shape[1] == 1  # Rewards should be a single value
    assert dataset["terminals"].shape[1] == 1  # Terminals should be a single value
    mean_return, std_return, max_return, min_return = get_return_stats(dataset)
    print(f"Mean return: {mean_return}, Std: {std_return}, Max: {max_return}, Min: {min_return}")

def test_tar_buffer_convert_D4RL():
    dataset = call_tar_dataset(
        dir_name="data/target_dataset", 
        tar_env_name="halfcheetah_kinematic", 
        tar_datatype="medium"
    )
    tar_buffer = SequenceReplayBuffer(
        state_dim=dataset["observations"].shape[1],
        action_dim=dataset["actions"].shape[1],
        max_size=10000,
        device="cpu",
        seq_len=5
    )
    tar_buffer.convert_D4RL(dataset)
    assert isinstance(tar_buffer, SequenceReplayBuffer), "tar_buffer should be an instance of SequenceReplayBuffer"
    assert tar_buffer.state.shape[1] == dataset["observations"].shape[1], "State dimension mismatch"
    assert tar_buffer.action.shape[1] == dataset["actions"].shape[1], "Action dimension mismatch"
    assert tar_buffer.max_size == 10000, "Max size mismatch"
    assert tar_buffer.size == len(dataset["observations"]), "Buffer size does not match dataset size"
    assert tar_buffer.state.shape[0] == len(dataset["observations"]), "Observations shape mismatch"
    assert tar_buffer.action.shape[0] == len(dataset["actions"]), "Actions shape mismatch"
    assert tar_buffer.reward.shape[0] == len(dataset["rewards"]), "Rewards shape mismatch"
    assert tar_buffer.next_state.shape[0] == len(dataset["next_observations"]), "Next observations shape mismatch"
    assert tar_buffer.not_done.shape[0] == len(dataset["terminals"]), "Terminals shape mismatch"

if __name__ == "__main__":
    test_call_d4rl_dataset()
    test_call_tar_dataset()
    test_src_buffer_convert_D4RL()
    test_tar_buffer_convert_D4RL()
    print("All tests passed!")