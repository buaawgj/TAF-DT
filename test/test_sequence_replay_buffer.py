import os 
import sys
import numpy as np
import torch
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_transformer.misc.utils import SequenceReplayBuffer

class DummyDevice:
    def __str__(self):
        return 'cpu'
    def __repr__(self):
        return 'cpu'

def test_add_and_sample():
    state_dim = 3
    action_dim = 2
    buffer = SequenceReplayBuffer(state_dim, action_dim, device='cpu', max_size=100, seq_len=5)
    for i in range(10):
        state = np.ones(state_dim) * i
        action = np.ones(action_dim) * (i+1)
        next_state = np.ones(state_dim) * (i+2)
        reward = i
        done = 0 if i < 9 else 1
        buffer.add(state, action, next_state, reward, done)
    batch = buffer.sample(4)
    assert all(b.shape[0] == 4 for b in batch[:5])
    assert batch[0].shape[1] == state_dim
    assert batch[1].shape[1] == action_dim
    assert batch[2].shape[1] == state_dim
    assert batch[3].shape[1] == 1
    assert batch[4].shape[1] == 1
    assert batch[5].shape[1] == 1
    assert batch[6].shape[1] == 1

def test_get_sequence_padding():
    state_dim = 2
    action_dim = 1
    buffer = SequenceReplayBuffer(state_dim, action_dim, device='cpu', max_size=10, seq_len=5)
    # Add a short episode (3 steps)
    for i in range(3):
        buffer.add(
            np.ones(state_dim)*i, 
            np.ones(action_dim)*i, 
            np.ones(state_dim)*(i+1), 
            i, 
            0 if i < 2 else 1  # last step is done
        )
    results = buffer.get_sequence(0)
    states, actions, next_states, rewards, not_dones, timesteps, costs = results['observations'], results['actions'], results['next_observations'], results['rewards'], results['not_dones'], results['timesteps'], results['costs']
    # Should be padded to length 5
    assert all(len(s) == 5 for s in [states, actions, next_states, rewards, not_dones, timesteps, costs])
    # Check zero padding for the first two steps
    assert np.all(states[:2] == 0)
    assert np.all(actions[:2] == 0)
    assert np.all(next_states[:2] == 0)
    assert np.all(rewards[:2] == 0)
    assert np.all(not_dones[:2] == 0)
    assert np.all(timesteps[:2] == 0)
    assert np.all(costs[:2] == 0)
    # Check 

def test_sample_sequence_batch():
    state_dim = 2
    action_dim = 1
    buffer = SequenceReplayBuffer(state_dim, action_dim, device='cpu', max_size=20, seq_len=3)
    # Add two episodes
    for i in range(6):
        done = 1 if (i+1)%3==0 else 0
        buffer.add(np.ones(state_dim)*i, np.ones(action_dim)*i, np.ones(state_dim)*(i+1), i, done)
    p_sample = buffer.not_done[:buffer.size].flatten()
    p_sample = p_sample / p_sample.sum()  # Normalize probabilities
    batch = buffer.sample_sequence_batch(2, p_sample=p_sample)
    # assert batch[0].shape == (2, 3, state_dim)
    # assert batch[1].shape == (2, 3, action_dim)
    # assert batch[2].shape == (2, 3, state_dim)
    # assert batch[3].shape == (2, 3, 1)
    # assert batch[4].shape == (2, 3, 1)
    # assert batch[5].shape == (2, 3, 1)
    # assert batch[6].shape == (2, 3, 1)
    assert batch['observations'].shape == (2, 3, state_dim)
    assert batch['actions'].shape == (2, 3, action_dim)
    assert batch['next_observations'].shape == (2, 3, state_dim)
    assert batch['rewards'].shape == (2, 3, 1)
    assert batch['not_dones'].shape == (2, 3)
    assert batch['timesteps'].shape == (2, 3)
    assert batch['costs'].shape == (2, 3, 1)

def test_convert_D4RL():
    state_dim = 2
    action_dim = 1
    N = 5
    dataset = {
        'observations': np.random.randn(N, state_dim),
        'actions': np.random.randn(N, action_dim),
        'next_observations': np.random.randn(N, state_dim),
        'rewards': np.random.randn(N),
        'terminals': np.array([0, 0, 0, 0, 1]),  # last one is terminal
    }
    buffer = SequenceReplayBuffer(state_dim, action_dim, device='cpu', max_size=10, seq_len=3)
    buffer.convert_D4RL(dataset)
    assert buffer.state.shape == (N, state_dim)
    assert buffer.action.shape == (N, action_dim)
    assert buffer.next_state.shape == (N, state_dim)
    assert buffer.reward.shape == (N, 1)
    assert buffer.not_done.shape == (N, 1)
    assert buffer.rtg.shape == (N, 1)
    assert buffer.timestep.shape == (N, 1)
    assert buffer.cost.shape == (N, 1)
    # check buffer rtg correctness
    for i in range(buffer.size-1, 0, -1):
        # check that rtg is cumulative sum of rewards
        if buffer.not_done[i] == 0:
            assert buffer.rtg[i] == buffer.reward[i]
        else:
            assert buffer.rtg[i] == buffer.reward[i] + buffer.rtg[i+1]


def test_preprocess():
    state_dim = 2
    action_dim = 1
    buffer = SequenceReplayBuffer(state_dim, action_dim, device='cpu', max_size=10, seq_len=3)
    for i in range(6):
        buffer.add(np.ones(state_dim)*i, np.ones(action_dim)*i, np.ones(state_dim)*(i+1), i, 0)
    buffer.cost[:6] = np.random.permutation(6).reshape(-1, 1)  # Random costs for testing
    # states, actions, next_states, rewards, not_dones, timesteps, costs = buffer.get_sequence(0)
    results = buffer.get_sequence(0)
    states, actions, next_states, rewards, not_dones, timesteps, costs = results['observations'], results['actions'], results['next_observations'], results['rewards'], results['not_dones'], results['timesteps'], results['costs']
    assert states.shape == (3, state_dim)
    assert actions.shape == (3, action_dim)
    assert next_states.shape == (3, state_dim)
    assert rewards.shape == (3, 1)
    assert not_dones.shape == (3,)
    assert timesteps.shape == (3,)
    assert costs.shape == (3, 1)
    print("Before preprocessing:")
    print("States:", states)
    print("Actions:", actions)
    print("Next States:", next_states)
    print("Rewards:", rewards)
    print("Not Dones:", not_dones)
    print("Timesteps:", timesteps)
    print("Costs:", costs)
    buffer.preprocess(3)
    results = buffer.get_sequence(0)
    states, actions, next_states, rewards, not_dones, timesteps, costs = results['observations'], results['actions'], results['next_observations'], results['rewards'], results['not_dones'], results['timesteps'], results['costs']
    print("After preprocessing:")
    print("States:", states)
    print("Actions:", actions)
    print("Next States:", next_states)
    print("Rewards:", rewards)
    print("Not Dones:", not_dones)
    print("Timesteps:", timesteps)
    print("Costs:", costs)
    assert buffer.size == 3
    assert np.all(buffer.cost == np.array([[0], [1], [2]]))


if __name__ == "__main__":
    pytest.main([__file__])
