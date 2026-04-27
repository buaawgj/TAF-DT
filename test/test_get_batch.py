import os
import sys

sys.path.append(str(os.path.dirname(os.path.abspath(__file__)) + '/..'))

import numpy as np

from decision_transformer.misc.utils import call_d4rl_dataset, call_tar_dataset, get_keys, discounted_cum_sum, DataFilteringTrajectoryBuffer

def load_trajectories_from_source(gym_name):

    dataset = call_d4rl_dataset(env_name=gym_name)

    trajectories = []
    curr_traj = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'returns': [],
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
            curr_traj['returns'] = discounted_cum_sum(
                curr_traj['rewards'], discount=1.0
            )
            # print("returns shape:", curr_traj['returns'].shape)
            trajectories.append(curr_traj)
            curr_traj = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'returns': [],
                'next_observations': [],
                'terminals': [],
                'masks': [],
                'timesteps': []
            }
            curr_timestep = 0
    return trajectories

def get_batch_from_trajectories(trajectories, batch_size, max_len, scale, seed=None, sample_full_trajectories=False):
    if seed is not None:
        np.random.seed(seed)
    num_trajectories = len(trajectories)
    state_dim = trajectories[0]['observations'].shape[-1]
    act_dim = trajectories[0]['actions'].shape[-1]
    state_mean = np.mean(np.concatenate([traj['observations'] for traj in trajectories], axis=0), axis=0)
    state_std = np.std(np.concatenate([traj['observations'] for traj in trajectories], axis=0), axis=0)
    max_ep_len = max([len(traj['observations']) for traj in trajectories])

    p_sample = np.asarray([
        len(traj['observations']) for traj in trajectories
    ], dtype=np.float32)
    p_sample /= np.sum(p_sample)  # normalize to sum to 1
    # batch_inds = np.random.choice(
    #     np.arange(num_trajectories),
    #     size=batch_size,
    #     replace=True,
    #     p=p_sample,  # reweights so we sample according to timesteps
    # )

    s, a, r, d, rtg, timesteps, mask, target_a = [], [], [], [], [], [], [], []
    for i in range(batch_size):
        traj_idx = np.random.choice(
            np.arange(num_trajectories),
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )
        traj = trajectories[traj_idx]
        if sample_full_trajectories:
            si = np.random.randint(0, traj['rewards'].shape[0]-max_len-1) 
        else:
            si = np.random.randint(0, traj['rewards'].shape[0] - 1) 

        # print(f"From source: sampling trajectory {i+1}/{batch_size} from trajectory {traj_idx} starting at index {si}")

        # get sequences from dataset
        s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
        a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
        target_a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
        if 'terminals' in traj:
            d.append(traj['terminals'][si:si + max_len].reshape(1, -1, 1))
        else:
            d.append(traj['dones'][si:si + max_len].reshape(1, -1, 1))
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff

        # if variant['reward_tune'] == 'cql_antmaze':
        #     traj_rewards = (traj['rewards']-0.5) * 4.0
        # else:
        traj_rewards = traj['rewards']
        r.append(traj_rewards[si:si + max_len].reshape(1, -1, 1))
        rtg.append(discounted_cum_sum(traj_rewards[si:], discount=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
        if rtg[-1].shape[1] <= s[-1].shape[1]:
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
        
        # padding and state + reward normalization
        tlen = s[-1].shape[1]
        s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
        s[-1] = (s[-1] - state_mean) / state_std
        a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), a[-1]], axis=1)
        r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
        target_a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), target_a[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, max_len - tlen, 1)), d[-1]], axis=1)
        rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) * scale
        timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
        mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

    s = np.concatenate(s, axis=0)
    a = np.concatenate(a, axis=0)
    r = np.concatenate(r, axis=0)
    target_a = np.concatenate(target_a, axis=0)
    d = np.concatenate(d, axis=0)
    rtg = np.concatenate(rtg, axis=0)
    timesteps = np.concatenate(timesteps, axis=0)
    mask = np.concatenate(mask, axis=0)

    # s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    # a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    # r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
    # target_a = torch.from_numpy(np.concatenate(target_a, axis=0)).to(dtype=torch.float32, device=device)
    # d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
    # rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
    # timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    # mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    # return s, a, r, target_a, d, rtg, timesteps, mask
    return {
        'observations': s,
        'actions': a,
        'rewards': r,
        'target_actions': target_a,
        'terminals': d,
        'returns': rtg,
        'timesteps': timesteps,
        'masks': mask,
    }

def get_batch_from_buffer(buffer: DataFilteringTrajectoryBuffer, batch_size, max_len, scale):
    """
    Get a batch of data from the buffer.
    
    :param buffer: The trajectory buffer to sample from.
    :param batch_size: Number of trajectories to sample.
    :param max_len: Maximum length of each trajectory.
    :param scale: Scale factor for return to go.
    :return: Tuple containing states, actions, rewards, target actions, dones, returns to go, timesteps, and masks.
    """
    states_arr = buffer.observations[buffer.masks == 1].reshape(-1, buffer.observations.shape[-1])
    state_mean = np.mean(states_arr, axis=0)
    state_std = np.std(states_arr, axis=0)

    sample_batch = buffer.get_batch(batch_size=batch_size, seed=seed)
    s, a, r, target_a, d, rtg, timesteps, mask = (
        sample_batch['observations'],
        sample_batch['actions'],
        sample_batch['rewards'],
        sample_batch['actions'],
        sample_batch['terminals'] if 'terminals' in sample_batch else sample_batch['dones'],
        sample_batch['returns'],
        sample_batch['timesteps'],
        sample_batch['masks']
    )
    
    # Normalize states

    s = (s - state_mean) / (state_std + 1e-8)
    # append to end of rtg for compatibility with DT
    last_rtg = rtg[:, -1:, :]  - r[:, -1:, :] * scale  # last rtg is the same as last reward
    rtg = np.concatenate([rtg, last_rtg], axis=1)

    return {
        'observations': s,
        'actions': a,
        'rewards': r,
        'target_actions': target_a,
        'terminals': d,
        'returns': rtg,
        'timesteps': timesteps,
        'masks': mask,
    }


if __name__ == "__main__":

    gym_name = 'hopper-medium-v2'
    seed = 42
    scale = 0.001  # scale for return to go

    trajectories = load_trajectories_from_source(gym_name)

    dataset = call_d4rl_dataset(env_name=gym_name)
    buffer = DataFilteringTrajectoryBuffer(
        dataset=dataset,
        seq_len=20,
        padding="left",
        return_scale=0.001,
    )

    # vibe check: check state means, stds and lengths are equal between buffer and trajectories

    buffer_states = buffer.observations[buffer.masks == 1].reshape(-1, buffer.observations.shape[-1])
    buffer_state_mean = np.mean(buffer_states, axis=0)
    buffer_state_std = np.std(buffer_states, axis=0)

    trajectories_states = np.concatenate([traj['observations'] for traj in trajectories], axis=0)
    trajectories_state_mean = np.mean(trajectories_states, axis=0)
    trajectories_state_std = np.std(trajectories_states, axis=0)

    buffer_lens = buffer.traj_lens
    trajectories_lens = np.array([len(traj['observations']) for traj in trajectories])

    try:
        assert np.allclose(buffer_state_mean, trajectories_state_mean), "State means do not match!"
        assert np.allclose(buffer_state_std, trajectories_state_std), "State stds do not match!"
        assert np.allclose(buffer_lens, trajectories_lens), "Trajectory lengths do not match!"
    except AssertionError as e:
        print(e)
        print("Buffer state mean:", buffer_state_mean)
        print("Trajectories state mean:", trajectories_state_mean)
        print("Buffer state std:", buffer_state_std)
        print("Trajectories state std:", trajectories_state_std)
        print("Buffer lengths:", buffer_lens)
        print("Trajectories lengths:", trajectories_lens)
        sys.exit(1)
    print("Buffer and trajectories state means, stds and lengths match!")

    seeds_list = [seed + i for i in range(10)]
    batch_size = 16

    for seed in seeds_list:
        np.random.seed(seed)
        traj_batch = get_batch_from_trajectories(
            trajectories=trajectories,
            batch_size=batch_size,
            max_len=20,
            scale=scale
        )

        buffer_batch = get_batch_from_buffer(
            buffer=buffer,
            batch_size=batch_size,
            max_len=20,
            scale=scale
        )

        for key in traj_batch:
            try:
                if key in buffer_batch:
                    assert np.allclose(traj_batch[key], buffer_batch[key]), f"Batch mismatch for key: {key}"
                else:
                    print(f"Key {key} not found in buffer batch")
            except AssertionError as e:
                print(e)
                print(f"Traj batch {key} shape: {traj_batch[key].shape}, Buffer batch {key} shape: {buffer_batch[key].shape}")
                # check which location is different
                for i in range(traj_batch[key].shape[0]):
                    diff = set(np.where(traj_batch[key][i] != buffer_batch[key][i])[0])
                    if len(diff) > 0:
                        print(f"Difference found at index {i} for key {key}: {diff}")
                        print("Traj batch:", traj_batch[key][i])
                        print("Buffer batch:", buffer_batch[key][i])


    print(f"Loaded {len(trajectories)} trajectories from {gym_name}")
    # print("Example trajectory:", trajectories[0])
    # print("Keys in trajectory:", trajectories[0].keys())


