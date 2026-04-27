import numpy as np
import torch

from decision_transformer.misc.utils import compute_gae


def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    with torch.no_grad():
        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)

        state = env.reset()

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = [torch.zeros((1, act_dim), device=device, dtype=torch.float32)]
        rewards = [torch.zeros((1, 1), device=device, dtype=torch.float32)]
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        episode_return, episode_length = 0, 0
        for t in range(max_ep_len):
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                torch.cat(actions, dim=0).to(dtype=torch.float32),
                torch.cat(rewards, dim=0).to(dtype=torch.float32),
                None,
                timesteps=timesteps.to(dtype=torch.long),
            )            
            
            action = action.detach().cpu().numpy()

            state, reward, done, _ = env.step(action)
            
            actions.insert(-1, torch.from_numpy(action).reshape(1, act_dim).to(device))
            rewards.insert(-1, torch.tensor(reward).reshape(1).unsqueeze(0).to(device))

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
#             rewards[-1] = reward

            timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        critic,
        critic_q,
        critic_v,
        command,
        max_ep_len=2000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        relabeling_type='command',
        fixed_timestep=False,
        command_state_normalization=False,
        test_mode=True,
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    # print("state_mean: ", state_mean)
    # print("state_std: ", state_std)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = [torch.zeros((1, act_dim), device=device, dtype=torch.float32)]
    
    initial_value = critic_v.v_mean(states.to(dtype=torch.float32)).detach().cpu().squeeze()

    if relabeling_type == 'command':
        if not command_state_normalization:
            # do not use the normalized state to compute the command
            ep_return = command.select_command(states.to(dtype=torch.float32)).detach()
        elif command_state_normalization:
            # use the normalized state to compute the command
            ep_return = command.select_command(
                (states.to(dtype=torch.float32) - state_mean) / (state_std + 1e-8)).detach()
    elif relabeling_type == 'value':
        ep_return = critic_v.v_mean(states.to(dtype=torch.float32)).detach().cpu().reshape(1)
    elif relabeling_type == 'rtg':
        ep_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    elif relabeling_type == 'adv':
        # need to call compute_gae here but not sure what it is
        ep_return = critic_v.v_mean(states.to(dtype=torch.float32)).detach().cpu().reshape(1)

    if not test_mode and relabeling_type != 'rtg':
        ep_return = torch.cat([ep_return, ep_return / 2.], dim=0).squeeze()

    if len(ep_return) > 1:
        #! choose this way to compute the target return
        # print("ep_return: ", ep_return)
        target_return = ep_return.to(device=device, dtype=torch.float32).unsqueeze(-1)
    else:
        target_return = ep_return.to(device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    rewards = [torch.zeros((1, 1), device=device, dtype=torch.float32)]
    initial_value = critic_v.v_mean(states.to(dtype=torch.float32)).detach().cpu().squeeze()
    values = [torch.tensor(initial_value, device=device, dtype=torch.float32).reshape(1, 1)]
    qs = [torch.zeros((1, 1), device=device, dtype=torch.float32)]

    sim_states = []

    episode_return, episode_length = 0, 0
    with torch.no_grad():
        for t in range(max_ep_len):
            action = model.get_action(
                critic,
                (states.to(dtype=torch.float32) - state_mean) / (state_std + 1e-8),
                torch.cat(actions, dim=0).to(dtype=torch.float32),
                torch.cat(rewards, dim=1).to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                test_mode=test_mode,
            )

            action = action.detach().cpu().numpy().flatten()

            state, reward, done, _ = env.step(action)
            
            actions.insert(-1, torch.from_numpy(action).reshape(1, act_dim).to(device))
            rewards.insert(-1, torch.tensor(reward).reshape(1).unsqueeze(0).to(device))

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)

            action_tensor = torch.from_numpy(action).reshape(1, act_dim).to(device=device, dtype=torch.float32)
            q_value = critic_q.q_mean(cur_state.to(dtype=torch.float32), action_tensor).detach()
            v_value = critic_v.v_mean(cur_state.to(dtype=torch.float32)).detach()

            values.insert(-1, v_value.reshape(1, 1))
            qs.insert(-1, q_value.reshape(1, 1))

            # use the fixed timestep to test the model
            if fixed_timestep:
                timesteps = torch.arange(
                    max(0, model.max_length - t - 1), model.max_length, device=device).reshape(1, -1)
                # timesteps = torch.arange(
                #     0, min(model.max_length, t+1), device=device).reshape(1, -1)

            if relabeling_type == 'command':
                if not command_state_normalization:
                    # not use the normalized state to compute the command
                    pred_return = command.select_command(cur_state.to(dtype=torch.float32)).detach()
                elif command_state_normalization:
                    # use the normalized state to compute the command
                    pred_return = command.select_command((cur_state.to(dtype=torch.float32) - state_mean) / (state_std + 1e-8)).detach()
            elif relabeling_type == 'rtg':
                pred_return = (target_return[:, -1] - reward / scale).reshape(-1, 1)
            elif relabeling_type == 'value':
                pred_return = critic_v.v_mean(cur_state.to(dtype=torch.float32)).detach().reshape(-1, 1)
            elif relabeling_type == 'adv':
                # Convert action back to tensor for Q network
                action_tensor = torch.from_numpy(action).reshape(1, act_dim).to(device=device, dtype=torch.float32)
                q_value = critic_q.q_mean(cur_state.to(dtype=torch.float32), action_tensor).detach()
                v_value = critic_v.v_mean(cur_state.to(dtype=torch.float32)).detach()
                pred_return = (q_value - v_value).reshape(-1, 1)

            if not test_mode and relabeling_type != 'rtg':
                pred_return = torch.cat([pred_return, pred_return], dim=0)

            # print("pred_return shape: ", pred_return.shape)
            # print("target_return shape: ", target_return.shape)
            target_return = torch.cat([target_return, pred_return], dim=1)
            
            # the original timestep
            if not fixed_timestep:
                timesteps = torch.cat(
                    [timesteps,
                    torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                break
    
    actions = torch.cat(actions, dim=0)
    values = torch.cat(values, dim=0)
    qs = torch.cat(qs, dim=0)
    rewards = torch.cat(rewards, dim=0)

    # action jump = |current action - previous action|
    action_jumps = (actions[1:] - actions[:-1])
    action_jump_mean = action_jumps.abs().mean().item()
    q_jumps = (qs[1:] - qs[:-1])
    q_jump_mean = q_jumps.abs().mean().item()
    # td error = r + gamma * V(s') - Q(s, a), gamma = 1
    td_errors = rewards[1:] + values[1:] - qs[:-1]
    td_error_mean = td_errors.abs().mean().item()

    return episode_return, episode_length, action_jump_mean, q_jump_mean, td_error_mean
