# import robohive
import d4rl
import gym
import numpy as np
import torch
import yaml

import argparse
import pickle
import random
import sys
import os
import pathlib
import time

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg
# trainer: trainer for offline training with source dataset only
# qt_trainer: trainer for off-dynamics training with source and target datasets
from decision_transformer.training.trainer import Trainer
from decision_transformer.models.ql_DT import (
    DecisionTransformer, 
    Critic, 
    Value, 
    SingleCritic,
    VAE_Policy,
)
from decision_transformer.misc.command import InSampleMaxCommand, ConstantCommand
import envs.infos
from envs.common import call_env
from decision_transformer.misc.utils import (
    call_d4rl_dataset,
    call_tar_dataset,
    load_from_hdf5,
    discounted_cum_sum,
    TrajectoryBuffer,
    DataFilteringTrajectoryBuffer
)
from torch.utils.tensorboard import SummaryWriter
def save_checkpoint(state,name):
  filename =name
  torch.save(state, filename)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')

    src_env_name, src_datatype = variant['env'], variant['dataset']
    tar_env_name, tar_datatype = variant['tar_env'], variant['tar_dataset']

    # NOTE: if target environment is not specified, use source environment
    if tar_env_name is None:
        tar_env_name = src_env_name
    # NOTE: if target datatype is not specified, use source datatype
    if tar_datatype is None:
        tar_datatype = src_datatype
    seed = variant['seed']
    group_name = f'{exp_prefix}-{src_env_name}-{src_datatype}-to-{tar_env_name}-{tar_datatype}'
    print(f"Experiment group name: {group_name}")
    timestr = time.strftime("%y%m%d-%H%M%S")
    exp_prefix = f'{group_name}-{seed}-{timestr}'


    # get parent directory of this file and go to env/(env_name).yaml
    with open(os.path.join(os.path.dirname(__file__), 'config', f'{tar_env_name}.yaml'), 'r') as f:
        tar_env_config = yaml.safe_load(f).get('tar_env_config', {})

    # TODO: refactor to include in config file
    dversion = 2
    scale = 1000.
    # env_targets = tar_env_config.get('env_targets', [5000, 4000, 2500]) # arbitrary default targets
    if 'hopper' in tar_env_name:
        dversion = 2
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif 'halfcheetah' in tar_env_name:
        dversion = 2
        env_targets = [12000, 9000, 6000]
        scale = 1000.
    elif 'ant' in tar_env_name:
        dversion = 2
        env_targets = [5000, 4000, 2500]
        scale = 1000.
    elif 'walker2d' in tar_env_name:
        dversion = 2
        env_targets = [5000, 4000, 2500]
        scale = 1000.
    elif 'reacher2d' in tar_env_name:
        # from decision_transformer.envs.reacher_2d import Reacher2dEnv
        # env = Reacher2dEnv()
        env_targets = [76, 40]
        scale = 10.
        dversion = 2
    elif 'pen' in tar_env_name:
        dversion = 1
        env_targets = [12000, 6000]
        scale = 1000.
    elif 'hammer' in tar_env_name:
        dversion = 1
        env_targets = [12000, 6000, 3000]
        scale = 1000.
    elif 'door' in tar_env_name:
        dversion = 1
        env_targets = [2000, 1000, 500]
        scale = 100.
    elif 'relocate' in tar_env_name:
        dversion = 1
        env_targets = [3000, 1000]
        scale = 1000.
        dversion = 1
    elif 'kitchen' in tar_env_name:
        dversion = 0
        env_targets = [500, 250]
        scale = 100.
    elif 'maze2d' in tar_env_name:
        if 'open' in tar_datatype:
            dversion = 0
        else:
            dversion = 1
        env_targets = [300, 200, 150,  100, 50, 20]
        scale = 10.
    elif 'antmaze' in tar_env_name:
        dversion = 0
        env_targets = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
        scale = 1.
    else:
        raise NotImplementedError
    
    # NOTE: create source and target environments, using D4RL as source environments and OTDF datasets as target environments
    src_gym_name = f'{src_env_name}-{src_datatype}-v{dversion}'
    tar_gym_name = f'{tar_env_name}-{tar_datatype}-v{dversion}'


    print("=" * 70)
    print(f"Starting QT experiment: {group_name}")
    print(f"  Source: {src_gym_name}")
    print(f"  Target: {tar_gym_name}")
    print(f"  Seed: {seed}")
    print(f"  Max iterations: {variant['max_iters']}")
    print(f"  Steps per iteration: {variant['num_steps_per_iter']}")
    print("=" * 70)
    if 'reacher' in tar_gym_name:
        tar_gym_name = 'Reacher2d-v4'
    src_env = gym.make(src_gym_name)

    if tar_gym_name == src_gym_name:
        tar_env = src_env
    else:
        if 'gravity' in tar_datatype or 'morph' in tar_datatype or 'kinematic' in tar_datatype:
            term = tar_datatype.split('_')[0]
            tar_env_config_path = f'config/{tar_env_name}_{term}.yaml'
        else:
            raise ValueError(f"Unknown target datatype: {tar_datatype}")
        with open(tar_env_config_path, 'r') as f:
            tar_config = yaml.safe_load(f)
        tar_env = call_env(tar_config['tar_env_config'])

    print(f"src_gym_name: {src_gym_name}")
    print(f"tar_gym_name: {tar_gym_name}")

    # NOTE: get ref min and max scores for the target environment

    if tar_gym_name in d4rl.infos.REF_MIN_SCORE:
        ref_min_score = d4rl.infos.REF_MIN_SCORE[tar_gym_name]
        ref_max_score = d4rl.infos.REF_MAX_SCORE[tar_gym_name]
    else:
        if 'gravity' in tar_datatype:
            tar_ref_score_name = f"{tar_env_name}_gravity_0.5"
        elif 'morph' in tar_datatype:
            tar_ref_score_name = f"{tar_env_name}_morph"
        elif 'kinematic' in tar_datatype:
            tar_ref_score_name = f"{tar_env_name}_kinematic"
        ref_min_score = envs.infos.REF_MIN_SCORE.get(tar_ref_score_name, -np.inf)
        ref_max_score = envs.infos.REF_MAX_SCORE.get(tar_ref_score_name, np.inf)

    print("ref_min_score: ", ref_min_score)
    print("ref_max_score: ", ref_max_score)
    print(f"Using environment targets: {env_targets}")

    # end of new creating environments code

    if 'spec' in tar_env.__dict__:
        max_ep_len = tar_env.spec.max_episode_steps
    else:
        max_ep_len = tar_env._max_episode_steps if hasattr(tar_env, '_max_episode_steps') else 1000
    
    if variant['scale'] is not None:
        scale = variant['scale']
    
    variant['max_ep_len'] = max_ep_len
    variant['env_targets'] = env_targets
    variant['scale'] = scale
    if variant['test_scale'] is None:
        variant['test_scale'] = scale

    if not os.path.exists(os.path.join(variant['save_path'], exp_prefix)):
        pathlib.Path(
        args.save_path +
        exp_prefix).mkdir(
        parents=True,
        exist_ok=True)

    mode = variant.get('mode', 'normal')

    tar_env.reset(seed=variant['seed'])
    set_seed(variant['seed'])

    state_dim = tar_env.observation_space.shape[0]
    act_dim = tar_env.action_space.shape[0]

    print("Environment setup completed.")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {act_dim}")
    print(f"  Max episode length: {max_ep_len}")



    src_dataset = call_d4rl_dataset(env_name=src_gym_name)

    if any([name in tar_datatype for name in ['gravity', 'kinematic', 'morph']]):
        tar_dataset = call_tar_dataset(
            dir_name="data/target_dataset",
            tar_env_name=tar_env_name,
            tar_datatype=tar_datatype
        )
    else:
        tar_dataset = call_d4rl_dataset(env_name=tar_gym_name)

    state_dim = src_dataset['observations'].shape[1]
    action_dim = src_dataset['actions'].shape[1]
    max_action = float(src_env.action_space.high[0])
    min_action = -max_action

    src_buffer = DataFilteringTrajectoryBuffer(
        dataset=src_dataset,
        seq_len=variant['K'],
        padding="left",
        return_scale=1./scale
    )
    
    # print("Observation mean of the source buffer: {}".format(src_buffer.obs_mean))
    # print("Observation std of the source buffer: {}".format(src_buffer.obs_std))

    # NOTE: load costs from hdf5 file if exists, else sets costs to random values between 0 and 1
    if variant['cost_dir'] is not None:
        cost_dir = variant['cost_dir']
    else:
        cost_dir = f"data/new_costs_{variant['K']}"
    cost_path = f"{cost_dir}/{src_env_name}-srcdatatype-{src_datatype}-tardatatype-{tar_datatype}.hdf5"

    ot_costs = None
    mmd_costs = None
    if os.path.exists(cost_path):
        costs = load_from_hdf5(cost_path)
        ot_costs = costs['ot_cost'].reshape(src_buffer.ot_costs.shape)
        mmd_costs = costs['mmd_cost'].reshape(src_buffer.mmd_costs.shape)
    else:
        print(f"Warning: No costs found at {cost_path}. Setting small random costs for sampling.")
        ot_costs = np.random.normal(loc=0.0, scale=0.01, size=src_buffer.ot_costs.shape)
        mmd_costs = np.random.normal(loc=0.0, scale=0.01, size=src_buffer.mmd_costs.shape)
        # raise FileNotFoundError(
        #     f"Cost file {cost_path} not found. Please ensure the costs are generated and saved correctly."
        # )


    # check count of non-zero rewards


    # NOTE: prepare costs for the source buffer by either loading data or setting a default value
    src_buffer.prepare_costs(
        ot_costs=ot_costs,
        mmd_costs=mmd_costs,
        proportion=variant.get('proportion', 1.0),
        mask_mode=variant["loss_weighting_type"],
    )

    # check count of non-zero rewards after preparing costs

    # end of cost loading code

    # NOTE: old code formatting trajectories in a list is gone, now TrajectoryBuffer handles it

    tar_buffer = TrajectoryBuffer(
        dataset=tar_dataset,
        seq_len=variant['K'],
        padding="left",
        return_scale=1./scale,
        max_len = src_buffer.max_len,
    )
    
    # print("Observation mean of the target buffer: {}".format(tar_buffer.obs_mean))
    # print("Observation std of the target buffer: {}".format(tar_buffer.obs_std))

    # check count of non-zero rewards after preparing costs

    traj_lens = src_buffer.traj_lens
    num_timesteps = np.sum(traj_lens)
    returns = np.sum(src_buffer.rewards[src_buffer.masks == 1], axis=1)  # sum rewards for valid steps

    # only train on top pct_traj trajectories (for %BC experiment)
    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    num_trajectories = len(src_buffer.traj_lens)

    # NOTE: special sampling case for hopper medium implemented with sample mask instead of changing np.random.choice
    if 'hopper-medium' in src_gym_name:
        if src_buffer.padding == 'left':
            src_buffer.sample_mask[:, -K-1:] = 0  # don't sample last K+1 steps
            tar_buffer.sample_mask[:, -K-1:] = 0  # don't sample last K+1 steps
        else:
            for i in range(src_buffer.traj_lens.shape[0]):
                src_buffer.sample_mask[i, src_buffer.traj_lens[i] - K - 1:] = 0  # don't sample last K+1 steps
                tar_buffer.sample_mask[i, tar_buffer.traj_lens[i] - K - 1:] = 0  # don't sample last K+1 steps

    writer = SummaryWriter(os.path.join(variant['save_path'], exp_prefix))

    start_text = (
        '=' * 50 + '\n'
        f'Starting new experiment: {tar_gym_name}\n'
        f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found\n'
        f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}\n'
        f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}\n'
        '=' * 50
    )
    writer.add_text('summary/experiment_info', start_text)

    # NOTE: Old get_batch code that was here is removed, replaced by get_batch in TrajectoryBuffer

    def eval_episodes(target_rew):
        def fn(model, critic, critic_q, critic_v, command, test_mode=False):
            returns, lengths = [], []
            action_jumps, q_jumps, td_error_means = [], [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    # NOTE: use source buffer means and stds for evaluation
                    ret, length, action_jump, q_jump, td_error_mean = evaluate_episode_rtg(
                        tar_env,
                        state_dim,
                        act_dim,
                        model,
                        critic,
                        critic_q,
                        critic_v,
                        command,
                        max_ep_len=max_ep_len,
                        scale=variant['test_scale'],
                        target_return=[t/variant['test_scale'] for t in target_rew],
                        mode=mode,
                        relabeling_type=variant['relabeling_type'],
                        state_mean=tar_buffer.obs_mean, 
                        state_std=tar_buffer.obs_std,
                        fixed_timestep=variant['fixed_timestep'],
                        command_state_normalization=variant['command_state_normalization'],
                        test_mode=test_mode,
                        device=device,
                    )
                returns.append(ret)
                lengths.append(length)
                action_jumps.append(action_jump)
                q_jumps.append(q_jump)
                td_error_means.append(td_error_mean)
            # NOTE: manually calculate normalized score
            if ref_max_score == np.inf or ref_min_score == -np.inf or ref_max_score == ref_min_score:
                normalized_score = np.mean(returns) / scale
            else:
                normalized_score = (np.mean(returns) - ref_min_score) / (ref_max_score - ref_min_score)
            return {
                f'target_return_mean': np.mean(returns),
                f'target_return_std': np.std(returns),
                f'target_length_mean': np.mean(lengths),
                f'target_length_std': np.std(lengths),
                f'target_action_jump_mean': np.mean(action_jumps),
                f'target_action_jump_std': np.std(action_jumps),
                f'target_q_jump_mean': np.mean(q_jumps),
                f'target_q_jump_std': np.std(q_jumps),
                f'target_td_error_mean': np.mean(td_error_means),
                f'target_td_error_std': np.std(td_error_means),
                f'target_normalized_score': normalized_score,
            }
        return fn

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4*variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_ctx=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
        scale=scale,
        sar=variant['sar'],
        rtg_no_q=variant['rtg_no_q'],
        infer_no_q=variant['infer_no_q']
    )
    vae_policy = VAE_Policy(
        state_dim=state_dim, 
        action_dim=act_dim,
        latent_dim=2*act_dim,
        max_action=max_action,
        hidden_dim=750,
        device=device,
    )
    critic = Critic(state_dim, act_dim, hidden_dim=variant['embed_dim'])
    # use the Critic class to create the critic q
    critic_q = Critic(state_dim, act_dim, hidden_dim=variant['embed_dim'])
    # use the Value class to create the critic_v
    critic_v = Value(state_dim, hidden_dim=variant['embed_dim'])

    model = model.to(device=device)
    critic = critic.to(device=device)
    critic_q = critic_q.to(device=device)
    critic_v = critic_v.to(device=device)
    vae_policy = vae_policy.to(device=device)
    
    if variant['command_type'] == "in_sample_max":
        command_backend = SingleCritic(
            state_dim=state_dim, 
            hidden_dim=variant['embed_dim'], 
        ).to(device)
        command = InSampleMaxCommand(
            command_module=command_backend, 
            is_agent=True, 
            expectile=variant['expectile'], 
            enhance=variant['enhance'], 
            device=device
        ).to(device)
        command.configure_optimizers(variant['learning_rate'], command_weight_decay=variant['command_weight_decay'])
    elif variant['command_type'] == "constant":
        command = ConstantCommand(
            init=0, 
            polyak=0.995, 
            device=device, 
        ).to(device)

    # NOTE: use new arguments present in adv_rtg, except command modules as not used in tests replicating QT
    # Also added device argument to Trainer to assign tensors to the correct device inside trainer rather than with a function
    trainer = Trainer(
        model=model,
        vae_policy=vae_policy,
        task=group_name, 
        critic=critic,
        critic_v=critic_v,
        critic_q=critic_q,
        command=command,
        command_type=variant['command_type'],
        batch_size=batch_size,
        pretrain_batch_size=batch_size,
        tau=variant['tau'],
        discount=variant['discount'],
        src_buffer=src_buffer,
        tar_buffer=tar_buffer,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
        lambda_=variant['lambda'],
        adv_scale=variant['adv_scale'],
        eval_fns=[eval_episodes(env_targets)],
        max_q_backup=variant['max_q_backup'],
        eta=variant['eta'],
        eta2=variant['eta2'],
        ema_decay=0.995,
        step_start_ema=1000,
        update_ema_every=5,
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
        lr_decay=variant['lr_decay'],
        lr_maxt=variant['max_iters'],
        lr_min=variant['lr_min'],
        grad_norm=variant['grad_norm'],
        scale=scale,
        k_rewards=variant['k_rewards'],
        use_discount=variant['use_discount'],
        proportion=variant['proportion'],
        load_path=variant['load_path'],  # path for load the saved critic q and v models and rtg network
        save_path=variant['save_path'],  # path for saving the q, v and rtg models
        v_target=variant['v_target'],    # the target v network for computing the adv
        iql_tau=variant['iql_tau'],      # the parameter for the expectile regression
        use_mean_reduce=variant['use_mean_reduce'],  # the parameter for computing the new adv rtg
        relabel_adv=variant['relabel_adv'],  # whether to relabel the adv rtg
        load_command_path=variant['load_command_path'],  # path to load the command network, if None, will not load
        fixed_timestep=variant['fixed_timestep'], # whether to use fixed timestep for evaluation
        rescale_reward=variant['rescale_reward'],  # whether to rescale the reward by the ot weight
        ot_filter=variant['ot_filter'],  # whether to filter the ot costs by masks
        ot_proportion=variant['ot_proportion'], # the ot filtering proportion of the source buffer
        use_mmd_weights=variant['use_mmd_weights'],  # whether to only filter the data by the mmd costs
        pi_reg=variant['pi_reg'],  # whether to use pi regularization
        pi_reg_weight=variant['pi_reg_weight'],  # the weight of pi regularization
        vae_beta= variant['vae_beta'],  # beta for vae policy dataset probability calculation
        test_mode=variant['test_mode'],  # whether to run in test mode, if true, will choose the greedy action
        training_normalization=variant['training_normalization'],  # whether to use state normalization for the model training
        use_full_pi_reg=variant['use_full_pi_reg'],  # whether to use full pi regularization
        use_weighted_qloss= variant['use_weighted_qloss'],  # whether to use weighted q loss for the actor
        use_mmd_enhanced_qloss= variant['use_mmd_enhanced_qloss'],  # whether to use mmd enhanced q loss for the actor
        cql_weight=variant['cql_weight'], # weight for the cql loss
        cql_temp=variant['cql_temp'], # temperature for the cql loss
        use_cql_loss=variant['use_cql_loss'], # whether to use the cql loss
        min_action=min_action,
        max_action=max_action,
        # ablation arguments
        loss_weighting_type=variant['loss_weighting_type'], # type of loss weighting, options are 'none', 'mmd', 'ot', 'both'
        relabeling_type=variant['relabeling_type'], # type of relabeling, options are 'rtg', 'adv', 'value', 'command'
        remove_critic_loss=variant['remove_critic_loss'], # whether to remove the critic loss during training
        ablation=variant['ablation'],  # whether to run ablation study
        device=device,
    )
    
    if variant['relabel_adv']:
        trainer.init_step() 
        trainer.pretrain_q_and_v(variant['max_iters']//5, variant['num_steps_per_iter'], writer)
        trainer.init_step()
        trainer.train_rtg_network(
            variant['max_iters']//10,
            variant['num_steps_per_iter'], writer, 
            adv_mean_reduce=variant['adv_mean_reduce'],
            command_state_normalization=variant['command_state_normalization']
        )
        trainer.init_step()
        if variant['pi_reg']:
            trainer.train_vae(variant['max_iters']//5, variant['num_steps_per_iter'], writer)
        # trainer.train_vae(50, 100, writer)
    
    ############## pretrain the critic q and v for relabeling adv ##############
    # trainer.pretrain_q_and_v(variant['max_iters']//5, variant['num_steps_per_iter'], writer)
    
    ############## relabel buffer and train adv network ##############
    # trainer.init_step()
    # trainer.train_rtg_network(variant['num_steps_per_iter'])

    best_ret = -10000
    best_nor_ret = -1000
    best_iter = -1
    # set step = 0
    early_epoch = variant.get('early_epoch', 100) if variant['early_stop'] else variant['max_iters']
    trainer.init_step()
    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(variant['num_steps_per_iter'], writer, iter_num=iter+1)
        #! Do we need to scale up eta here?
        # NOTE: scaling up eta to improve the weight of the bc loss
        trainer.scale_up_eta(variant['lambda'])
        ret = outputs['best_return_mean']
        nor_ret = outputs['best_normalized_score']
        if ret > best_ret or (iter + 1) == early_epoch // 2:
            if (iter + 1) >= early_epoch // 2:
                state = {
                    'epoch': iter+1,
                    'actor': trainer.actor.state_dict(),
                    'critic': trainer.critic_target.state_dict(),
                }
                save_checkpoint(state, os.path.join(variant['save_path'], exp_prefix, 'epoch_{}.pth'.format(iter + 1)))
            best_ret = ret
            best_nor_ret = nor_ret
            best_iter = iter + 1
        message_current = f'Current return mean is {ret}, normalized score is {nor_ret*100}, Iteration {iter + 1}'
        message_best = f'Current best return mean is {best_ret}, normalized score is {best_nor_ret*100}, Iteration {best_iter}'
        print("Current experiment: ", exp_prefix)
        print(message_current)
        print(message_best)
        if writer is not None:
            writer.add_text('summary/current_return', message_current, iter + 1)
            writer.add_text('summary/best_return', message_best, iter + 1)
            
        
        if variant['early_stop'] and iter >= variant['early_epoch']:
            break
    final_message = f'The final best return mean is {best_ret}, normalized score is {best_nor_ret * 100}'
    print(final_message)
    if writer is not None:
        writer.add_text('summary/final_best_return', final_message, iter + 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='gym-experiment')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    # NOTE: tar_env now takes no default value, meaning it will be set to the same value as env if not specified
    parser.add_argument('--tar_env', type=str)
    # NOTE: tar_dataset now takes no default value, meaning it will be set to the same value as dataset if not specified
    parser.add_argument('--tar_dataset', type=str)  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    # NOTE: changed batch size to 128 by default, since 2 batches of 128 is the original batch size of 256
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
    parser.add_argument('--lr_min', type=float, default=0.)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--num_steps_per_iter', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_path', type=str, default='./save/')

    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--eta", default=1.0, type=float)
    parser.add_argument("--eta2", default=1.0, type=float)
    parser.add_argument("--lambda", default=0.0, type=float)  # the default of lambda = 0.
    parser.add_argument("--max_q_backup", action='store_true', default=False)
    parser.add_argument("--lr_decay", action='store_true', default=False)
    parser.add_argument("--grad_norm", default=2.0, type=float)
    parser.add_argument("--early_stop", action='store_true', default=False)
    parser.add_argument("--early_epoch", type=int, default=100)
    parser.add_argument("--k_rewards", action='store_true', default=False)
    parser.add_argument("--use_discount", action='store_true', default=False)
    parser.add_argument("--sar", action='store_true', default=False)
    parser.add_argument("--reward_tune", default='no', type=str)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--test_scale", type=float, default=None)
    parser.add_argument("--rtg_no_q", action='store_true', default=False)
    parser.add_argument("--infer_no_q", action='store_true', default=False)
    # the mmd data filtering parameter
    parser.add_argument("--proportion", type=float, default=1.0)
    parser.add_argument("--ot_proportion", type=float, default=1.0)  # ot filtering proportion of the source buffer
    # path of the saved critic q and v models and rtg model
    parser.add_argument("--load_path", type=str, default=None)
    # this parameter is used to decide if we use target value network to update the q and v networks
    parser.add_argument('--v_target', action='store_true', default=False)
    # the parameter of training v network
    parser.add_argument('--iql_tau', type=float, default=0.5)
    # the parameter of computing the V value for obtaining the adv rtg. If true, the V value will be the average of two V values; 
    # if false, the V value will be the minimal V value of two V values.
    parser.add_argument('--use_mean_reduce', action='store_true', default=False)
    parser.add_argument('--relabel_adv', action='store_true', default=False)
    parser.add_argument('--command_type', type=str, default='in_sample_max')  # choose the type of command
    parser.add_argument('--command_weight_decay', type=float, default=5e-4)   # the parameter of command
    parser.add_argument('--expectile', type=float, default=0.98)  # the expectile parameter of command 
    parser.add_argument('--enhance', action='store_true', default=False)  # the patameter of command
    parser.add_argument('--load_command_path', type=str, default=None) # path to load the command network, if None, will not load
    parser.add_argument('--adv_scale', type=float, default=None) # the scale of the adv rtg, if None, will not scale the adv rtg
    parser.add_argument('--adv_mean_reduce', action='store_true', default=False)  # whether to use mean reduce for adv rtg normalization
    parser.add_argument('--fixed_timestep', action='store_true', default=False) # whether to use fixed timestep for evaluation
    parser.add_argument('--command_state_normalization', action='store_true', default=False)  # whether to use state normalization for command network
    parser.add_argument('--rescale_reward', action='store_true', default=False)  # whether to rescale the reward by the ot weight
    parser.add_argument('--ot_filter', action='store_true', default=False)  # whether to filter the ot costs by masks
    parser.add_argument('--use_mmd_weights', action='store_true', default=False)  # whether to only filter the mmd costs by masks
    parser.add_argument('--pi_reg', action='store_true', default=False)  # whether to use pi regularization
    parser.add_argument('--pi_reg_weight', type=float, default=0.5)  # the weight of pi regularization
    parser.add_argument('--load_vae_path', type=str, default=None)  # path to load the vae policy, if None, will not load
    parser.add_argument('--vae_beta', type=float, default=0.4)  # beta for vae policy dataset probability calculation
    parser.add_argument('--test_mode', action='store_true', default=False)  # whether to run in test mode, if true, will choose the greedy action
    parser.add_argument('--training_normalization', action='store_true', default=False)  # whether to use state normalization for the model training
    parser.add_argument('--use_full_pi_reg', action='store_true', default=False)  # whether to use full pi regularization, if true, will use the full pi regularization
    parser.add_argument('--use_weighted_qloss', action='store_true', default=False)  # whether to use weighted q loss for the actor, if true, will use the weighted q loss
    parser.add_argument('--cost_dir', type=str, default=None)  # path to cost directory, if None will use default data/costs_{K}
    parser.add_argument('--use_mmd_enhanced_qloss', action='store_true', default=False)  # whether to use mmd enhanced q loss for the actor, if true, will use the mmd enhanced q loss
    parser.add_argument('--cql_weight', type=float, default=0.1)  # whether to train with cql loss
    parser.add_argument('--cql_temp', type=float, default=1.0)  # temperature for cql loss
    parser.add_argument('--use_cql_loss', action='store_true', default=False)  # whether to use cql loss for the q function training, if true, will use the cql loss
    parser.add_argument('--loss_weighting_type', type=str, default='original')  # type of loss weighting, options are 'none', 'mmd', 'ot', 'original'
    parser.add_argument('--relabeling_type', type=str, default='command')  # type of relabeling, options are 'rtg', 'adv', 'value', 'command'
    parser.add_argument('--remove_critic_loss', action='store_true', default=False)  # whether to remove the critic loss, if true, will remove the critic loss
    parser.add_argument('--ablation', action='store_true', default=False)  # whether to run ablation study, if true, will run ablation study
    parser.add_argument('--ot_compute', type=str, default='sars')  # choose the way to compute ot costs, options are 'sars' and 'sas'
    args = parser.parse_args()

    experiment(args.exp_name, variant=vars(args))
