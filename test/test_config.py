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
    DataFilteringTrajectoryBuffer,
    load_config_for_env,
)
from torch.utils.tensorboard import SummaryWriter
def save_checkpoint(state,name):
  filename =name
  torch.save(state, filename)
from run_experiments import ExperimentRunner


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
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', f'{tar_env_name}.yaml'), 'r') as f:
        tar_env_config = yaml.safe_load(f).get('tar_env_config', {})

    # TODO: refactor to include in config file
    dversion = 2
    scale = 1000.
    env_targets = tar_env_config.get('env_targets', [5000, 4000, 2500]) # arbitrary default targets
    
    # NOTE: create source and target environments, using D4RL as source environments and OTDF datasets as target environments
    src_gym_name = f'{src_env_name}-{src_datatype}-v{dversion}'
    tar_gym_name = f'{tar_env_name}-{tar_datatype}-v{dversion}'
    if 'reacher' in tar_gym_name:
        tar_gym_name = 'Reacher2d-v4'
    src_env = gym.make(src_gym_name)

    if tar_gym_name == src_gym_name:
        tar_env = src_env
    else:
        # if 'gravity' in tar_datatype or 'morph' in tar_datatype:
        #     term = tar_datatype.split('_')[0]
        #     tar_env_config_path = f'config/{tar_env_name}_{term}.yaml'
        # elif 'kinematic' in tar_datatype:
        #     tar_env_config_path = f'config/{tar_env_name}.yaml'
        # else:
        #     raise ValueError(f"Unknown target datatype: {tar_datatype}")
        # with open(tar_env_config_path, 'r') as f:
        #     tar_config = yaml.safe_load(f)
        tar_config = load_config_for_env(src_gym_name, tar_gym_name)
        print("Loaded target environment config: ", tar_config)
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
    parser.add_argument('--use_mmd_enhanced_qloss', action='store_true', default=False)  # whether to use mmd enhanced q loss for the actor, if true, will use the mmd enhanced q loss
    parser.add_argument('--cql_weight', type=float, default=0.1)  # whether to train with cql loss
    parser.add_argument('--cql_temp', type=float, default=1.0)  # temperature for cql loss
    parser.add_argument('--use_cql_loss', action='store_true', default=False)  # whether to use cql loss for the q function training, if true, will use the cql loss
    parser.add_argument('--variation', help='Environment variation (required for single mode)')
    parser.add_argument('--srctype', default='medium', help='Source dataset type')
    parser.add_argument('--tartype', default='medium', help='Target dataset type')
    parser.add_argument('--short', action='store_true', help='Use short mode (reduced iterations)')
    parser.add_argument('--override', action='append', default=[], 
                       help='Override config parameters (format: section.key=value, e.g., experiment.learning_rate=1e-4)')
    parser.add_argument('--output-dir', default=None, help='Output directory for results')
    parser.add_argument('--num-workers', type=int, default=1, 
                       help='Number of parallel workers (for core12 and full108 modes)')
    args = parser.parse_args()

    experiment(args.exp_name, variant=vars(args))
    
    # Create runner
    runner = ExperimentRunner(args.output_dir)
    
    runner.load_config_for_env(args.env, args.variation, args.srctype, args.tartype)