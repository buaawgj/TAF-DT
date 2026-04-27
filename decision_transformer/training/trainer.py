
import time
import copy
import os
from typing import Union, List, Callable, Dict, Any, Optional, Tuple
from operator import itemgetter


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm, trange

from decision_transformer.misc.utils import (
    convert_to_tensor, expectile_regression, make_target, 
    compute_action_weight, concat_tensor_dict, compute_gae,
    _broadcast_bounds,
)


class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer:

    def __init__(self, 
                model,
                vae_policy,
                task, 
                critic,
                critic_q,
                critic_v,
                # NOTE: removed command fields, add it back later
                command,
                command_type,
                batch_size, 
                pretrain_batch_size,
                tau,
                discount,
                src_buffer,
                tar_buffer,
                loss_fn, 
                lambda_,
                adv_scale,
                eval_fns=None,
                max_q_backup=False,
                # NOTE: delayed_reward and use_mean_reduce set to False by default
                delayed_reward=False,
                use_mean_reduce=False,
                eta=1.0,
                eta2=1.0,
                ema_decay=0.995,
                step_start_ema=1000,
                update_ema_every=5,
                lr=3e-4,
                weight_decay=1e-4,
                lr_decay=False,
                lr_maxt=100000,
                lr_min=0.,
                grad_norm=1.0,
                scale=1.0,
                k_rewards=True,
                use_discount=True,
                v_target=False,
                # path for loading the saved critic q and v models and rtg network
                load_path=None,
                # path for loading the saved command network
                load_command_path=None,
                load_vae_path=None,  # path for loading the saved vae policy
                # path for saving the trained q, v and rtg models 
                save_path=None,
                # NOTE: change proportion default to 1.0, added device parameter
                proportion=1.0,
                ot_proportion=1.0,
                iql_tau=0.5,
                relabel_adv=True,
                fixed_timestep=False,
                rescale_reward=False,
                ot_filter=False,
                use_mmd_weights=False,
                pi_reg=False,  # whether to add policy regularization
                pi_reg_weight=0.01,  # the weight for the policy regularization
                vae_beta=0.4,  # beta for vae policy dataset probability calculation
                test_mode=False,  # whether to run in test mode, if true, will choose the greedy action
                training_normalization=False,  # whether to use the training normalization for the transformer network
                use_full_pi_reg=False,  # whether to use the full policy regularization
                use_weighted_qloss=False,  # whether to use the weighted q loss for the actor
                use_mmd_enhanced_qloss=False,  # whether to use the mmd weights to enhance the q loss
                cql_weight=0.1, # weight for the cql loss
                cql_temp=1.0, # temperature for the cql loss
                use_cql_loss=False, # whether to use the cql loss
                min_action=None, # min action value, used for the cql loss
                max_action=None, # max action value, used for the cql loss
                # ablation arguments
                loss_weighting_type='original', # type of loss weighting, options are 'none', 'mmd', 'ot', 'original'
                relabeling_type='command', # type of relabeling, options are 'rtg', 'adv', 'value', 'command'
                remove_critic_loss=False, # whether to remove the critic loss during training
                ablation=False,  # whether to run ablation study
                device=None,
            ):
        
        self.task = task
        
        self.actor = model
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)
        self.vae_policy = vae_policy
        #! The learning rate for the VAE policy is different from the OTDF.
        self.vae_policy_optimizer = torch.optim.Adam(self.vae_policy.parameters(), lr=lr, weight_decay=weight_decay)

        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
     
        self.src_buffer = src_buffer
        self.tar_buffer = tar_buffer 
        
        self.v_target = v_target
        # NOTE: removed command and command_type fields, please add it back later
        self.command = command
        self.command_type = command_type
        if self.v_target:
            self.critic_q = critic_q
            self.critic_v = critic_v
            self.critic_v_target = make_target(critic_v)
        else: 
            self.critic_q = critic_q
            self.critic_v = critic_v
            self.critic_q_target = make_target(critic_q)
            
        self.critic_q_optim = torch.optim.Adam(self.critic_q.parameters(), lr=lr)
        self.critic_v_optim = torch.optim.Adam(self.critic_v.parameters(), lr=lr)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=lr_min)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=lr_min)

        self.batch_size = batch_size
        self.pretrain_batch_size = pretrain_batch_size
        self.loss_fn = loss_fn
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.tau = tau
        self.max_q_backup = max_q_backup
        self.discount = discount
        self.grad_norm = grad_norm
        self.eta = eta
        self.eta2 = eta2
        self.lr_decay = lr_decay
        self.scale = scale
        self.k_rewards = k_rewards
        self.use_discount = use_discount
        self.fixed_timestep = fixed_timestep

        # parameters for the relabeling of the buffers
        self.lambda_ = lambda_
        self.adv_scale = adv_scale
        self.delayed_reward = delayed_reward
        self.use_mean_reduce = use_mean_reduce
        self.proportion = proportion
        self.ot_proportion = ot_proportion
        self.load_path = load_path
        self.save_path = save_path
        self._iql_tau = iql_tau  # the parameter of training the v network
        self.load_command_path = load_command_path  # for the test purpose, delete it after testing
        self.load_vae_path = load_vae_path  # path to load the vae policy, if None, will not load
        self.relabel_adv = relabel_adv # possible conflict with line above?
        self.rescale_reward = rescale_reward # whether to rescale the reward by the ot weight
        self.ot_filter = ot_filter # whether to filter the transitions based on the ot costs
        self.use_mmd_weights = use_mmd_weights # whether to filter the transitions based on the mmd costs only
        self.pi_reg = pi_reg  # whether to add policy regularization
        self.pi_reg_weight = pi_reg_weight  # the weight for the policy regularization
        self.vae_beta = vae_beta  # beta for vae policy dataset probability calculation
        self.test_mode = test_mode  # whether to run in test mode, if true, will choose the greedy action
        self.training_normalization = training_normalization  # whether to use the training normalization for the transformer network
        self.use_full_pi_reg = use_full_pi_reg  # whether to use the full policy regularization
        self.use_weighted_qloss = use_weighted_qloss  # whether to use the weighted q loss for the actor
        self.use_mmd_enhanced_qloss = use_mmd_enhanced_qloss  # whether to use the mmd weights to enhance the q loss
        self.cql_weight = cql_weight  # weight for the cql loss
        self.cql_temp = cql_temp  # temperature for the cql loss
        self.use_cql_loss = use_cql_loss  # whether to use the cql loss
        # ablation arguments
        self.loss_weighting_type = loss_weighting_type # type of loss weighting, options are 'none', 'mmd', 'ot', 'original'
        self.relabeling_type = relabeling_type # type of relabeling, options are 'rtg', 'adv', 'value', 'command'
        self.remove_critic_loss = remove_critic_loss # whether to remove the critic loss during training
        self.ablation = ablation # whether to run ablation study

        self.min_action = min_action # min action value, used for the cql loss
        self.max_action = max_action # max action value, used for the cql loss

        self.start_time = time.time()
        self.step = 0

        # NOTE: added device parameters
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
    
    def init_step(self):
        self.step = 0        
    
    def step_ema(self):
        if self.step > self.step_start_ema and self.step % self.update_ema_every == 0:
            self.ema.update_model_average(self.ema_model, self.actor)
     
    @torch.no_grad()
    def select_command(self, states):
        states = torch.from_numpy(states).float().to(self.device)
        return self.command.select_command(states).detach().cpu().numpy()
    
    def update_command(self, batch: Dict):
        loss_metrics = self.command.update(batch)
        return loss_metrics

    def concat_src_tar_batch(
        self, src_batch: Dict[str, Any], tar_batch: Dict[str, Any], compute_weight: bool=True, compute_adv: bool=False
        ) -> Dict[str, Any]:
        """
        Concatenate source and target batches into a single batch.
        """
        for _key, _value in src_batch.items():
            src_batch[_key] = convert_to_tensor(_value, self.device)
        for _key, _value in tar_batch.items():
            tar_batch[_key] = convert_to_tensor(_value, self.device)
        # NOTE: added timesteps to load from source and target trajectories
        #!!! Add mmd_costs and ot_costs to src_batch
        src_s, src_a, src_r, src_next_s, src_terminals, src_masks, src_timesteps, src_mmd_costs, src_ot_costs = itemgetter(
            "observations", "actions", "rewards", "next_observations", "terminals", "masks", "timesteps", "mmd_costs", "ot_costs",
        )(src_batch)
        tar_s, tar_a, tar_r, tar_next_s, tar_terminals, tar_masks, tar_timesteps = itemgetter(
            "observations", "actions", "rewards", "next_observations", "terminals", "masks", "timesteps",
        )(tar_batch)
        
        if compute_adv:
            if self.relabeling_type == 'value':
                src_advs = src_batch.get("values", None)
                tar_advs = tar_batch.get("values", None)
            elif self.relabeling_type == 'rtg':
                src_advs = src_batch.get("returns", None)
                tar_advs = tar_batch.get("returns", None)
            else:
                src_advs = src_batch.get("agent_advs", None)
                tar_advs = tar_batch.get("agent_advs", None)
        # else:
        #     src_rtgs = src_batch.get("returns", None)
        #     tar_rtgs = tar_batch.get("returns", None)
            

        # NOTE: mmd_costs and ot_costs normalization removed since it should be normalized in the buffer rather than computed here, but the normalization code is not written in the buffer yet

        # normalization
        # mmd_costs = (mmd_costs - torch.max(mmd_costs))/(torch.max(mmd_costs) - torch.min(mmd_costs))
        # ot_costs = (ot_costs - torch.max(ot_costs))/(torch.max(ot_costs) - torch.min(ot_costs))
        
        # filter out transitions
        batch_size = src_s.shape[0]
        src_filter_num = int(batch_size * self.ot_proportion)
        
        # NOTE: filtering moved to experiment.py, keeping this code commented for later tests
        
        if self.ot_filter and (self.loss_weighting_type == 'ot' or self.loss_weighting_type == 'original'):
            src_ot_costs = src_ot_costs * src_masks.unsqueeze(-1)
            src_ot_cost_sum = src_ot_costs.sum(dim=[-1, -2])
            src_mask_sum = src_masks.sum(dim=-1)
            src_ot_cost_mean = src_ot_cost_sum / (src_mask_sum + 1e-12)
            filter_cost, indices = torch.topk(src_ot_cost_mean, src_filter_num)
        # elif self.loss_weighting_type == 'mmd':
        #     src_mmd_costs = src_mmd_costs * src_masks.unsqueeze(-1)
        #     src_mmd_cost_sum = src_mmd_costs.sum(dim=[-1, -2])
        #     src_mask_sum = src_masks.sum(dim=-1)
        #     src_mmd_cost_mean = src_mmd_cost_sum / (src_mask_sum + 1e-12)
        #     filter_cost, indices = torch.topk(src_mmd_cost_mean, src_filter_num)
        else:
            indices = torch.arange(batch_size)
            
        src_s = src_s[indices]
        src_a = src_a[indices]
        src_next_s = src_next_s[indices]
        src_r = src_r[indices]
        src_terminals = src_terminals[indices]
        src_mmd_costs = src_mmd_costs[indices]
        src_ot_costs = src_ot_costs[indices]
        src_masks = src_masks[indices]
        src_timesteps = src_timesteps[indices]
        if compute_adv:
            src_advs = src_advs[indices]
        src_mask_sum = src_masks.sum(dim=-1)

        
        state = torch.cat([src_s, tar_s], 0)
        action = torch.cat([src_a, tar_a], 0)
        next_state = torch.cat([src_next_s, tar_next_s], 0)
        reward = torch.cat([src_r, tar_r], 0)
        terminal = torch.cat([src_terminals, tar_terminals], 0)
        mask = torch.cat([src_masks, tar_masks], 0)
        timesteps = torch.cat([src_timesteps, tar_timesteps], 0)
        # NOTE: not always load advs, since we do not need this term when training q, v
        if compute_adv:
            adv = torch.cat([src_advs, tar_advs], 0)
        
        length = src_s.shape[0]
        
        # concat the mmd costs and ot costs
        mmd_costs = torch.zeros_like(reward).to(self.device)
        ot_costs = torch.zeros_like(reward).to(self.device)
        
        mmd_costs[:length] = src_mmd_costs
        ot_costs[:length] = src_ot_costs
        
        # NOTE: changed weight initialization to match final batch dimensions
        ot_weight = torch.ones_like(reward).to(self.device)
        mmd_weight = torch.ones_like(reward).to(self.device)

        if compute_weight:
            # calculate cost weight
            src_ot_weight = torch.exp(src_ot_costs)
            ot_weight[:length] = src_ot_weight
            
            src_mmd_weight = torch.exp(src_mmd_costs)
            mmd_weight[:length] = src_mmd_weight
            
            #! rescale the reward
            if self.rescale_reward and not self.ablation:
                # NOTE: need to change loss weighting?
                if self.loss_weighting_type == 'mmd':
                    reward = reward * mmd_weight
                if self.loss_weighting_type == 'ot':
                    reward = reward * ot_weight
                if self.loss_weighting_type == 'original':
                    reward = reward * ot_weight * mmd_weight
        
        # NOTE: changed the result to a dictionary
        if compute_adv:
            result = {
                "observations": state,
                "actions": action,
                "next_observations": next_state,
                "rewards": reward,
                "terminals": terminal,
                "masks": mask,
                "timesteps": timesteps,
                "ot_weights": ot_weight,
                # add a weight computed from mmd costs
                "mmd_weights": mmd_weight,
                "mmd_costs": mmd_costs,
                "ot_costs": ot_costs,
                # new rtg
                "agent_advs": adv,
            }
        else:
            result = {
                "observations": state,
                "actions": action,
                "next_observations": next_state,
                "rewards": reward,
                # "agent_advs": rtgs,  # renamed from returns to agent_advs
                "terminals": terminal,
                "masks": mask,
                "timesteps": timesteps,
                "ot_weights": ot_weight,
                # add a weight computed from mmd costs
                "mmd_weights": mmd_weight,
                "mmd_costs": mmd_costs,
                "ot_costs": ot_costs,
            }
        
        return result
        
    def _sync_target(self):
        """synchronize the parameter of the target network"""
        if self.v_target:
            for o, n in zip(self.critic_v_target.parameters(), self.critic_v.parameters()):
                o.data.copy_(o.data * (1.0 - self.tau) + n.data * self.tau)
        else:
            for o, n in zip(self.critic_q_target.parameters(), self.critic_q.parameters()):
                o.data.copy_(o.data * (1.0 - self.tau) + n.data * self.tau)
            
    def update_critic(self, states, actions, rewards, next_states, terminals, masks, weights):   
        #! The weights shoule be computed from the ot and mmd costs, not only the ot costs.
        if self.v_target:
            # Use the batch to compute the q loss
            with torch.no_grad():
                target = self.critic_v_target.v_min(next_states)
                target = rewards + self.discount * (1-terminals) * target
            q_pred1, q_pred2 = self.critic_q(states, actions)
            q_loss = weights * ((target - q_pred1)**2 + (target - q_pred2)**2)
            q_loss = (q_loss * masks.unsqueeze(-1)).mean()
            
            self.critic_q_optim.zero_grad()
            q_loss.backward()
            self.critic_q_optim.step()
            
            # Use the batch to compute the value loss
            v_pred1, v_pred2 = self.critic_v(states)
            v_loss1 = expectile_regression(v_pred1, target, expectile=self._iql_tau)
            v_loss2 = expectile_regression(v_pred2, target, expectile=self._iql_tau)
            v_loss = ((v_loss1 + v_loss2) * masks.unsqueeze(-1)).mean()
            
            self.critic_v_optim.zero_grad()
            v_loss.backward()
            self.critic_v_optim.step()
            
        else:
            # Use the batch to compute the q loss
            with torch.no_grad():
                q_target = self.critic_v.v_min(next_states)
                q_target = rewards + self.discount * (1-terminals) * q_target
                q_old = self.critic_q_target.q_min(states, actions)
            q_pred1, q_pred2 = self.critic_q(states, actions)
            q_loss = weights * ((q_target - q_pred1)**2 + (q_target - q_pred2)**2)
            q_loss = (q_loss * masks.unsqueeze(-1)).mean()

            self.critic_q_optim.zero_grad()
            q_loss.backward()
            self.critic_q_optim.step()
            
            # Update the target batch to compute the value loss
            v_pred1, v_pred2 = self.critic_v(states, reduce=False)
            v_loss1 = expectile_regression(v_pred1, q_old, expectile=self._iql_tau)
            v_loss2 = expectile_regression(v_pred2, q_old, expectile=self._iql_tau)
            v_loss = ((v_loss1 + v_loss2) * masks.unsqueeze(-1)).mean()
            
            self.critic_v_optim.zero_grad()
            v_loss.backward()
            self.critic_v_optim.step()
            
        self._sync_target()
        return {
            "q_loss": q_loss.item(), 
            "v_loss": v_loss.item(), 
            "q_pred": q_pred1.mean().item(), 
            "v_pred": v_pred1.mean().item(),
        }
        
    def pretrain_q_and_v(self, max_iters, num_steps_per_iter, writer=None):
        """ Pretrain the critic q and v networks for computing the RTG of states in the buffers."""
        # NOTE: removed DataLoader iterators
        if self.load_path is None:
            for i_epoch in trange(1, int(max_iters+1), desc="pretrain critic", mininterval=5.0):
                for i_step in range(int(num_steps_per_iter)):
                    # NOTE: replaced DataLoader with buffer.get_batch code, but yet to test on real data
                    src_batch = self.src_buffer.get_batch(self.pretrain_batch_size)
                    tar_batch = self.tar_buffer.get_batch(self.pretrain_batch_size)
                    
                    # # NOTE: load state mean and std from source buffer, may move this step to experiment.py later
                    # if self.training_normalization:
                    #     src_state_mean = self.src_buffer.obs_mean
                    #     src_state_std = self.src_buffer.obs_std
                    #     src_states = src_batch['observations']
                    #     src_states = (src_states - src_state_mean) / (src_state_std + 1e-8)
                    #     src_batch['observations'] = src_states
                        
                    #     tar_state_mean = self.tar_buffer.obs_mean
                    #     tar_state_std = self.tar_buffer.obs_std
                    #     tar_states = tar_batch['observations']
                    #     tar_states = (tar_states - src_state_mean) / (src_state_std + 1e-8)
                    #     tar_batch['observations'] = tar_states
                    
                    batch = self.concat_src_tar_batch(src_batch, tar_batch, compute_weight=True)
                    
                    state, action, next_state, reward, terminal, timestep, mask, ot_weight, mmd_weight, mmd_cost = (
                        batch['observations'],
                        batch['actions'],
                        batch['next_observations'],
                        batch['rewards'],
                        batch['terminals'],
                        batch['timesteps'],
                        batch['masks'],
                        batch['ot_weights'],
                        batch['mmd_weights'],
                        batch['mmd_costs'],
                    )
                    
                    if self.loss_weighting_type == 'none' or self.loss_weighting_type == 'mmd':
                        weight = torch.ones_like(ot_weight).to(self.device)
                    elif self.loss_weighting_type == 'ot' or self.loss_weighting_type == 'original':
                        weight = ot_weight

                    train_metrics = self.update_critic(state, action, reward, next_state, terminal, mask, weight)
                    
                    q_loss = train_metrics['q_loss']
                    v_loss = train_metrics['v_loss']
                    q_pred = train_metrics['q_pred']
                    v_pred = train_metrics['v_pred']
                    
                    if writer is not None:
                        writer.add_scalar('step/rtg_v_loss', v_loss, self.step)
                        writer.add_scalar('step/rtg_q_loss', q_loss, self.step)
                        writer.add_scalar('step/rtg_v_pred', v_pred, self.step)
                        writer.add_scalar('step/rtg_q_pred', q_pred, self.step)   
                    
                    self.step += 1
                    
                print("Pretrain critic | q loss : {:.3f} --|-- q pred : {:.3f} |".format(q_loss, q_pred))     
            
            if self.save_path is not None:
                save_path = os.path.join(self.save_path, self.task)
                if not os.path.exists(save_path): 
                    os.makedirs(save_path)
                torch.save(self.critic_q.state_dict(), os.path.join(save_path, "critic_q.pt"))
                torch.save(self.critic_v.state_dict(), os.path.join(save_path, "critic_v.pt"))
                
        elif self.load_path is not None:
            load_path = os.path.join(self.load_path, self.task)
            self.critic_q.load_state_dict(torch.load(os.path.join(load_path, "critic_q.pt"), map_location="cpu"))
            self.critic_v.load_state_dict(torch.load(os.path.join(load_path, "critic_v.pt"), map_location="cpu"))
            self.critic_q.to(self.device)
            self.critic_v.to(self.device)
    
    @torch.no_grad()        
    def relabel_buffer(self, buffer):
        """ Relabel the RTG of the states in the buffer using the critic q and v networks."""
        values = np.zeros_like(buffer.rewards)
        agent_advs = np.zeros_like(buffer.rewards)
        traj_start = 0
        traj_num = len(buffer.traj_lens)
        with torch.no_grad():
            for i in range(traj_num):
                if self.lambda_ is None:
                    traj_len = buffer.traj_lens[i]
                    obss = torch.from_numpy(buffer.observations[i, -traj_len:]).to(self.device)
                    actions = torch.from_numpy(buffer.actions[i, -traj_len:]).to(self.device)
                    
                    # # NOTE: load state mean and std from source buffer, may move this step to experiment.py later
                    # if self.training_normalization:
                    #     src_state_mean = self.src_buffer.obs_mean
                    #     src_state_std = self.src_buffer.obs_std
                    #     obss = (obss - src_state_mean) / (src_state_std + 1e-8)
                    
                    if self.use_mean_reduce:
                        vs = self.critic_v.v_mean(obss).unsqueeze(-1).detach().cpu().numpy()
                        qs = self.critic_q.q_mean(obss, actions).detach().cpu().numpy()
                    else:
                        vs = self.critic_v.v_min(obss).unsqueeze(-1).detach().cpu().numpy()
                        qs = self.critic_q.q_min(obss, actions).detach().cpu().numpy()
                    values[i, -traj_len:] = vs
                    agent_advs[i, -traj_len:] = qs - vs
                else:
                    traj_len = buffer.traj_lens[i]
                    rewards = buffer.rewards[i, -traj_len:]
                    mask = None
                    if hasattr(buffer, "masks"):
                        mask = buffer.masks[i, -traj_len:]
                    obss = torch.from_numpy(buffer.observations[i, -traj_len:]).to(self.device)
                    last_obs = torch.from_numpy(buffer.next_observations[i, -1]).to(self.device)
                    
                    # # NOTE: load state mean and std from source buffer, may move this step to experiment.py later
                    # if self.training_normalization:
                    #     src_state_mean = self.src_buffer.obs_mean
                    #     src_state_std = self.src_buffer.obs_std
                    #     obss = (obss - src_state_mean) / (src_state_std + 1e-8)
                    #     last_obs = (last_obs - src_state_mean) / (src_state_std + 1e-8)
                    
                    if self.use_mean_reduce:
                        # NOTE choose this way to compute the adv
                        vs = self.critic_v.v_mean(obss).unsqueeze(-1).detach().cpu().numpy()
                        last_v = self.critic_v.v_mean(last_obs).reshape(-1,1).detach().cpu().numpy()
                        # print("vs: ", vs.shape)
                        # print("last v: ", last_v.shape)
                    else:
                        vs = self.critic_v.v_min(obss).unsqueeze(-1).detach().cpu().numpy()
                        last_v = self.critic_v.v_min(last_obs).reshape(-1,1).detach().cpu().numpy()
                    # print("rewards: ", rewards)
                    # print("vs: ", vs.shape)
                    # print("last_v: ", last_v.shape)
                    # print("use_mean_reduce: ", self.use_mean_reduce)
                    gae, _ = compute_gae(rewards, vs, last_v, gamma=self.discount, lam=self.lambda_, dim=0, mask=mask, K=20)
                    values[i, -traj_len:] = vs
                    agent_advs[i, -traj_len:] = gae
                    # print("gae: ", gae)
                    
        return values, agent_advs
                    
        # if self.adv_scale is not None:
        #     if self.delayed_reward:
        #         max_adv = agent_advs.max()
        #         agent_advs = agent_advs / max_adv * self.adv_scale
        #     else:
        #         advs_ = agent_advs[agent_advs!=0]
        #         std_ = advs_.std()
        #         agent_advs = (agent_advs / std_) * self.adv_scale
        
        # # add new members into the buffer
        # buffer.agent_advs = agent_advs
        # buffer.values = values
        
    def normalize_adv(self, src_advs, tar_advs, adv_mean_reduce=False):
        """ 
        Normalize the advantages of the source and target buffers.
        -- adv_mean_reduce: if True, normalize the advantages by subtracting the mean and dividing by the standard deviation.
        -- if False, normalize the advantages by dividing by the standard deviation.
        """
        advs = np.concatenate([src_advs, tar_advs], axis=None)
        if self.adv_scale is not None:
            if self.delayed_reward:
                max_adv = advs.max()
                nor_src_advs = src_advs / max_adv * self.adv_scale
                nor_tar_advs = tar_advs / max_adv * self.adv_scale
            else:
                advs_ = advs[advs!=0]
                std_ = advs_.std()
                mean_ = advs_.mean()
                if not adv_mean_reduce:
                    nor_src_advs = (src_advs / std_) * self.adv_scale
                    nor_tar_advs = (tar_advs / std_) * self.adv_scale
                elif adv_mean_reduce:
                    nor_src_advs = ((src_advs - mean_) / std_) * self.adv_scale
                    nor_tar_advs = ((tar_advs - mean_) / std_) * self.adv_scale
        
            return nor_src_advs, nor_tar_advs
        else:
            # if adv_scale is None, we do not normalize the advantages
            return src_advs, tar_advs
            
    def train_rtg_network(self, pretrain_command_epoch, step_per_epoch, writer=None, adv_mean_reduce=False, command_state_normalization=False):
        """ Train a network for predicting the reward to go (RTG) of states generated during training"""
        # Compute the RTG for the source and target buffers
        src_values, src_advs = self.relabel_buffer(self.src_buffer)
        tar_values, tar_advs = self.relabel_buffer(self.tar_buffer)
        
        nor_src_advs, nor_tar_advs = self.normalize_adv(src_advs, tar_advs, adv_mean_reduce=adv_mean_reduce)
        self.src_buffer.agent_advs = nor_src_advs
        self.tar_buffer.agent_advs = nor_tar_advs
        self.src_buffer.values = src_values
        self.tar_buffer.values = tar_values

        # NOTE: removed DataLoader iterators
        if self.load_command_path == None:
            # train the command network
            if self.command_type == "in_sample_max":
                for i_epoch in trange(1, pretrain_command_epoch+1, desc="pretrain_command", mininterval=5.0):
                    for i_step in range(step_per_epoch):
                        # NOTE: replaced DataLoader with buffer.get_batch code, but yet to test on real data
                        src_batch = self.src_buffer.get_batch(self.pretrain_batch_size)
                        tar_batch = self.tar_buffer.get_batch(self.pretrain_batch_size)
                        
                        # NOTE: load state mean and std from source buffer, may move this step to experiment.py later
                        if command_state_normalization:
                            src_state_mean = self.src_buffer.obs_mean
                            src_state_std = self.src_buffer.obs_std
                            src_states = src_batch['observations']
                            src_states = (src_states - src_state_mean) / (src_state_std + 1e-8)
                            src_batch['observations'] = src_states
                            
                            tar_state_mean = self.tar_buffer.obs_mean
                            tar_state_std = self.tar_buffer.obs_std
                            tar_states = tar_batch['observations']
                            tar_states = (tar_states - tar_state_mean) / (tar_state_std + 1e-8)
                            tar_batch['observations'] = tar_states
                        
                        for _key, _value in src_batch.items():
                            src_batch[_key] = convert_to_tensor(_value, self.device)
                        for _key, _value in tar_batch.items():
                            tar_batch[_key] = convert_to_tensor(_value, self.device)

                        batch = concat_tensor_dict([tar_batch, src_batch])
                        
                        # # # NOTE: normalize the states in the batch 
                        # if command_state_normalization:
                        #     states = batch['observations']
                        #     states = (states - state_mean) / (state_std + 1e-8)
                        #     batch['observations'] = states
                    
                        train_metrics = self.update_command(batch)

                    print("Gap between the true and predicted commands: ", train_metrics[f"{self.command.id}_ISM_gap"])
                    print("Loss of the command network: ", train_metrics[f"{self.command.id}_ISM_loss"])
                    
                    if writer is not None:
                        for key in train_metrics.keys():
                            writer.add_scalar('step/'+key, train_metrics[key], self.step)
                            
                    self.step += 1

                if self.save_path is not None:
                    save_path = os.path.join(self.save_path, self.task)
                    if not os.path.exists(save_path): 
                        os.makedirs(save_path)
                    torch.save(self.command.state_dict(), os.path.join(save_path, "command.pt"))
                    
            elif self.command_type == "constant":
                # For constant command, we only the mean of the positive agent advantages of the target buffer.
                self.command.set_value(self.tar_buffer.agent_advs[self.tar_buffer.agent_advs > 0.01].mean())
                print("Final command value: ", self.command.constant.item())
            # NOTE: removed iterator deletion code, as we are not using DataLoader iterators anymore
            
        elif self.load_command_path is not None:
            load_command_path = os.path.join(self.load_command_path, self.task)
            self.command.load_state_dict(torch.load(os.path.join(load_command_path, "command.pt"), map_location="cpu"))
            self.command.to(self.device)
            
    def train_vae(self, pretrain_vae_epoch, step_per_epoch, writer=None):
        # NOTE: load state mean and std from target buffer, may move this step to experiment.py later
        state_mean = convert_to_tensor(self.tar_buffer.obs_mean, self.device)
        state_std = convert_to_tensor(self.tar_buffer.obs_std, self.device)
        
        # NOTE: removed DataLoader iterators
        if self.load_vae_path == None:
            for i_epoch in trange(1, pretrain_vae_epoch+1, desc="train VAE", mininterval=5.0):
                for i_step in range(step_per_epoch):
                    # tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)
                    tar_batch = self.tar_buffer.get_batch(self.pretrain_batch_size)
                    for _key, _value in tar_batch.items():
                        tar_batch[_key] = convert_to_tensor(_value, self.device)
                    tar_s, tar_a, tar_r, tar_next_s, tar_terminals, tar_masks, tar_timesteps = itemgetter(
                        "observations", "actions", "rewards", "next_observations", "terminals", "masks", "timesteps",
                    )(tar_batch)
                    
                    tar_s = (tar_s - state_mean) / (state_std + 1e-8)

                    # Variational Auto-Encoder Training
                    recon, mean, std    = self.vae_policy(tar_s, tar_a)
                    recon_loss          = (recon - tar_a)**2
                    recon_loss          = (recon_loss * tar_masks.unsqueeze(-1)).mean()
                    KL_loss             = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2))
                    KL_loss             = (KL_loss * tar_masks.unsqueeze(-1)).mean()
                    vae_loss            = recon_loss + 0.5 * KL_loss

                    self.vae_policy_optimizer.zero_grad()
                    vae_loss.backward()
                    self.vae_policy_optimizer.step()
                    
                if writer is not None:
                    writer.add_scalar('step/vae_loss', vae_loss.item(), self.step)
                    print("Pretrain  |  VAE loss: ", vae_loss.item())
                
                self.step += 1
            
            if self.save_path is not None:
                save_path = os.path.join(self.save_path, self.task)
                if not os.path.exists(save_path): 
                    os.makedirs(save_path)
                torch.save(self.vae_policy.state_dict(), os.path.join(save_path, "vae_policy.pt"))
                
        elif self.load_vae_path is not None:
            load_vae_path = os.path.join(self.load_vae_path, self.task)
            self.vae_policy.load_state_dict(torch.load(os.path.join(load_vae_path, "vae_policy.pt"), map_location="cpu"))
            self.vae_policy.to(self.device)
                  
    def compute_cql_penalty(
        self,
        current_q1: torch.Tensor,    # [B, T]
        current_q2: torch.Tensor,    # [B, T]
        states: torch.Tensor,           # [B, T, state_dim]
        actions: torch.Tensor,          # [B, T, action_dim]  (dataset actions)
        action_preds: torch.Tensor,     # [B, T, action_dim]  (policy predicted actions, aligned with states)
        attention_mask: torch.Tensor,   # [B, T] or [B, T, 1]
        ot_weights: Optional[torch.Tensor] = None,      # [B, T] or [B, T, 1]
        mmd_weights: Optional[torch.Tensor] = None,     # [B, T] or [B, T, 1]
        num_random: int = 10,
        temp: float = 1.0,
        use_mmd_enhanced_weights: bool = False,
        action_low: Union[float, torch.Tensor] = -1.0,
        action_high: Union[float, torch.Tensor] = 1.0,
        chunk_size: Optional[int] = None,               # If N*K is large, random action Q can be calculated in blocks.
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Return:
        - cql_penalty: Scalar tensor (unmultiplied alpha)
        - info: Convenient statistics for recording (such as the mean of q data/pi/rand, etc.)

        Calculation formula (trajectory-wise, by significant bits) :
        penalty = E_{s~D} [ logsumexp( {Q(s, a_pi)} U {Q(s, a_rand)_K} ) - Q(s, a_data) ]
        Among them, logsumexp has the temperature temp.
        """
        device = states.device
        B, T, A = actions.shape

        # unify the mask/weight shape
        if attention_mask.dim() == 3 and attention_mask.size(-1) == 1:
            attn = attention_mask[..., 0]  # [B, T]
        else:
            attn = attention_mask          # [B, T]

        # Align with your current goals: Only at t=0,... T-2 compute the value (to avoid target misalignment of the last bit)
        valid_mask_bt = attn[:, :-1] > 0   # [B, T-1]
        N = valid_mask_bt.sum().item()
        if N == 0:
            # No significant bit
            return torch.zeros((), device=device), {
                "q_data_mean": 0.0, "q_pi_mean": 0.0, "q_rand_mean": 0.0, "n_valid": 0
            }

        # select the valid positions (s, a_data, a_pi)
        s_valid  = states[:, :-1, :][valid_mask_bt]          # [N, state_dim]
        a_pi     = action_preds[:, :-1, :][valid_mask_bt]    # [N, action_dim]

        # Q(s, a_data) and Q(s, a_pi) (participate in the gradient! Do not no_grad)
        # q1_d, q2_d = self.critic(s_valid, a_data)                 # [N,1] , [N,1]
        q1_d   = current_q1[:, :-1][valid_mask_bt]
        q2_d   = current_q2[:, :-1][valid_mask_bt]
        q_d    = torch.min(q1_d, q2_d).squeeze(-1)              # [N]

        q1_pi, q2_pi = self.critic(s_valid, a_pi)                 # [N,1] , [N,1]
        q_pi = torch.min(q1_pi, q2_pi).squeeze(-1)           # [N]

        # random action Q(s, a_rand), supports block computing to save video memory
        if num_random > 0:
            # sample from the actual range of motion
            low_t, high_t = _broadcast_bounds(action_low, action_high, (N, A), device)
            # generate random actions of [N, K, A]
            # NOTE: grad is not required here
            with torch.no_grad():
                a_rand = torch.rand(N, num_random, A, device=device) * (high_t - low_t) + low_t  # [N,K,A]
            # run critic in blocks
            q_r_list = []
            if chunk_size is None:
                chunk_size = N * num_random  # One-time Run
            flat_s = s_valid.unsqueeze(1).expand(-1, num_random, -1).reshape(-1, s_valid.shape[-1])  # [N*K, S]
            flat_a = a_rand.reshape(-1, A)  # [N*K, A]
            for start in range(0, flat_s.size(0), chunk_size):
                end = min(start + chunk_size, flat_s.size(0))
                q1_r, q2_r = self.critic(flat_s[start:end], flat_a[start:end])   # [M,1]
                q_r_list.append(torch.min(q1_r, q2_r))                      # [M,1]
            q_r = torch.cat(q_r_list, dim=0).reshape(N, num_random)         # [N,K]
        else:
            q_r = torch.empty(N, 0, device=device)                          # [N,0]

        # logsumexp aggregation
        q_cat = torch.cat([q_pi.unsqueeze(1), q_r], dim=1) if q_r.numel() > 0 else q_pi.unsqueeze(1)  # [N, K+1]
        cql_term = (torch.logsumexp(q_cat / temp, dim=1) * temp) - q_d       # [N]

        #  weights (aligned with critic MSE)
        if ot_weights is None or self.loss_weighting_type == 'none':
            w_flat = None
        else:
            if self.loss_weighting_type == 'ot' or self.loss_weighting_type == 'original':
                w = ot_weights[:, :-1]
                if w.dim() == 3 and w.size(-1) == 1: w = w[..., 0]
            else:
                w = torch.ones_like(ot_weights[:, :-1])
            if use_mmd_enhanced_weights and (mmd_weights is not None):
                if self.loss_weighting_type == 'mmd' or self.loss_weighting_type == 'original':
                    m = mmd_weights[:, :-1]
                    if m.dim() == 3 and m.size(-1) == 1: m = m[..., 0]
                    w = torch.sqrt(torch.clamp(w, min=0)) * torch.sqrt(torch.clamp(m, min=0))
            w_flat = w[valid_mask_bt]  # [N]

        # remove unless it's a numerical value
        finite_mask = torch.isfinite(cql_term)
        cql_term = cql_term[finite_mask]
        if w_flat is not None:
            w_flat = w_flat[finite_mask]

        if cql_term.numel() == 0:
            penalty = torch.zeros((), device=device)
        else:
            penalty = (w_flat * cql_term).mean() if (w_flat is not None and w_flat.numel() == cql_term.numel()) else cql_term.mean()

        # logging info
        with torch.no_grad():
            info = {
                "q_data_mean": float(q_d.mean().item()),
                "q_pi_mean":   float(q_pi.mean().item()),
                "q_rand_mean": float(q_r.mean().item()) if q_r.numel() > 0 else 0.0,
                "n_valid":     int(N),
            }

        return penalty, info
                       
    # NOTE: removed pin_memory and num_workers parameters, as they are not used in the function
    # NOTE: removed logger parameter as its functionality is included in writer
    def train_iteration(self, num_steps, writer, iter_num=0):
        
        logs = dict()

        train_start = time.time()

        self.actor.train()
        self.critic.train()
        loss_metric = {
            'bc_loss': [],
            'ql_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'target_q_mean': [],
        }

        # Main training loop
        for _ in trange(num_steps, mininterval=5.0):
            src_batch = self.src_buffer.get_batch(self.batch_size)
            tar_batch = self.tar_buffer.get_batch(self.batch_size)
            
            # NOTE: load state mean and std from source buffer, may move this step to experiment.py later
            if self.training_normalization:
                src_state_mean = self.src_buffer.obs_mean
                src_state_std = self.src_buffer.obs_std
                src_states = src_batch['observations']
                src_states = (src_states - src_state_mean) / (src_state_std + 1e-8)
                src_batch['observations'] = src_states
                
                tar_state_mean = self.tar_buffer.obs_mean
                tar_state_std = self.tar_buffer.obs_std
                tar_states = tar_batch['observations']
                tar_states = (tar_states - tar_state_mean) / (tar_state_std + 1e-8)
                tar_batch['observations'] = tar_states
            
            batch = self.concat_src_tar_batch(
                src_batch, tar_batch, compute_weight=True, compute_adv=True
            )
            loss_metric = self.train_step(batch, writer, loss_metric)
        
        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        # NOTE: changed logging code
        logs['train/bc_loss'] = np.mean(loss_metric['bc_loss'])
        logs['train/ql_loss'] = np.mean(loss_metric['ql_loss'])
        logs['train/actor_loss'] = np.mean(loss_metric['actor_loss'])
        logs['train/critic_loss'] = np.mean(loss_metric['critic_loss'])
        logs['train/target_q_mean'] = np.mean(loss_metric['target_q_mean'])
        # end of logging change

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.actor.eval()
        self.critic.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.actor, self.critic_target, self.critic_q, self.critic_v, self.command, test_mode=self.test_mode)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        best_ret = -10000
        best_nor_ret = -10000

        for k, v in logs.items():
            if 'return_mean' in k:
                best_ret = max(best_ret, float(v))
            if 'normalized_score' in k:
                best_nor_ret = max(best_nor_ret, float(v))

        # NOTE: changed logging code to use writer
        for k, v in logs.items():
            writer.add_scalar(k, v, self.step)
        writer.add_scalar('train/actor_lr', self.actor_optimizer.param_groups[0]['lr'], self.step)
        writer.add_scalar('train/critic_lr', self.critic_optimizer.param_groups[0]['lr'], self.step)

        logs['best_return_mean'] = best_ret
        logs['best_normalized_score'] = best_nor_ret

        # end of logging change
        return logs
    
    def scale_up_eta(self, lambda_):
        if lambda_ != 0.0:
            self.eta2 = self.eta2 / lambda_  
        
    # NOTE: changed function input to a batch dictionary instead of individual arguments
    def train_step(self, batch, writer=None, loss_metric={}):
        '''
            Train the model for one step
            states: (batch_size, max_len, state_dim)
        '''

        # NOTE: load terms from batch dictionary
        # timesteps are extracted from batches rather than computed on the spot
        # For now rtg and agent_adv are treated as the same thing
        states, actions, rewards, action_target, dones, rtg, timesteps, attention_mask, ot_weights, mmd_weights, mmd_costs = (
            batch['observations'],
            batch['actions'],
            batch['rewards'],
            batch['actions'],
            batch['terminals'],
            batch['agent_advs'],
            batch['timesteps'],
            batch['masks'],
            batch['ot_weights'],
            batch['mmd_weights'],
            batch['mmd_costs'],
        )

        # print("before normalization:")
        # print(states)
        # print("adv: ", rtg)

        # # NOTE: load state mean and std from source buffer, may move this step to experiment.py later
        # state_mean = convert_to_tensor(self.src_buffer.obs_mean, self.device)
        # state_std = convert_to_tensor(self.src_buffer.obs_std, self.device)
        # states = (states - state_mean) / (state_std + 1e-8)

        # TODO: maybe can replace with convert_to_tensor calls
        # states = torch.from_numpy(states).to(dtype=torch.float32, device=self.device)
        # actions = torch.from_numpy(actions).to(dtype=torch.float32, device=self.device)
        # rewards = torch.from_numpy(rewards).to(dtype=torch.float32, device=self.device)
        # action_target = torch.from_numpy(action_target).to(dtype=torch.float32, device=self.device)
        # dones = torch.from_numpy(dones).to(dtype=torch.long, device=self.device)
        # rtg = torch.from_numpy(rtg).to(dtype=torch.float32, device=self.device)
        # timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=self.device)
        # attention_mask = torch.from_numpy(attention_mask).to(dtype=torch.float32, device=self.device)
        
        # NOTE: use the same timesteps for all the states in the batch
        if self.fixed_timestep:
            timesteps = torch.arange(
                states.shape[1], 
                dtype=torch.int64, 
                device=states.device
            ).unsqueeze(0).repeat(states.shape[0], 1)

        batch_size = states.shape[0]
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        device = states.device


        '''Q Training'''
        current_q1, current_q2 = self.critic.forward(states, actions)

        T = current_q1.shape[1]
        repeat_num = 10

        if self.max_q_backup:
            states_rpt = torch.repeat_interleave(states, repeats=repeat_num, dim=0)
            actions_rpt = torch.repeat_interleave(actions, repeats=repeat_num, dim=0)
            rewards_rpt = torch.repeat_interleave(rewards, repeats=repeat_num, dim=0)
            noise = torch.zeros(1, 1, 1)
            noise = torch.cat([noise, torch.randn(repeat_num-1, 1, 1)], dim=0).repeat(batch_size, 1, 1).to(device) # keep rtg logic
            rtg_rpt = torch.repeat_interleave(rtg, repeats=repeat_num, dim=0)
            # NOTE: rtg_rpt has one less timestep: [:, -2:-1] -> [:, -1:]
            rtg_rpt[:, -1:] = rtg_rpt[:, -1:] + noise * 0.1
            timesteps_rpt = torch.repeat_interleave(timesteps, repeats=repeat_num, dim=0)
            attention_mask_rpt = torch.repeat_interleave(attention_mask, repeats=repeat_num, dim=0)
            # NOTE: rtg_rpt has one less timestep: rtg_rpt[:, -1:] -> rtg_rpt
            _, next_action, _ = self.ema_model(
                states_rpt, actions_rpt, rewards_rpt, None, rtg_rpt, timesteps_rpt, attention_mask=attention_mask_rpt,
            )
        else:
            # NOTE: rtg_rpt has one less timesteprtg_rpt: [:, -1:] -> rtg_rpt
            _, next_action, _ = self.ema_model(
                states, actions, rewards, action_target, rtg, timesteps, attention_mask=attention_mask,
            )

        if self.k_rewards:
            # k-step rewards
            if self.max_q_backup:
                critic_next_states = states_rpt[:, -1]
                next_action_ = next_action[:, -1]
                target_q1, target_q2 = self.critic_target(critic_next_states, next_action_)
                target_q1 = target_q1.view(batch_size, repeat_num).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, repeat_num).max(dim=1, keepdim=True)[0]
            else:
                # NOTE: choose this way to compute the target q
                critic_next_states = states[:, -1]
                next_action_ = next_action[:, -1]
                target_q1, target_q2 = self.critic_target(critic_next_states, next_action_)
            target_q = torch.min(target_q1, target_q2) # [B, 1]

            not_done =(1 - dones[:, -1]) # [B, 1]
            if self.use_discount:
                # k-step rewards with discount
                rewards[:, -1] = 0.
                mask_ = attention_mask.sum(dim=1).detach().cpu() # [B]
                discount = [i - 1 - torch.arange(i) for i in mask_]
                discount = torch.stack([torch.cat([i, torch.zeros(T - len(i))], dim=0) for i in discount], dim=0) # [B, T]
                discount = (self.discount ** discount).unsqueeze(-1).to(device) # [B, T, 1]
                k_rewards = torch.cumsum(rewards.flip(dims=[1]) * discount, dim=1).flip(dims=[1]) # [B, T, 1]

                discount = [torch.arange(i) for i in mask_] # 
                discount = torch.stack([torch.cat([torch.zeros(T - len(i)), i], dim=0) for i in discount], dim=0)
                discount =  (self.discount ** discount).unsqueeze(-1).to(device)
                k_rewards = k_rewards / discount
                
                discount = [i - 1 - torch.arange(i) for i in mask_] # [B]
                discount = torch.stack([torch.cat([torch.zeros(T - len(i)), i], dim=0) for i in discount], dim=0)
                discount = (self.discount ** discount).to(device) # [B, T]
                target_q = (k_rewards + (not_done * discount * target_q).unsqueeze(-1)).detach() # [B, T, 1]
                
            else:
                # NOTE: rtg_rpt has one less timestep
                # rtg_rpt[:, -2:-1] -> rtg_rpt[:, -1:]
                # rtg_rpt[:, -1:] -> rtg_rpt
                k_rewards = (rtg - rtg[:, -1:]) * self.scale # [B, T, 1]
                target_q = (k_rewards + (not_done * target_q).unsqueeze(-1)).detach() # [B, T, 1]
        else:
            if self.max_q_backup:
                target_q1, target_q2 = self.critic_target(states_rpt, next_action_) # [B*repeat, T, 1]
                target_q1 = target_q1.view(batch_size, repeat_num, T, 1).max(dim=1)[0]
                target_q2 = target_q2.view(batch_size, repeat_num, T, 1).max(dim=1)[0]
            else:
                target_q1, target_q2 = self.critic_target(states, next_action_) # [B, T, 1]
            target_q = torch.min(target_q1, target_q2) # [B, T, 1]
            target_q = rewards[:, :-1] + self.discount * target_q[:, 1:]
            target_q = torch.cat([target_q, torch.zeros(batch_size, 1, 1).to(device)], dim=1) 

        # Apply weights before boolean indexing
        # if self.use_mmd_weights:
        #     weighted_loss1 = mmd_weights[:, :-1] * (current_q1[:, :-1] - target_q[:, :-1]) ** 2
        #     weighted_loss2 = mmd_weights[:, :-1] * (current_q2[:, :-1] - target_q[:, :-1]) ** 2
        # else:
        #     weighted_loss1 = ot_weights[:, :-1] * (current_q1[:, :-1] - target_q[:, :-1]) ** 2
        #     weighted_loss2 = ot_weights[:, :-1] * (current_q2[:, :-1] - target_q[:, :-1]) ** 2

        weighted_loss1 = (current_q1[:, :-1] - target_q[:, :-1]) ** 2
        weighted_loss2 = (current_q2[:, :-1] - target_q[:, :-1]) ** 2
        if self.loss_weighting_type == 'mmd':
            weighted_loss1 = weighted_loss1
            weighted_loss2 = weighted_loss2
        elif self.loss_weighting_type == 'ot' or self.loss_weighting_type == 'original':
            weighted_loss1 = ot_weights[:, :-1] * weighted_loss1
            weighted_loss2 = ot_weights[:, :-1] * weighted_loss2

        # if self.use_mmd_weights:
        #     weighted_loss1 = mmd_weights[:, :-1] * (current_q1[:, :-1] - target_q[:, :-1]) ** 2
        #     weighted_loss2 = mmd_weights[:, :-1] * (current_q2[:, :-1] - target_q[:, :-1]) ** 2
        # else:
        #     weighted_loss1 = ot_weights[:, :-1] * (current_q1[:, :-1] - target_q[:, :-1]) ** 2
        #     weighted_loss2 = ot_weights[:, :-1] * (current_q2[:, :-1] - target_q[:, :-1]) ** 2
        
        # Then apply attention mask
        critic_loss1 = weighted_loss1[attention_mask[:, :-1] > 0]
        critic_loss2 = weighted_loss2[attention_mask[:, :-1] > 0]
        critic_loss = (critic_loss1 + critic_loss2).mean()
        
        if self.use_cql_loss:
            cql_penalty, cql_info = self.compute_cql_penalty(
                current_q1,
                current_q2,
                states=states,
                actions=actions,
                action_preds=next_action,
                attention_mask=attention_mask,
                ot_weights=ot_weights,
                mmd_weights=mmd_weights,
                num_random=getattr(self, "cql_num_random", 10),
                temp=getattr(self, "cql_temp", 1.0),
                use_mmd_enhanced_weights=getattr(self, "use_mmd_enhanced_weights", False),
                action_low=self.min_action, action_high=self.max_action,         
                chunk_size=None,
            )
            critic_loss = critic_loss + getattr(self, "cql_weight", 1e-1) * cql_penalty

        # old critic loss
        # critic_loss = F.mse_loss(current_q1[:, :-1][attention_mask[:, :-1]>0], target_q[:, :-1][attention_mask[:, :-1]>0]) \
        #     + F.mse_loss(current_q2[:, :-1][attention_mask[:, :-1]>0], target_q[:, :-1][attention_mask[:, :-1]>0]) 

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        '''Policy Training'''        
        # NOTE: rtg_rpt has one less timestep: rtg_rpt[:, -1:] -> rtg_rpt
        state_preds, action_preds, reward_preds = self.actor.forward(
            states, actions, rewards, action_target, rtg, timesteps, attention_mask=attention_mask,
        )

        action_preds_ = action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        action_target_ = action_target.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        # if self.loss_weighting_type == 'mmd' or self.use_mmd_weights:
        #     mmd_weights_ = mmd_weights.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        #     action_loss = (mmd_weights_ * (action_preds_ - action_target_)**2).mean()
        if self.loss_weighting_type == 'ot' or self.loss_weighting_type == 'original':
            ot_weights_ = ot_weights.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            action_loss = (ot_weights_ * (action_preds_ - action_target_)**2).mean()
        else:
            action_loss = ((action_preds_ - action_target_)**2).mean()
        
        # NOTE: mmd_costs is used for adjusting the weight of the src data in the state loss
        # NOTE Here we do not use the first column of state_target, so delete the first column from the mmd_weights
        state_preds = state_preds[:, :-1]
        state_target = states[:, 1:]
        mmd_weights_ = mmd_weights[:, 1:]
        if self.loss_weighting_type == 'ot' or self.loss_weighting_type == 'original':
            ot_weights_ = ot_weights[:, 1:]
            states_loss = (ot_weights_ * (state_preds - state_target) ** 2)[attention_mask[:, 1:]>0].mean()
        else:
            states_loss = ((state_preds - state_target) ** 2)[attention_mask[:, 1:]>0].mean()
        # states_loss = (mmd_weights_ * (state_preds - state_target) ** 2)[attention_mask[:, 1:]>0].mean()
        if reward_preds is not None:
            reward_preds = reward_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            reward_target = rewards.reshape(-1, 1)[attention_mask.reshape(-1) > 0] / self.scale
            if self.loss_weighting_type == 'ot' or self.loss_weighting_type == 'original':
                rewards_loss = (ot_weights_ * (reward_preds - reward_target) ** 2).mean()
            else:
                rewards_loss = ((reward_preds - reward_target) ** 2).mean()
            # rewards_loss = (mmd_weights_ * (reward_preds - reward_target) ** 2).mean()
        else:
            rewards_loss = 0
        bc_loss = action_loss + states_loss + rewards_loss

        # NOTE: mmd_weights and ot_weights are used for adjusting the weight of the src data in the state loss
        actor_states = states.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        q1_new_action, q2_new_action = self.critic(actor_states, action_preds_)
        if self.use_weighted_qloss:
            mmd_weights_ = mmd_weights.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            ot_weights_ = ot_weights.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            if self.use_mmd_enhanced_qloss:
                factor = 1
                if self.loss_weighting_type == 'mmd':
                    # factor = mmd_weights_
                    pass
                elif self.loss_weighting_type == 'ot' or self.loss_weighting_type == 'original':
                    factor = ot_weights_
                q1_new_action = q1_new_action.reshape(-1, 1) * factor
                q2_new_action = q2_new_action.reshape(-1, 1) * factor
        if np.random.uniform() > 0.5:
            q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        
        if not self.remove_critic_loss:
            actor_loss = self.eta2 * bc_loss + self.eta * q_loss
        else:
            actor_loss = bc_loss
        
        # whether to add policy regularization
        if self.pi_reg:
            #! states are already normalized
            #! check if states and action_preds are aligned
            if not self.use_full_pi_reg:
                log_beta = self.vae_policy.dataset_prob(
                    states[:, -1].unsqueeze(-2), 
                    action_preds[:, -1].unsqueeze(-2), 
                    beta=self.vae_beta, 
                    num_samples=10)  # beta = 0.4
                log_beta = log_beta.reshape(-1, 1)[attention_mask[:, -1].reshape(-1) > 0]
            else:
                log_beta = self.vae_policy.dataset_prob(states, action_preds, beta=self.vae_beta, num_samples=10)  # beta = 0.4
                log_beta = log_beta.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            actor_loss -= self.pi_reg_weight * log_beta.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0: 
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.actor_optimizer.step()

        """ Step Target network """
        self.step_ema()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.step += 1

        # NOTE: logging updates
        with torch.no_grad():
            self.diagnostics['train/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
            writer.add_scalar('step/action_error', self.diagnostics['train/action_error'], self.step)

        if writer is not None:
            if self.grad_norm > 0:
                writer.add_scalar('step/actor_grad_norm', actor_grad_norms.max().item(), self.step)
                writer.add_scalar('step/critic_grad_norm', critic_grad_norms.max().item(), self.step)
            writer.add_scalar('step/bc_loss', bc_loss.item(), self.step)
            writer.add_scalar('step/ql_loss', q_loss.item(), self.step)
            writer.add_scalar('step/actor_loss', actor_loss.item(), self.step)
            writer.add_scalar('step/critic_loss', critic_loss.item(), self.step)
            writer.add_scalar('step/target_q_mean', target_q.mean().item(), self.step)
            if self.use_cql_loss:
                writer.add_scalar('step/cql_penalty', cql_penalty.item(), self.step)
                writer.add_scalar('step/cql_q_data_mean', cql_info['q_data_mean'], self.step)
                writer.add_scalar('step/cql_q_pi_mean',   cql_info['q_pi_mean'],   self.step)
                writer.add_scalar('step/cql_q_rand_mean', cql_info['q_rand_mean'], self.step)
        # end of logging change
        
        loss_metric['bc_loss'].append(bc_loss.item())
        loss_metric['ql_loss'].append(q_loss.item())
        loss_metric['critic_loss'].append(critic_loss.item())
        loss_metric['actor_loss'].append(actor_loss.item())
        loss_metric['target_q_mean'].append(target_q.mean().item())
        loss_metric['actor_grad_norm'] = actor_grad_norms.max().item() if self.grad_norm > 0 else 0
        loss_metric['critic_grad_norm'] = critic_grad_norms.max().item() if self.grad_norm > 0 else 0
        if self.use_cql_loss:
            loss_metric['cql_penalty'] = cql_penalty.item()

        return loss_metric