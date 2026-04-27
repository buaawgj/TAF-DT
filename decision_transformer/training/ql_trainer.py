import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import copy
import os
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Union, List, Callable, Dict, Any
from operator import itemgetter

from decision_transformer.misc.utils import (
    convert_to_tensor, expectile_regression, make_target, 
    compute_action_weight, concat_tensor_dict, compute_gae,
    compute_position_ids)


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
                task, 
                critic,
                critic_q,
                critic_v,
                #! use the SingleCritic for the command network
                command,
                command_type,
                batch_size, 
                pretrain_batch_size,
                pretrain_command_epoch,
                tau,
                discount,
                src_buffer,
                tar_buffer,
                loss_fn, 
                lambda_,
                adv_scale,
                delayed_reward,
                use_mean_reduce,
                eval_fns=None,
                max_q_backup=False,
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
                load_path=False,
                proportion=0.25,
            ):
        
        self.task = task
        self.actor = model
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)

        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.src_buffer = src_buffer
        self.tar_buffer = tar_buffer 
        
        self.load_path = load_path
        self.v_target = v_target
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

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=lr_min)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=lr_min)

        self.batch_size = batch_size
        self.pretrain_batch_size = pretrain_batch_size
        self.pretrain_command_epoch = pretrain_command_epoch
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
        # parameters for the relabeling of the buffers
        self.lambda_ = lambda_
        self.adv_scale = adv_scale
        self.delayed_reward = delayed_reward
        self.use_mean_reduce = use_mean_reduce
        self.proportion = proportion

        self.start_time = time.time()
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
        #!!! Add mmd_costs and ot_costs to src_batch
        src_s, src_a, src_r, src_next_s, src_terminals, src_masks, mmd_costs, ot_costs = itemgetter(
            "observations", "actions", "rewards", "next_observations", "terminals", "masks", "mmd_costs", "ot_costs"
        )(src_batch)
        tar_s, tar_a, tar_r, tar_next_s, tar_terminals, tar_masks = itemgetter(
            "observations", "actions", "rewards", "next_observations", "terminals", "masks", 
        )(tar_batch)
        
        if compute_adv:
            src_advs = src_batch.get("agent_advs", None)
            tar_advs = tar_batch.get("agent_advs", None)
            
        # normalization
        mmd_costs = (mmd_costs - torch.max(mmd_costs))/(torch.max(mmd_costs) - torch.min(mmd_costs))
        ot_costs = (ot_costs - torch.max(ot_costs))/(torch.max(ot_costs) - torch.min(ot_costs))
        
        # filter out transitions
        batch_size = src_s.shape[0]
        src_filter_num = int(batch_size * self.proportion)
        
        filter_cost, indices = torch.topk(mmd_costs, src_filter_num)
        
        src_s = src_s[indices]
        src_a = src_a[indices]
        src_next_s = src_next_s[indices]
        src_r = src_r[indices]
        src_terminals = src_terminals[indices]
        mmd_costs = mmd_costs[indices]
        ot_costs = ot_costs[indices]
        src_masks = src_masks[indices]
        if compute_adv:
            src_advs = src_advs[indices]
        
        state = torch.cat([src_s, tar_s], 0)
        action = torch.cat([src_a, tar_a], 0)
        next_state = torch.cat([src_next_s, tar_next_s], 0)
        reward = torch.cat([src_r, tar_r], 0)
        terminal = torch.cat([src_terminals, tar_terminals], 0)
        mask = torch.cat([src_masks, tar_masks], 0)
        if compute_adv:
           adv = torch.cat([src_advs, tar_advs], 0)
        
        weight = torch.ones_like(reward.flatten()).to(self.device)
        if compute_weight:
            # calculate cost weight
            cost_weight = torch.exp(ot_costs)
            weight[:src_s.shape[0]] = cost_weight

            weight = self.weight.unsqueeze(1)
            
            #! A potential problem: scale the reward by the weight
            reward = reward * weight
            
        if compute_adv:
            return state, action, next_state, reward, terminal, mask, weight, mmd_costs, ot_costs, adv
            
        return state, action, next_state, reward, terminal, mask, weight, mmd_costs, ot_costs
            
    def update_critic(self, states, actions, rewards, next_states, terminals, masks, weights):   
        if self.v_target:
            # Use the batch to compute the q loss
            with torch.no_grad():
                target = self.critic_v_target.v_min(next_states)
                target = rewards + self._discount * (1-terminals) * target
            q_pred1, q_pred2 = self.critic_q(states, actions, reduce=False)
            q_loss = weights * ((target - q_pred1)**2 + (target - q_pred2)**2)
            #! do we need the unsqueeze function here?
            q_loss = (q_loss * masks.unsqueeze(-1)).mean()
            
            self.critic_q_optim.zero_grad()
            q_loss.backward()
            self.critic_q_optim.step()
            
            # Use the batch to compute the value loss
            v_pred1, v_pred2 = self.critic_v(states, reduce=False)
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
                q_target = rewards + self._discount * (1-terminals) * q_target
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
        
    def pretrain_q_and_v(self, num_workers=1, pin_memory=False):
        """ Pretrain the critic q and v networks for computing the RTG of states in the buffers."""
        src_buffer_iter = iter(DataLoader(self.src_buffer, batch_size=self.pretrain_batch_size, pin_memory=pin_memory, num_workers=num_workers))
        tar_buffer_iter = iter(DataLoader(self.tar_buffer, batch_size=self.pretrain_batch_size, pin_memory=pin_memory, num_workers=num_workers))
        if self.load_path is None:
            for i_epoch in trange(1, self.max_iters+1, desc="pretrain critic"):
                for i_step in range(self.num_steps_per_iter):
                    #! How to get a batch of data from the source and target buffers
                    src_batch = next(src_buffer_iter)
                    tar_batch = next(tar_buffer_iter)
                    
                    state, action, next_state, reward, terminal, mask, weight, mmd_costs, ot_costs = self.concat_src_tar_batch(
                        src_batch, tar_batch, compute_weight=True)
                    
                    train_metrics = self.update_critic(state, action, next_state, reward, terminal, mask, weight)
                    
                # log loss information
                # if i_epoch % 10 == 0:
                    # # logger.info(f"Epoch {i_epoch}: \n{train_metrics}")
                    # logger.log_scalars("pretrain", train_metrics, step=i_epoch)
            
            if self.save_path is not None:
                save_path = self.save_path
                if not os.path.exists(save_path): 
                    os.makedirs(save_path)
                torch.save(self.critic_q.state_dict(), os.path.join(save_path, "critic_q.pt"))
                torch.save(self.critic_v.state_dict(), os.path.join(save_path, "critic_v.pt"))
                
        elif self.load_path is not None:
            load_path = self.load_path
            device = self.actor.device
            self.critic_q.load_state_dict(torch.load(os.path.join(load_path, "critic_q.pt"), map_location="cpu"))
            self.critic_v.load_state_dict(torch.load(os.path.join(load_path, "critic_v.pt"), map_location="cpu"))
            self.critic_q.to(device)
            self.critic_v.to(device)
    
    @torch.no_grad()        
    def relabel_buffer(self, buffer):
        """ Relabel the RTG of the states in the buffer using the critic q and v networks."""
        values = np.zeros_like(buffer.reward)
        agent_advs = np.zeros_like(buffer.reward)
        traj_start = 0
        device = self.actor.device
        traj_num = len(buffer.traj_len)
        with torch.no_grad():
            for i in range(traj_num):
                if self.lambda_ is None:
                    traj_len = buffer.traj_len[i]
                    obss = torch.from_numpy(buffer.state[i, :traj_len]).to(device)
                    actions = torch.from_numpy(buffer.action[i, :traj_len]).to(device)
                    if self.use_mean_reduce:
                        vs = self.critic_v.v_mean(obss).detach().cpu().numpy()
                        qs = self.critic_q.q_mean(obss, actions).detach().cpu().numpy()
                    else:
                        vs = self.critic_v.v_min(obss).detach().cpu().numpy()
                        qs = self.critic_q.q_min(obss, actions).detach().cpu().numpy()
                    values[i, :traj_len] = vs
                    agent_advs[i, :traj_len] = qs - vs
                else:
                    traj_len = buffer.traj_len[i]
                    rewards = buffer.reward[i, :traj_len]
                    mask = None
                    if "mask" in buffer.keys():
                        mask = buffer.mask[i, :traj_len]
                    obss = torch.from_numpy(buffer.observations[i, :traj_len]).to(device)
                    last_obs = torch.from_numpy(buffer.next_observations[i, traj_len-1:traj_len]).to(device)
                    if self.use_mean_reduce:
                        vs = self.critic_v.v_mean(obss).detach().cpu().numpy()
                        last_v = self.critic_v(last_obs).cpu().numpy()
                    else:
                        vs = self.critic_v.v_min(obss).detach().cpu().numpy()
                        last_v = self.critic_v.v_min(last_obs).cpu().numpy()
                    gae, _ = compute_gae(rewards, vs, last_v, gamma=self.discount, lam=self.lambda_, dim=-1, mask=mask)
                    values[i, :traj_len] = vs
                    agent_advs[i, :traj_len] = gae
                    
        if self.adv_scale is not None:
            if self.delayed_reward:
                max_adv = agent_advs.max()
                agent_advs = agent_advs / max_adv * self.adv_scale
            else:
                advs_ = self.agent_advs[self.agent_advs!=0]
                std_ = advs_.std()
                agent_advs = (agent_advs / std_) * self.adv_scale
        
        buffer["agent_advs"] = agent_advs
        buffer["values"] = values
            
    def train_rtg_network(self, step_per_epoch, pin_memory=False, num_workers=0):
        """ Train a network for predicting the reward to go (RTG) of states generated during training"""
        # Compute the RTG for the source and target buffers
        self.relabel_buffer(self.src_buffer)
        
        # define the data loaders for the source and target buffers
        src_buffer_iter = iter(DataLoader(self.src_buffer, batch_size=self.pretrain_batch_size, pin_memory=pin_memory, num_workers=num_workers))
        tar_buffer_iter = iter(DataLoader(self.tar_buffer, batch_size=self.pretrain_batch_size, pin_memory=pin_memory, num_workers=num_workers))

        if self.command_type == "in_sample_max":
            # offline_buffer_iter = iter(DataLoader(offline_buffer, batch_size=args.pretrain_command_batch_size, pin_memory=pin_memory, num_workers=num_workers))
            for i_epoch in trange(1, self.pretrain_command_epoch+1, desc="pretrain_command"):
                for i_step in range(step_per_epoch):
                    # batch = next(offline_buffer_iter)
                    # src_batch = next(src_buffer_iter)
                    # tar_batch = next(tar_buffer_iter)
                    try:
                        src_batch = next(src_buffer_iter)
                        tar_batch = next(tar_buffer_iter)
                    except StopIteration:
                        # 重新创建迭代器（或跳过、报错等处理）
                        src_buffer_iter = iter(DataLoader(self.src_buffer, batch_size=self.pretrain_batch_size, pin_memory=pin_memory, num_workers=num_workers))
                        tar_buffer_iter = iter(DataLoader(self.tar_buffer, batch_size=self.pretrain_batch_size, pin_memory=pin_memory, num_workers=num_workers))
                        src_batch = next(src_buffer_iter)
                        tar_batch = next(tar_buffer_iter)
                    
                    batch = concat_tensor_dict([src_batch, tar_batch])
                    train_metrics = self.update_command(batch)
            #     if i_epoch % 10 == 0:
            #         logger.log_scalars("pretrain", train_metrics, step=i_epoch)
            # del offline_buffer_iter
                print("Gap between the true and predicted commands: ", train_metrics[f"{self.command.id}_ISM_gap"])
                print("Loss of the command network: ", train_metrics[f"{self.command.id}_ISM_loss"])

            if self.save_path is not None:
                save_path = os.path.join(self.save_path, self.task)
                if not os.path.exists(save_path): 
                    os.makedirs(save_path)
                torch.save(self.command.state_dict(), os.path.join(save_path, "command.pt"))
                
        elif self.command_type == "constant":
            # For constant command, we only the mean of the positive agent advantages of the target buffer.
            self.command.set_value(self.tar_buffer.agent_advs[self.tar_buffer.agent_advs > 0.01].mean())
            print("Final command value: ", self.command.constant.item())
            
        del src_buffer_iter
        del tar_buffer_iter

    def train_iteration(self, num_steps, logger, batch_size, pin_memory, num_workers, iter_num=0, log_writer=None):

        logs = dict()

        train_start = time.time()
        
        self.src_buffer_iter = iter(
            DataLoader(self.src_buffer, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
            ) if not hasattr(self, 'src_buffer_iter') else self.src_buffer_iter
        self.tar_buffer_iter = iter(
            DataLoader(self.tar_buffer, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
            ) if not hasattr(self, 'tar_buffer_iter') else self.tar_buffer_iter

        self.actor.train()
        self.critic.train()
        loss_metric = {
            'bc_loss': [],
            'ql_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'target_q_mean': [],
        }
        for _ in trange(num_steps):
            try:
                src_batch = next(self.src_buffer_iter)
                tar_batch = next(self.tar_buffer_iter)
            except StopIteration:
                # 重新创建迭代器（或跳过、报错等处理）
                self.src_buffer_iter = iter(DataLoader(self.src_buffer, batch_size=self.pretrain_batch_size, pin_memory=pin_memory, num_workers=num_workers))
                self.tar_buffer_iter = iter(DataLoader(self.tar_buffer, batch_size=self.pretrain_batch_size, pin_memory=pin_memory, num_workers=num_workers))
                src_batch = next(self.src_buffer_iter)
                tar_batch = next(self.tar_buffer_iter)
                
            state, action, next_state, reward, terminal, mask, weight, mmd_costs, ot_costs, adv = self.concat_src_tar_batch(
                src_batch, tar_batch, compute_weight=True, compute_adv=True)
            loss_metric = self.train_step(state, action, reward, action, terminal, adv, mask, weight,log_writer, loss_metric)
        
        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
        logger.record_tabular('QL Loss', np.mean(loss_metric['ql_loss']))
        logger.record_tabular('Actor Loss', np.mean(loss_metric['actor_loss']))
        logger.record_tabular('Critic Loss', np.mean(loss_metric['critic_loss']))
        logger.record_tabular('Target Q Mean', np.mean(loss_metric['target_q_mean']))
        logger.dump_tabular()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()


        self.actor.eval()
        self.critic.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.actor, self.critic_target)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        logger.log('=' * 80)
        logger.log(f'Iteration {iter_num}')
        best_ret = -10000
        best_nor_ret = -10000
        for k, v in logs.items():
            if 'return_mean' in k:
                best_ret = max(best_ret, float(v))
            if 'normalized_score' in k:
                best_nor_ret = max(best_nor_ret, float(v))
            logger.record_tabular(k, float(v))
        logger.record_tabular('Current actor learning rate', self.actor_optimizer.param_groups[0]['lr'])
        logger.record_tabular('Current critic learning rate', self.critic_optimizer.param_groups[0]['lr'])
        logger.dump_tabular()

        logs['Best_return_mean'] = best_ret
        logs['Best_normalized_score'] = best_nor_ret
        return logs
    
    def scale_up_eta(self, lambda_):
        self.eta2 = self.eta2 / lambda_

    def train_step(self, states, actions, rewards, action_target, dones, rtg, attention_mask, weights, log_writer=None, loss_metric={}):
        '''
            Train the model for one step
            states: (batch_size, max_len, state_dim)
        '''
        timesteps = compute_position_ids(states, self.task, self.device) # [B, T]
        # action_target = torch.clone(actions)
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
            rtg_rpt[:, -2:-1] = rtg_rpt[:, -2:-1] + noise * 0.1
            timesteps_rpt = torch.repeat_interleave(timesteps, repeats=repeat_num, dim=0)
            attention_mask_rpt = torch.repeat_interleave(attention_mask, repeats=repeat_num, dim=0)
            _, next_action, _ = self.ema_model(
                states_rpt, actions_rpt, rewards_rpt, None, rtg_rpt[:,:-1], timesteps_rpt, attention_mask=attention_mask_rpt,
            )
        else:
            #! QT chooses this way to compute the next action
            # print("states shape", states.shape)
            # print("actions shape", actions.shape)
            # print("rewards shape", rewards.shape)
            # print("rtg shape", rtg.shape)
            # print("timesteps shape", timesteps.shape)
            # print("attention_mask shape", attention_mask.shape)
            _, next_action, _ = self.ema_model(
                states, actions, rewards, action_target, rtg[:,:-1], timesteps, attention_mask=attention_mask,
            )

        if self.k_rewards:
            #! QT uses the k-rewards to compute the target q
            if self.max_q_backup:
                critic_next_states = states_rpt[:, -1]
                next_action = next_action[:, -1]
                target_q1, target_q2 = self.critic_target(critic_next_states, next_action)
                target_q1 = target_q1.view(batch_size, repeat_num).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, repeat_num).max(dim=1, keepdim=True)[0]
            else:
                #! QT uses the last state to compute the target q
                critic_next_states = states[:, -1]
                next_action = next_action[:, -1]
                target_q1, target_q2 = self.critic_target(critic_next_states, next_action)
            target_q = torch.min(target_q1, target_q2) # [B, 1]

            not_done =(1 - dones[:, -1]) # [B, 1]
            if self.use_discount:
                #! QT chooses this way to compute the k-rewards
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
                k_rewards = (rtg[:,:-1] - rtg[:, -2:-1])* self.scale # [B, T, 1]
                target_q = (k_rewards + (not_done * target_q).unsqueeze(-1)).detach() # [B, T, 1]
        else:
            if self.max_q_backup:
                target_q1, target_q2 = self.critic_target(states_rpt, next_action) # [B*repeat, T, 1]
                target_q1 = target_q1.view(batch_size, repeat_num, T, 1).max(dim=1)[0]
                target_q2 = target_q2.view(batch_size, repeat_num, T, 1).max(dim=1)[0]
            else:
                target_q1, target_q2 = self.critic_target(states, next_action) # [B, T, 1]
            target_q = torch.min(target_q1, target_q2) # [B, T, 1]
            target_q = rewards[:, :-1] + self.discount * target_q[:, 1:]
            target_q = torch.cat([target_q, torch.zeros(batch_size, 1, 1).to(device)], dim=1) 
        
        #! why use :-1 here?
        critic_loss1 = weights[:, :-1] * (current_q1[:, :-1][attention_mask[:, :-1]>0] - target_q[:, :-1][attention_mask[:, :-1]>0]) ** 2
        critic_loss2 = weights[:, :-1] * (current_q2[:, :-1][attention_mask[:, :-1]>0] - target_q[:, :-1][attention_mask[:, :-1]>0]) ** 2
        critic_loss = (critic_loss1 + critic_loss2).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        '''Policy Training'''        
        state_preds, action_preds, reward_preds = self.actor.forward(
            states, actions, rewards, action_target, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        action_preds_ = action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        action_target_ = action_target.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        weights_ = weights.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        action_loss = (weights_ * (action_preds_ - action_target_)**2).mean()
        state_preds = state_preds[:, :-1]
        state_target = states[:, 1:]
        states_loss = ((state_preds - state_target) ** 2)[attention_mask[:, :-1]>0].mean()
        if reward_preds is not None:
            reward_preds = reward_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            reward_target = rewards.reshape(-1, 1)[attention_mask.reshape(-1) > 0] / self.scale
            rewards_loss = (weights_ * (reward_preds - reward_target) ** 2).mean()
        else:
            rewards_loss = 0
        # bc_loss = F.mse_loss(action_preds_, action_target_) + states_loss + rewards_loss
        bc_loss = action_loss + states_loss + rewards_loss

        actor_states = states.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        q1_new_action, q2_new_action = self.critic(actor_states, action_preds_)
        # q1_new_action, q2_new_action = self.critic(state_target, action_preds_)
        if np.random.uniform() > 0.5:
            q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            
        #! add a dynamic weight for the BC loss
        actor_loss = self.eta2 * bc_loss + self.eta * q_loss
        # actor_loss = self.eta * bc_loss + q_loss

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

        with torch.no_grad():
            self.diagnostics['train/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        if log_writer is not None:
            if self.grad_norm > 0:
                log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
            log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
            log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
            log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
            log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

        loss_metric['bc_loss'].append(bc_loss.item())
        loss_metric['ql_loss'].append(q_loss.item())
        loss_metric['critic_loss'].append(critic_loss.item())
        loss_metric['actor_loss'].append(actor_loss.item())
        loss_metric['target_q_mean'].append(target_q.mean().item())
        loss_metric['actor_grad_norm'] = actor_grad_norms.max().item() if self.grad_norm > 0 else 0
        loss_metric['critic_grad_norm'] = critic_grad_norms.max().item() if self.grad_norm > 0 else 0

        return loss_metric
