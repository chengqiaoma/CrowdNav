import torch
import math
import torch.nn as nn
import torch.optim as optim
from .lagrange import Lagrange


class PPOLag():
    """
    [Volatility-Aware NCBF-PPO]
    融合历史波动率特征 (Volatility ACI) 与神经控制障碍函数 (NCBF) 约束的 PPO。

    本版本使用更贴合 crowd navigation 的 per-human near-future risk auxiliary。
    """

    def __init__(self,
                 actor_critic,
                 cost_actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 cost_limit,
                 lag_init=0.0,
                 lag_lr=9e-4,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 cost_loss_coef=1.0,
                 lagrangian_upper_bound=None,
                 **kwargs):

        self.actor_critic = actor_critic
        self.cost_actor_critic = cost_actor_critic
        self.share_reward_cost_encoder = bool(
            kwargs.get('share_reward_cost_encoder', actor_critic is cost_actor_critic)
        )

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.cost_loss_coef = cost_loss_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=eps)
        if self.share_reward_cost_encoder:
            self.cost_optimizer = self.optimizer
        else:
            self.cost_optimizer = optim.Adam(self.cost_actor_critic.parameters(), lr=lr / 2, eps=eps)

        self._lagrange = Lagrange(
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lag_init,
            lambda_lr=lag_lr,
            lambda_optimizer='Adam',
            lagrangian_upper_bound=lagrangian_upper_bound,
            soft_gating_lower_bound=kwargs.get('soft_gating_lower_bound', 0.18),
            soft_gating_upper_bound=kwargs.get('soft_gating_upper_bound', 0.45),
            gate_start_step=kwargs.get('gate_start_step', 15_000_000),
            gate_full_step=kwargs.get('gate_full_step', 35_000_000),
            jc_ema_beta=kwargs.get('jc_ema_beta', 0.90),
            violation_deadband=kwargs.get('violation_deadband', 0.05),
            lambda_kp=kwargs.get('lambda_kp', 0.08),
            lambda_ki=kwargs.get('lambda_ki', 0.01),
            integral_min=kwargs.get('integral_min', -50.0),
            integral_max=kwargs.get('integral_max', 50.0),
            lambda_output_tau=kwargs.get('lambda_output_tau', 0.85),
            device=kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        )

        self.use_pure_linear_lagrangian = kwargs.get("use_pure_linear_lagrangian", False)

        self.aux_per_human_risk_coef = kwargs.get(
            'aux_per_human_risk_coef',
            kwargs.get('aux_danger_coef', 0.03),
        )
        self.aux_current_risk_eta = float(kwargs.get('aux_current_risk_eta', 0.25))
        self.human_risk_pos_thresh = float(kwargs.get('human_risk_pos_thresh', 0.03))
        self.human_risk_pos_weight = float(kwargs.get('human_risk_pos_weight', 4.0))
        self.human_risk_linear_weight = float(kwargs.get('human_risk_linear_weight', 2.0))
        self.human_risk_mid_thresh = float(kwargs.get('human_risk_mid_thresh', 0.10))
        self.human_risk_high_thresh = float(kwargs.get('human_risk_high_thresh', 0.40))
        self.human_risk_mid_weight = float(kwargs.get('human_risk_mid_weight', 5.0))
        self.human_risk_high_weight = float(kwargs.get('human_risk_high_weight', 8.0))
        self.aux_coef_warmup_steps = int(kwargs.get('aux_coef_warmup_steps', 0))
        self.aux_coef_ramp_end_steps = int(kwargs.get('aux_coef_ramp_end_steps', 0))
        self.aux_coef_decay_start_steps = int(kwargs.get('aux_coef_decay_start_steps', 0))
        self.aux_coef_final_scale = float(kwargs.get('aux_coef_final_scale', 1.0))
        self.aux_coef_total_steps = int(kwargs.get('aux_coef_total_steps', 0))
        self.last_aux_stats = {
            'coef': 0.0,
            'target_mean': 0.0,
            'pred_mean': 0.0,
            'valid_slots': 0.0,
            'positive_frac': 0.0,
            'weight_mean': 0.0,
            'target_max': 0.0,
            'pred_max': 0.0,
            'target_p90': 0.0,
            'pred_p90': 0.0,
        }

    def _scheduled_aux_coef(self, global_step):
        base_coef = float(self.aux_per_human_risk_coef)
        if base_coef <= 0.0:
            return 0.0

        warmup = max(0, int(self.aux_coef_warmup_steps))
        ramp_end = max(warmup, int(self.aux_coef_ramp_end_steps))
        decay_start = max(ramp_end, int(self.aux_coef_decay_start_steps))
        total_steps = max(decay_start + 1, int(self.aux_coef_total_steps))
        final_scale = min(max(float(self.aux_coef_final_scale), 0.0), 1.0)
        step = int(global_step)

        if step < warmup:
            return 0.0
        if ramp_end > warmup and step < ramp_end:
            return base_coef * (step - warmup) / max(ramp_end - warmup, 1)
        if step < decay_start:
            return base_coef

        progress = min(max((step - decay_start) / max(total_steps - decay_start, 1), 0.0), 1.0)
        scale = 1.0 - (1.0 - final_scale) * progress
        return base_coef * scale

    def _unwrap_module(self, module):
        return getattr(module, '_orig_mod', module)

    def _get_aux_module(self):
        module = self._unwrap_module(self.actor_critic)
        base = getattr(module, 'base', module)
        return self._unwrap_module(base)

    def update_cost_limit(self, new_limit):
        if hasattr(self, '_lagrange'):
            if hasattr(self._lagrange, 'set_cost_limit'):
                self._lagrange.set_cost_limit(new_limit)
            else:
                self._lagrange.cost_limit = new_limit

    def update(self, rollouts, mean_ep_costs, global_step, success_rate=0.0):
        self._lagrange.update_lagrange_multiplier(mean_ep_costs, global_step, success_rate)
        penalty = self._lagrange.effective_lambda.item()

        raw_advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        raw_cost_advantages = rollouts.cost_returns[:-1] - rollouts.cost_value_preds[:-1]

        if self.use_pure_linear_lagrangian:
            advantages = raw_advantages
            cost_advantages = raw_cost_advantages
        else:
            advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-5)
            cost_advantages = (raw_cost_advantages - raw_cost_advantages.mean()) / (raw_cost_advantages.std() + 1e-5)
            cost_advantages = torch.clamp(cost_advantages, -5.0, 5.0)

        value_loss_epoch = 0.0
        cost_value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        human_risk_loss_epoch = 0.0
        c_human_risk = self._scheduled_aux_coef(global_step)
        aux_module = self._get_aux_module()
        can_reuse_aux_cache = c_human_risk > 0.0 and hasattr(aux_module, 'aux_from_cache')
        stats_device = raw_advantages.device
        aux_target_sum = torch.zeros((), device=stats_device)
        aux_pred_sum = torch.zeros((), device=stats_device)
        aux_positive_sum = torch.zeros((), device=stats_device)
        aux_valid_count = torch.zeros((), device=stats_device)
        aux_weight_sum = torch.zeros((), device=stats_device)
        aux_target_values = []
        aux_pred_values = []

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, cost_advantages, self.num_mini_batch
            )

            for sample in data_generator:
                obs_batch = sample[0]
                recurrent_hidden_states_batch = sample[1]
                cost_recurrent_hidden_states_batch = sample[2]
                actions_batch = sample[3]
                value_preds_batch = sample[4]
                cost_value_preds_batch = sample[5]
                return_batch = sample[6]
                cost_return_batch = sample[7]
                masks_batch = sample[8]
                old_action_log_probs_batch = sample[9]
                adv_targ = sample[10]
                cost_adv_targ = sample[11]
                per_human_future_risk_batch = sample[12]
                per_human_valid_mask_batch = sample[13]

                self.optimizer.zero_grad()
                if not self.share_reward_cost_encoder:
                    self.cost_optimizer.zero_grad()

                actor_hxs_batch = {k: v for k, v in recurrent_hidden_states_batch.items()}
                cost_hxs_batch = (
                    {k: v for k, v in recurrent_hidden_states_batch.items()}
                    if self.share_reward_cost_encoder
                    else cost_recurrent_hidden_states_batch
                )
                try:
                    eval_actor = self.actor_critic.evaluate_actions(
                        obs_batch,
                        actor_hxs_batch,
                        masks_batch,
                        actions_batch,
                        return_aux_cache=can_reuse_aux_cache,
                    )
                except TypeError:
                    eval_actor = self.actor_critic.evaluate_actions(
                        obs_batch,
                        actor_hxs_batch,
                        masks_batch,
                        actions_batch,
                    )
                values, action_log_probs, dist_entropy = eval_actor[0], eval_actor[1], eval_actor[2]
                aux_cache = eval_actor[4] if len(eval_actor) > 4 else None

                eval_cost = self.cost_actor_critic.get_cost_value(
                    obs_batch,
                    cost_hxs_batch,
                    masks_batch
                )
                cost_values = eval_cost[0]

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    v_l1 = (values - return_batch).pow(2)
                    v_l2 = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(v_l1, v_l2).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                cost_value_loss = 0.5 * (cost_return_batch - cost_values).pow(2).mean()

                if self.use_pure_linear_lagrangian:
                    adv_targ_combined = adv_targ - penalty * cost_adv_targ
                else:
                    alpha_max = 0.4
                    c_scale = 0.1
                    alpha = alpha_max * math.tanh(penalty * c_scale)
                    adv_targ_combined = (1.0 - alpha) * adv_targ - alpha * cost_adv_targ

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ_combined
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param
                ) * adv_targ_combined
                action_loss = -torch.min(surr1, surr2).mean()

                aux_loss = 0.0
                step_human_risk_loss = 0.0

                if c_human_risk > 0.0 and (
                    (aux_cache is not None and hasattr(aux_module, 'aux_from_cache'))
                    or hasattr(aux_module, 'aux_forward')
                ):
                    if aux_cache is not None and hasattr(aux_module, 'aux_from_cache'):
                        pred_human_risk, pred_valid_mask = aux_module.aux_from_cache(aux_cache)
                    else:
                        aux_hxs_batch = {k: v for k, v in recurrent_hidden_states_batch.items()}
                        pred_human_risk, pred_valid_mask = aux_module.aux_forward(
                            obs_batch,
                            aux_hxs_batch,
                            masks_batch,
                            actions=actions_batch
                        )

                    target_risk = per_human_future_risk_batch.view_as(pred_human_risk).float().clamp(0.0, 1.0)
                    valid_mask = per_human_valid_mask_batch.view_as(pred_human_risk).float()

                    if pred_valid_mask is not None:
                        valid_mask = valid_mask * pred_valid_mask.view_as(pred_human_risk).float()

                    loss_human_risk_mat = torch.nn.functional.smooth_l1_loss(
                        pred_human_risk,
                        target_risk,
                        reduction='none'
                    )

                    pos_mask = (target_risk > self.human_risk_pos_thresh).float()
                    mid_mask = (target_risk >= self.human_risk_mid_thresh).float()
                    high_mask = (target_risk >= self.human_risk_high_thresh).float()
                    # Piecewise max keeps weak positives useful while making
                    # rare high-risk tails matter enough to avoid range collapse.
                    sample_weight = torch.maximum(
                        torch.ones_like(target_risk),
                        self.human_risk_pos_weight * pos_mask,
                    )
                    sample_weight = torch.maximum(
                        sample_weight,
                        self.human_risk_mid_weight * mid_mask,
                    )
                    sample_weight = torch.maximum(
                        sample_weight,
                        self.human_risk_high_weight * high_mask,
                    )
                    sample_weight = sample_weight + self.human_risk_linear_weight * target_risk
                    weighted_valid_mask = valid_mask * sample_weight

                    loss_human_risk = (
                        (loss_human_risk_mat * weighted_valid_mask).sum()
                        / (weighted_valid_mask.sum() + 1e-5)
                    )

                    aux_loss = c_human_risk * loss_human_risk
                    step_human_risk_loss = loss_human_risk.item()

                    valid_slots = valid_mask.sum().detach()
                    aux_valid_count += valid_slots
                    aux_weight_sum += weighted_valid_mask.sum().detach()
                    aux_target_sum += (target_risk * valid_mask).sum().detach()
                    aux_pred_sum += (pred_human_risk * valid_mask).sum().detach()
                    aux_positive_sum += (pos_mask * valid_mask).sum().detach()

                    valid_entries = valid_mask > 0.5
                    selected_targets = target_risk[valid_entries].detach()
                    selected_preds = pred_human_risk[valid_entries].detach()
                    if selected_targets.numel() > 0:
                        aux_target_values.append(selected_targets)
                        aux_pred_values.append(selected_preds)

                    if e == 0 and global_step < 5000:
                        valid_slots_item = valid_slots.item()
                        debug_target_mean = (
                            (target_risk * valid_mask).sum().item() / (valid_slots_item + 1e-5)
                            if valid_slots_item > 0 else 0.0
                        )
                        debug_pred_mean = (
                            (pred_human_risk * valid_mask).sum().item() / (valid_slots_item + 1e-5)
                            if valid_slots_item > 0 else 0.0
                        )
                        print(
                            f"[DEBUG] per_human_target mean={debug_target_mean:.4f}, "
                            f"pred_human_risk mean={debug_pred_mean:.4f}, "
                            f"valid_slots={valid_slots_item:.1f}"
                        )

                total_actor_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )

                combined_loss = (
                    total_actor_loss
                    + cost_value_loss * self.cost_loss_coef
                    + aux_loss
                )

                combined_loss.backward()

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                if not self.share_reward_cost_encoder:
                    nn.utils.clip_grad_norm_(self.cost_actor_critic.parameters(), self.max_grad_norm)

                self.optimizer.step()
                if not self.share_reward_cost_encoder:
                    self.cost_optimizer.step()

                value_loss_epoch += value_loss.item()
                cost_value_loss_epoch += cost_value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                human_risk_loss_epoch += step_human_risk_loss

        num_updates = self.ppo_epoch * self.num_mini_batch
        aux_valid_count_item = aux_valid_count.item()
        if aux_valid_count_item > 0:
            all_target_values = (
                torch.cat(aux_target_values, dim=0)
                if aux_target_values
                else torch.zeros(1, device=raw_advantages.device)
            )
            all_pred_values = (
                torch.cat(aux_pred_values, dim=0)
                if aux_pred_values
                else torch.zeros(1, device=raw_advantages.device)
            )
            self.last_aux_stats = {
                'target_mean': (aux_target_sum / aux_valid_count).item(),
                'pred_mean': (aux_pred_sum / aux_valid_count).item(),
                'valid_slots': aux_valid_count_item / num_updates,
                'positive_frac': (aux_positive_sum / aux_valid_count).item(),
                'weight_mean': (aux_weight_sum / aux_valid_count).item(),
                'target_max': all_target_values.max().item(),
                'pred_max': all_pred_values.max().item(),
                'target_p90': torch.quantile(all_target_values, 0.90).item(),
                'pred_p90': torch.quantile(all_pred_values, 0.90).item(),
                'coef': c_human_risk,
            }
        else:
            self.last_aux_stats = {
                'coef': c_human_risk,
                'target_mean': 0.0,
                'pred_mean': 0.0,
                'valid_slots': 0.0,
                'positive_frac': 0.0,
                'weight_mean': 0.0,
                'target_max': 0.0,
                'pred_max': 0.0,
                'target_p90': 0.0,
                'pred_p90': 0.0,
            }

        return (
            value_loss_epoch / num_updates,
            cost_value_loss_epoch / num_updates,
            penalty,
            action_loss_epoch / num_updates,
            dist_entropy_epoch / num_updates,
            human_risk_loss_epoch / num_updates,
        )
