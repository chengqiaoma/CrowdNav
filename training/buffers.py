import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class RolloutBuffer(object):
    def __init__(
    self,
    num_steps,
    num_processes,
    obs_shape,
    action_space,
    recurrent_hidden_state_size,
    device,
    gamma=0.99,
    tau=0.95,
    cost_gamma=0.99,
    cost_tau=0.95,
    future_hard_K=6,
    future_hard_gamma=0.90,
    future_hard_thresh=0.30,
    future_collision_thresh=0.999,
    robot_radius=0.3,
    human_radius=0.25,
    human_safe_margin=0.10,
    human_base_dim=4,
    use_aci_feature=False,
    human_aci_scale=0.20,
    human_ttc_horizon=1.5,
    human_ttc_weight=0.70,
    future_human_risk_K=None,
    future_human_risk_gamma=None,
    enable_per_human_risk_aux=True,
    aux_use_obstacle_context=False,
    aux_obstacle_safe_dist=0.70,
    aux_squeeze_human_dist=1.25,
    aux_squeeze_risk_coef=0.40,
    aux_squeeze_risk_cap=0.50,
):
        self.obs = {}
        for key in obs_shape.spaces:
            self.obs[key] = torch.zeros(
                num_steps + 1, num_processes, *obs_shape.spaces[key].shape
            ).to(device)

        self.recurrent_hidden_states = {}
        self.recurrent_hidden_states['rnn'] = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size
        ).to(device)

        self.rewards = torch.zeros(num_steps, num_processes, 1).to(device)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1).to(device)

        self.cost_recurrent_hidden_states = {}
        self.cost_recurrent_hidden_states['rnn'] = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size
        ).to(device)

        self.costs = torch.zeros(num_steps, num_processes, 1).to(device)
        self.cost_value_preds = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        self.cost_returns = torch.zeros(num_steps + 1, num_processes, 1).to(device)

        # 当前 transition 是否以人碰撞终止，用于把终止碰撞归因到最近行人槽位。
        self.human_collision_flags = torch.zeros(num_steps, num_processes, 1).to(device)

        self.max_human_slots = int(obs_shape.spaces['spatial_edges'].shape[0])
        self.per_human_valid_masks = torch.zeros(
            num_steps, num_processes, self.max_human_slots
        ).to(device)
        self.per_human_future_risk_labels = torch.zeros(
            num_steps, num_processes, self.max_human_slots
        ).to(device)

        self.action_log_probs = torch.zeros(num_steps, num_processes, 1).to(device)

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_processes, action_shape).to(device)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()

        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(device)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1).to(device)

        self.num_steps = num_steps
        self.num_processes = num_processes
        self.step = 0
        self.gamma = gamma
        self.tau = tau
        self.cost_gamma = cost_gamma
        self.cost_tau = cost_tau
        self.device = device

        # future_hard_* stays in the constructor for older configs, but the
        # active auxiliary loss now uses only per-human future-risk labels.
        self.human_safe_dist_base = float(robot_radius + human_radius + human_safe_margin)
        self.human_base_dim = int(human_base_dim)
        self.use_aci_feature = bool(use_aci_feature)
        self.human_aci_scale = float(human_aci_scale)
        self.human_ttc_horizon = float(human_ttc_horizon)
        self.human_ttc_weight = float(human_ttc_weight)
        self.future_human_risk_K = int(
            future_human_risk_K if future_human_risk_K is not None else future_hard_K
        )
        self.future_human_risk_gamma = float(
            future_human_risk_gamma if future_human_risk_gamma is not None else future_hard_gamma
        )
        self.enable_per_human_risk_aux = bool(enable_per_human_risk_aux)
        self.aux_use_obstacle_context = bool(aux_use_obstacle_context)
        self.aux_obstacle_safe_dist = float(aux_obstacle_safe_dist)
        self.aux_squeeze_human_dist = float(aux_squeeze_human_dist)
        self.aux_squeeze_risk_coef = float(aux_squeeze_risk_coef)
        self.aux_squeeze_risk_cap = float(aux_squeeze_risk_cap)

    def to(self, device):
        self.device = device
        for key in self.obs:
            self.obs[key] = self.obs[key].to(device)
        for key in self.recurrent_hidden_states:
            self.recurrent_hidden_states[key] = self.recurrent_hidden_states[key].to(device)
        for key in self.cost_recurrent_hidden_states:
            self.cost_recurrent_hidden_states[key] = self.cost_recurrent_hidden_states[key].to(device)

        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)

        self.costs = self.costs.to(device)
        self.cost_value_preds = self.cost_value_preds.to(device)
        self.cost_returns = self.cost_returns.to(device)

        self.human_collision_flags = self.human_collision_flags.to(device)
        self.per_human_valid_masks = self.per_human_valid_masks.to(device)
        self.per_human_future_risk_labels = self.per_human_future_risk_labels.to(device)

        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(
    self,
    obs,
    recurrent_hidden_states,
    cost_recurrent_hidden_states,
    actions,
    action_log_probs,
    value_preds,
    cost_value_preds,
    rewards,
    costs,
    masks,
    bad_masks,
    human_collisions=None,
):
        for key in self.obs:
            self.obs[key][self.step + 1].copy_(obs[key])

        for key in self.recurrent_hidden_states:
            self.recurrent_hidden_states[key][self.step + 1].copy_(recurrent_hidden_states[key])

        for key in self.cost_recurrent_hidden_states:
            self.cost_recurrent_hidden_states[key][self.step + 1].copy_(cost_recurrent_hidden_states[key])

        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.cost_value_preds[self.step].copy_(cost_value_preds)
        self.costs[self.step].copy_(costs)

        if human_collisions is None:
            self.human_collision_flags[self.step].zero_()
        else:
            self.human_collision_flags[self.step].copy_(human_collisions)

        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.step = (self.step + 1) % self.num_steps

    @torch.no_grad()
    def compute_per_human_future_risk_labels(self, K=None, gamma=None):
        if K is None:
            K = self.future_human_risk_K
        if gamma is None:
            gamma = self.future_human_risk_gamma

        self.per_human_valid_masks.zero_()
        self.per_human_future_risk_labels.zero_()

        if 'spatial_edges' not in self.obs or 'detected_human_num' not in self.obs:
            return

        K = max(1, min(int(K), self.num_steps))
        gamma = float(gamma)

        spatial_edges = self.obs['spatial_edges'][:-1]
        counts = self.obs['detected_human_num'][:-1, :, 0].round().long()
        counts = counts.clamp(min=0, max=self.max_human_slots)

        if 'true_human_ids' in self.obs:
            human_ids = self.obs['true_human_ids'][:-1].long()
        else:
            human_ids = torch.full(
                (self.num_steps, self.num_processes, self.max_human_slots),
                -1,
                dtype=torch.long,
                device=self.device,
            )

        slot_idx = torch.arange(self.max_human_slots, device=self.device).view(1, 1, -1)
        valid_mask = (slot_idx < counts.unsqueeze(-1)) & (human_ids >= 0)

        distances = torch.linalg.norm(spatial_edges[..., :2], dim=-1)
        safe_dists = torch.full_like(distances, self.human_safe_dist_base)
        if self.use_aci_feature and spatial_edges.size(-1) > self.human_base_dim:
            aci_feat = spatial_edges[..., self.human_base_dim].clamp(0.0, 1.0)
            safe_dists = safe_dists + self.human_aci_scale * aci_feat

        distance_risk = torch.clamp(
            (safe_dists - distances) / safe_dists.clamp(min=1e-6),
            min=0.0,
            max=1.0,
        )
        ttc_risk = torch.zeros_like(distance_risk)
        if spatial_edges.size(-1) >= 4:
            rel_pos = spatial_edges[..., :2]
            rel_vel = spatial_edges[..., 2:4]
            dist_safe = distances.clamp(min=1e-6)
            closing_speed = -torch.sum(rel_pos * rel_vel, dim=-1) / dist_safe
            ttc = (distances - safe_dists) / closing_speed.clamp(min=1e-6)
            ttc_valid = (
                valid_mask
                & (closing_speed > 1e-6)
                & (distances > safe_dists)
                & (ttc >= 0.0)
                & (ttc < self.human_ttc_horizon)
            )
            ttc_risk = torch.clamp(
                (self.human_ttc_horizon - ttc) / max(self.human_ttc_horizon, 1e-6),
                min=0.0,
                max=1.0,
            )
            ttc_risk = ttc_risk * ttc_valid.float()

        step_risk = torch.maximum(distance_risk, self.human_ttc_weight * ttc_risk)
        step_risk = step_risk * valid_mask.float()

        if self.aux_use_obstacle_context and 'point_clouds' in self.obs:
            lidar = self.obs['point_clouds'][:-1].float().view(self.num_steps, self.num_processes, -1)
            min_lidar_dist = lidar.amin(dim=-1)
            obstacle_risk = torch.clamp(
                (self.aux_obstacle_safe_dist - min_lidar_dist)
                / max(self.aux_obstacle_safe_dist, 1e-6),
                min=0.0,
                max=1.0,
            )
            human_near_risk = torch.clamp(
                (self.aux_squeeze_human_dist - distances)
                / max(self.aux_squeeze_human_dist, 1e-6),
                min=0.0,
                max=1.0,
            )
            if spatial_edges.size(-1) >= 4:
                human_near_risk = torch.maximum(human_near_risk, self.human_ttc_weight * ttc_risk)

            squeeze_risk = (
                self.aux_squeeze_risk_coef
                * obstacle_risk.unsqueeze(-1)
                * human_near_risk
            )
            squeeze_risk = torch.clamp(squeeze_risk, min=0.0, max=self.aux_squeeze_risk_cap)
            step_risk = torch.maximum(step_risk, squeeze_risk * valid_mask.float())

        collision_mask = self.human_collision_flags[:, :, 0] > 0.5
        has_valid_human = valid_mask.any(dim=-1)
        terminal_human_hits = collision_mask & has_valid_human
        if terminal_human_hits.any():
            masked_distances = distances.masked_fill(~valid_mask, float('inf'))
            nearest_slots = masked_distances.argmin(dim=-1)
            hit_t, hit_env = terminal_human_hits.nonzero(as_tuple=True)
            step_risk[hit_t, hit_env, nearest_slots[hit_t, hit_env]] = 1.0

        future_labels = torch.zeros_like(step_risk)
        for delta in range(K):
            length = self.num_steps - delta
            if length <= 0:
                break

            if delta == 0:
                alive_envs = torch.ones(
                    length, self.num_processes, dtype=torch.bool, device=self.device
                )
            else:
                alive_envs = torch.ones(
                    length, self.num_processes, dtype=torch.bool, device=self.device
                )
                for mask_offset in range(1, delta + 1):
                    alive_envs = alive_envs & (
                        self.masks[mask_offset:mask_offset + length, :, 0] > 0.5
                    )

            current_ids = human_ids[:length]
            future_ids = human_ids[delta:delta + length]
            current_valid = valid_mask[:length]
            future_valid = valid_mask[delta:delta + length]

            matches = (
                (current_ids.unsqueeze(-1) == future_ids.unsqueeze(-2))
                & current_valid.unsqueeze(-1)
                & future_valid.unsqueeze(-2)
                & alive_envs.unsqueeze(-1).unsqueeze(-1)
            )
            matched_future_risk = torch.where(
                matches,
                step_risk[delta:delta + length].unsqueeze(-2),
                torch.zeros((), dtype=step_risk.dtype, device=self.device),
            ).amax(dim=-1)

            candidate = (gamma ** delta) * matched_future_risk
            future_labels[:length] = torch.maximum(future_labels[:length], candidate)

        self.per_human_valid_masks.copy_(valid_mask.float())
        self.per_human_future_risk_labels.copy_(future_labels)

    def after_update(self):
        for key in self.obs:
            self.obs[key][0].copy_(self.obs[key][-1])
        for key in self.recurrent_hidden_states:
            self.recurrent_hidden_states[key][0].copy_(self.recurrent_hidden_states[key][-1])
        for key in self.cost_recurrent_hidden_states:
            self.cost_recurrent_hidden_states[key][0].copy_(self.cost_recurrent_hidden_states[key][-1])

        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, next_value, next_cost_value, use_gae, gamma, gae_lambda, cost_gamma, cost_tau, use_proper_time_limits=True):
        # -----------------------------
        # Reward returns
        # -----------------------------
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = delta + self.gamma * self.tau * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]
                    ) * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = delta + self.gamma * self.tau * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

        # -----------------------------
        # Cost returns
        # -----------------------------
        if use_proper_time_limits:
            if use_gae:
                self.cost_value_preds[-1] = next_cost_value
                gae = 0
                for step in reversed(range(self.costs.size(0))):
                    delta = self.costs[step] + cost_gamma * self.cost_value_preds[step + 1] * self.masks[step + 1] - self.cost_value_preds[step]
                    gae = delta + cost_gamma * cost_tau * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.cost_returns[step] = gae + self.cost_value_preds[step]
            else:
                self.cost_returns[-1] = next_cost_value
                for step in reversed(range(self.costs.size(0))):
                    self.cost_returns[step] = (
                        self.cost_returns[step + 1] * cost_gamma * self.masks[step + 1] + self.costs[step]
                    ) * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.cost_value_preds[step]
        else:
            if use_gae:
                self.cost_value_preds[-1] = next_cost_value
                gae = 0
                for step in reversed(range(self.costs.size(0))):
                    delta = self.costs[step] + cost_gamma * self.cost_value_preds[step + 1] * self.masks[step + 1] - self.cost_value_preds[step]
                    gae = delta + cost_gamma * cost_tau * self.masks[step + 1] * gae
                    self.cost_returns[step] = gae + self.cost_value_preds[step]
            else:
                self.cost_returns[-1] = next_cost_value
                for step in reversed(range(self.costs.size(0))):
                    self.cost_returns[step] = self.cost_returns[step + 1] * cost_gamma * self.masks[step + 1] + self.costs[step]

        if self.enable_per_human_risk_aux:
            self.compute_per_human_future_risk_labels(
                K=min(self.future_human_risk_K, self.num_steps),
                gamma=self.future_human_risk_gamma,
            )
        else:
            self.per_human_valid_masks.zero_()
            self.per_human_future_risk_labels.zero_()

    def recurrent_generator(self, advantages, cost_advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "num_processes ({}) must be >= num_mini_batch ({})".format(
                num_processes, num_mini_batch
            )
        )

        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes, device=self.device)

        for start_ind in range(0, num_processes, num_envs_per_batch):
            ind = perm[start_ind:start_ind + num_envs_per_batch]

            obs_batch = {}
            recurrent_hidden_states_batch = {}
            cost_recurrent_hidden_states_batch = {}

            # 当前时刻 T 的 observation
            for key in self.obs:
                obs_batch[key] = self.obs[key][:-1, ind].reshape(
                    -1, *self.obs[key].size()[2:]
                )

            # RNN hidden state 取每条序列开头
            for key in self.recurrent_hidden_states:
                recurrent_hidden_states_batch[key] = self.recurrent_hidden_states[key][0, ind]

            for key in self.cost_recurrent_hidden_states:
                cost_recurrent_hidden_states_batch[key] = self.cost_recurrent_hidden_states[key][0, ind]

            actions_batch = self.actions[:, ind].reshape(-1, self.actions.size(-1))
            value_preds_batch = self.value_preds[:-1, ind].reshape(-1, 1)
            cost_value_preds_batch = self.cost_value_preds[:-1, ind].reshape(-1, 1)
            return_batch = self.returns[:-1, ind].reshape(-1, 1)
            cost_return_batch = self.cost_returns[:-1, ind].reshape(-1, 1)
            masks_batch = self.masks[:-1, ind].reshape(-1, 1)
            old_action_log_probs_batch = self.action_log_probs[:, ind].reshape(-1, 1)
            adv_targ = advantages[:, ind].reshape(-1, 1)
            cost_adv_targ = cost_advantages[:, ind].reshape(-1, 1)

            # -----------------------------
            # 当前只保留真实参与 loss 的 per-human 辅助监督字段。
            # -----------------------------
            per_human_future_risk_batch = self.per_human_future_risk_labels[:, ind].reshape(
                -1, self.max_human_slots
            )
            per_human_valid_mask_batch = self.per_human_valid_masks[:, ind].reshape(
                -1, self.max_human_slots
            )

            yield (
                obs_batch,                          # 0
                recurrent_hidden_states_batch,      # 1
                cost_recurrent_hidden_states_batch, # 2
                actions_batch,                      # 3
                value_preds_batch,                  # 4
                cost_value_preds_batch,             # 5
                return_batch,                       # 6
                cost_return_batch,                  # 7
                masks_batch,                        # 8
                old_action_log_probs_batch,         # 9
                adv_targ,                           # 10
                cost_adv_targ,                      # 11
                per_human_future_risk_batch,        # 12
                per_human_valid_mask_batch,         # 13
            )
