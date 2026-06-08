# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of Lagrange."""



from __future__ import annotations

import torch


class Lagrange:
    """
    Shadow-PI Lagrange:
    - shadow_lambda: 内部安全压力状态
    - effective_lambda: 真正作用到 actor 的乘子
    - lagrangian_multiplier: 为兼容旧代码，始终同步为 effective_lambda
    """

    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lambda_lr: float,
        lambda_optimizer: str = 'Adam',
        lagrangian_upper_bound: float | None = None,
        **kwargs,
    ) -> None:
        device = kwargs.get(
            'device',
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.device = torch.device(device)

        self.cost_limit = float(cost_limit)
        self.lambda_lr = float(lambda_lr)  # 保留接口兼容，不再实际使用
        self.lagrangian_upper_bound = (
            float(lagrangian_upper_bound)
            if lagrangian_upper_bound is not None else None
        )

        # -----------------------------
        # Soft gating
        # -----------------------------
        self.soft_gating_lower_bound = float(kwargs.get('soft_gating_lower_bound', 0.22))
        self.soft_gating_upper_bound = float(kwargs.get('soft_gating_upper_bound', 0.50))
        self.gate_start_step = int(kwargs.get('gate_start_step', 15_000_000))
        self.gate_full_step = int(kwargs.get('gate_full_step', 35_000_000))

        # -----------------------------
        # Shadow PI controller
        # -----------------------------
        self.jc_ema_beta = float(kwargs.get('jc_ema_beta', 0.90))
        self.violation_deadband = float(kwargs.get('violation_deadband', 0.05))
        self.lambda_kp = float(kwargs.get('lambda_kp', 0.05))
        self.lambda_ki = float(kwargs.get('lambda_ki', 0.006))
        self.integral_min = float(kwargs.get('integral_min', -50.0))
        self.integral_max = float(kwargs.get('integral_max', 50.0))
        self.lambda_output_tau = float(kwargs.get('lambda_output_tau', 0.92))

        init_value = max(float(lagrangian_multiplier_init), 0.0)

        # 内部状态
        self.shadow_lambda = torch.tensor(init_value, dtype=torch.float32, device=self.device)
        self.effective_lambda = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        # 兼容旧代码：外部还在读取 lagrangian_multiplier
        self.lagrangian_multiplier = self.effective_lambda.detach().clone()

        self.lambda_optimizer = None

        self.last_error = 0.0
        self.integral_error = 0.0
        self.jc_ema = None

        # 调试量
        self.last_success_gate = 0.0
        self.last_progress_gate = 0.0
        self.last_gate_weight = 0.0

    @staticmethod
    def _clamp(x: float, low: float, high: float) -> float:
        return max(low, min(x, high))

    @staticmethod
    def _smoothstep(x: float, x0: float, x1: float) -> float:
        if x1 <= x0:
            return 1.0 if x >= x1 else 0.0
        t = (x - x0) / (x1 - x0)
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _sync_public_multiplier(self) -> None:
        # 兼容旧代码读取 self.lagrangian_multiplier
        self.lagrangian_multiplier = self.effective_lambda.detach().clone()

    def set_cost_limit(self, new_limit: float) -> None:
        self.cost_limit = float(new_limit)

    def compute_lambda_loss(self, mean_ep_cost: float) -> torch.Tensor:
        # 仅保留兼容接口
        return -self.lagrangian_multiplier * (float(mean_ep_cost) - self.cost_limit)

    def update_lagrange_multiplier(self, Jc: float, global_step: int, success_rate: float = 0.0) -> None:
        """
        Shadow-PI + ability-first gate:
        1) shadow_lambda 持续跟踪安全压力
        2) effective_lambda = gate * shadow_lambda，并做平滑
        3) gate 采用：
        gate = progress + (1-progress) * success
        => 早期由 success 主导，后期由 progress 兜底拉满
        """
        Jc = float(Jc)
        global_step = int(global_step)
        success_rate = float(success_rate)

        # -----------------------------
        # 1. success gate + progress gate
        # -----------------------------
        success_gate = self._smoothstep(
            success_rate,
            self.soft_gating_lower_bound,
            self.soft_gating_upper_bound,
        )
        progress_gate = self._smoothstep(
            float(global_step),
            float(self.gate_start_step),
            float(self.gate_full_step),
        )

        # 核心修改：能力优先，时间兜底
        gate_weight = progress_gate + (1.0 - progress_gate) * success_gate

        self.last_success_gate = success_gate
        self.last_progress_gate = progress_gate
        self.last_gate_weight = gate_weight

        # -----------------------------
        # 2. EMA smooth episodic Jc
        # -----------------------------
        if self.jc_ema is None:
            self.jc_ema = Jc
        else:
            self.jc_ema = self.jc_ema_beta * self.jc_ema + (1.0 - self.jc_ema_beta) * Jc

        # -----------------------------
        # 3. Relative violation + deadband
        # -----------------------------
        denom = max(self.cost_limit, 1e-6)
        rel_error = (self.jc_ema - self.cost_limit) / denom

        if abs(rel_error) < self.violation_deadband:
            rel_error = 0.0

        self.last_error = rel_error

        # -----------------------------
        # 4. Update shadow lambda
        # -----------------------------
        current_shadow = float(self.shadow_lambda.item())
        upper_bound = self.lagrangian_upper_bound if self.lagrangian_upper_bound is not None else float('inf')

        is_hitting_upper_bound = current_shadow >= upper_bound - 1e-8
        is_hitting_lower_bound = current_shadow <= 0.0 + 1e-8

        if not ((is_hitting_upper_bound and rel_error > 0.0) or (is_hitting_lower_bound and rel_error < 0.0)):
            effective_integral_gain = 0.2 + 0.8 * gate_weight
            self.integral_error = 0.995 * self.integral_error + effective_integral_gain * rel_error
            self.integral_error = self._clamp(
                self.integral_error,
                self.integral_min,
                self.integral_max,
            )

        target_shadow = self.lambda_kp * rel_error + self.lambda_ki * self.integral_error
        target_shadow = self._clamp(target_shadow, 0.0, upper_bound)

        new_shadow = 0.90 * current_shadow + 0.10 * target_shadow
        new_shadow = self._clamp(new_shadow, 0.0, upper_bound)

        # -----------------------------
        # 5. Gate shadow -> effective lambda
        # -----------------------------
        target_effective = gate_weight * new_shadow
        current_effective = float(self.effective_lambda.item())

        new_effective = self.lambda_output_tau * current_effective + (1.0 - self.lambda_output_tau) * target_effective
        new_effective = self._clamp(new_effective, 0.0, upper_bound)

        self.shadow_lambda = torch.tensor(new_shadow, dtype=torch.float32, device=self.device)
        self.effective_lambda = torch.tensor(new_effective, dtype=torch.float32, device=self.device)
        self._sync_public_multiplier()

    def get_debug_dict(self) -> dict:
        return {
            'cost_limit': float(self.cost_limit),
            'jc_ema': None if self.jc_ema is None else float(self.jc_ema),
            'last_error': float(self.last_error),
            'integral_error': float(self.integral_error),
            'shadow_lambda': float(self.shadow_lambda.item()),
            'effective_lambda': float(self.effective_lambda.item()),
            'success_gate': float(self.last_success_gate),
            'progress_gate': float(self.last_progress_gate),
            'gate_weight': float(self.last_gate_weight),
        }