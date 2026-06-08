import gym
import numpy as np
import copy
from collections import deque, defaultdict


class SonicSafetyWrapper(gym.Wrapper):
    """
    Unified safety wrapper:
    1. 维护每个行人的速度历史
    2. 根据短窗口平滑波动率更新 human.aci_radius
    3. 将 spatial_edges 扩展为: 原始特征 + aci_radius + volatility(5维)
    4. 直接读取环境真值 danger_indicator / instant_cost
    """
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.robot_radius = config.robot.radius

        # ===== variant 开关 =====
        self.use_volatility_aci = getattr(config.training, 'use_volatility_aci', True)
        self.use_vs_obs = getattr(config.training, 'use_vs_obs', True)

        self.sample_num = 5
        self.aci_window = 3
        self.aci_alpha = 0.6

        self.observation_space = copy.deepcopy(env.observation_space)
        orig_space = self.observation_space.spaces['spatial_edges']
        orig_shape = orig_space.shape

        self.base_dim = orig_shape[1]
        if self.use_vs_obs:
            self.new_feature_dim = self.base_dim + 1 + self.sample_num
        else:
            self.new_feature_dim = self.base_dim

        new_shape = (orig_shape[0], self.new_feature_dim)

        self.observation_space.spaces['spatial_edges'] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=new_shape, dtype=np.float32
        )

        self.max_human_num = new_shape[0]
        self.target_spatial_dim = getattr(config.training, 'eval_target_spatial_dim', None)
        if self.target_spatial_dim is not None:
            self.target_spatial_dim = int(self.target_spatial_dim)
            self.observation_space.spaces['spatial_edges'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(orig_shape[0], self.target_spatial_dim),
                dtype=np.float32,
            )

        self.target_robot_node_dim = getattr(config.training, 'eval_target_robot_node_dim', None)
        if self.target_robot_node_dim is not None and 'robot_node' in self.observation_space.spaces:
            self.target_robot_node_dim = int(self.target_robot_node_dim)
            robot_shape = self.observation_space.spaces['robot_node'].shape
            self.observation_space.spaces['robot_node'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(robot_shape[0], self.target_robot_node_dim),
                dtype=np.float32,
            )

        if 'obstacle_vertices' not in self.observation_space.spaces:
            max_obs_num = getattr(self.config.sim, 'max_obs_num', 15)
            self.observation_space.spaces['obstacle_vertices'] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(max_obs_num, 8), dtype=np.float32
            )

        if 'obstacle_num' not in self.observation_space.spaces:
            self.observation_space.spaces['obstacle_num'] = gym.spaces.Box(
                low=0, high=np.inf, shape=(1,), dtype=np.int32
            )

        if 'danger_indicator' not in self.observation_space.spaces:
            self.observation_space.spaces['danger_indicator'] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )

        if 'anisotropic_risk_score' not in self.observation_space.spaces:
            self.observation_space.spaces['anisotropic_risk_score'] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )

        if 'true_human_ids' not in self.observation_space.spaces:
            self.observation_space.spaces['true_human_ids'] = gym.spaces.Box(
                low=-1.0, high=np.inf, shape=(self.max_human_num,), dtype=np.float32
            )

        self.vel_history = defaultdict(lambda: deque(maxlen=self.sample_num + 1))
        self.vol_history = defaultdict(lambda: deque(maxlen=self.sample_num))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.vel_history.clear()
        self.vol_history.clear()
        self._update_history_and_aci()
        return self._augment_observation(obs)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)

        self._update_history_and_aci()
        aug_obs = self._augment_observation(next_obs, info=info)

        raw_env = self.env.unwrapped
        info = dict(info)
        info['danger_indicator'] = float(
            getattr(raw_env, 'danger_indicator', info.get('danger_indicator', 0.0))
        )
        info['instant_cost'] = float(
            getattr(raw_env, 'instant_cost', info.get('instant_cost', 0.0))
        )
        info['anisotropic_risk_score'] = info['instant_cost']

        return aug_obs, reward, done, info

    def _compute_smoothed_volatility(self, vol_seq):
        if len(vol_seq) == 0:
            return 0.0

        diffs = np.asarray(vol_seq, dtype=np.float32)
        k = min(len(diffs), self.aci_window)
        return float(np.mean(diffs[-k:]))

    def _update_history_and_aci(self):
        try:
            raw_env = self.env.unwrapped
            humans = raw_env.humans

            for human in humans:
                hid = getattr(human, 'id', None)
                if hid is None:
                    continue

                cur_vel = np.array([human.vx, human.vy], dtype=np.float32)
                vel_seq = self.vel_history[hid]
                if len(vel_seq) > 0:
                    self.vol_history[hid].append(float(np.linalg.norm(cur_vel - vel_seq[-1])))
                vel_seq.append(cur_vel)

                if not hasattr(human, 'aci_radius'):
                    human.aci_radius = 0.05

                if self.use_volatility_aci:
                    smoothed_vol = self._compute_smoothed_volatility(self.vol_history[hid])
                    human.aci_radius = float(
                        np.clip(0.05 + self.aci_alpha * smoothed_vol, 0.05, 0.25)
                    )
                else:
                    # w/o Volatility-Aware ACI
                    human.aci_radius = 0.0

        except Exception:
            pass

    def _augment_observation(self, obs, info=None):
        obs = dict(obs)
        raw_env = self.env.unwrapped

        spatial_edges = np.asarray(obs['spatial_edges'], dtype=np.float32)
        true_human_ids = obs.get('true_human_ids', None)

        if not self.use_vs_obs:
            obs['spatial_edges'] = spatial_edges
        else:
            aug_spatial = np.zeros((spatial_edges.shape[0], self.new_feature_dim), dtype=np.float32)
            aug_spatial[:, :self.base_dim] = spatial_edges
            human_by_id = {
                getattr(h, 'id', None): h
                for h in raw_env.humans
                if getattr(h, 'id', None) is not None
            }

            for i in range(spatial_edges.shape[0]):
                hid = None
                if true_human_ids is not None:
                    hid = int(true_human_ids[i])

                aci_feat = 0.0
                volatility = np.zeros(self.sample_num, dtype=np.float32)

                if hid is not None and hid >= 0:
                    try:
                        human_obj = human_by_id.get(hid)
                        if human_obj is not None:
                            aci = float(getattr(human_obj, 'aci_radius', 0.05))
                            aci_clipped = np.clip(aci, 0.05, 0.25)
                            aci_feat = (aci_clipped - 0.05) / 0.20
                    except Exception:
                        aci_feat = 0.0

                    if hid in self.vol_history and len(self.vol_history[hid]) > 0:
                        diffs = np.asarray(self.vol_history[hid], dtype=np.float32)
                        take = min(len(diffs), self.sample_num)
                        volatility[:take] = diffs[-take:]

                aug_spatial[i, self.base_dim] = aci_feat
                aug_spatial[i, self.base_dim + 1:] = volatility

            obs['spatial_edges'] = aug_spatial

        if self.target_spatial_dim is not None:
            spatial_edges = np.asarray(obs['spatial_edges'], dtype=np.float32)
            if spatial_edges.shape[1] > self.target_spatial_dim:
                obs['spatial_edges'] = spatial_edges[:, :self.target_spatial_dim]
            elif spatial_edges.shape[1] < self.target_spatial_dim:
                pad = np.zeros(
                    (spatial_edges.shape[0], self.target_spatial_dim - spatial_edges.shape[1]),
                    dtype=np.float32,
                )
                obs['spatial_edges'] = np.concatenate([spatial_edges, pad], axis=1)

        if self.target_robot_node_dim is not None and 'robot_node' in obs:
            robot_node = np.asarray(obs['robot_node'], dtype=np.float32)
            if robot_node.shape[-1] > self.target_robot_node_dim:
                obs['robot_node'] = robot_node[..., :self.target_robot_node_dim]
            elif robot_node.shape[-1] < self.target_robot_node_dim:
                pad_shape = robot_node.shape[:-1] + (self.target_robot_node_dim - robot_node.shape[-1],)
                obs['robot_node'] = np.concatenate(
                    [robot_node, np.zeros(pad_shape, dtype=np.float32)],
                    axis=-1,
                )

        if info is not None:
            danger_val = float(info.get('danger_indicator', 0.0))
            risk_val = float(
                info.get('anisotropic_risk_score', info.get('instant_cost', 0.0))
            )
        else:
            danger_val = float(getattr(raw_env, 'danger_indicator', 0.0))
            risk_val = float(getattr(raw_env, 'instant_cost', 0.0))

        obs['danger_indicator'] = np.array([danger_val], dtype=np.float32)
        obs['anisotropic_risk_score'] = np.array([risk_val], dtype=np.float32)

        if 'obstacle_vertices' not in obs:
            max_obs_num = getattr(self.config.sim, 'max_obs_num', 15)
            obs['obstacle_vertices'] = np.zeros((max_obs_num, 8), dtype=np.float32)

        if 'obstacle_num' not in obs:
            obs['obstacle_num'] = np.array([0], dtype=np.int32)

        if 'true_human_ids' not in obs:
            if hasattr(raw_env, 'sorted_human_ids'):
                obs['true_human_ids'] = np.asarray(raw_env.sorted_human_ids, dtype=np.float32)
            else:
                obs['true_human_ids'] = np.full(self.max_human_num, -1, dtype=np.float32)

        return obs
