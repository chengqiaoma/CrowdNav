import numpy as np
import torch

from crowd_sim.envs.utils.info import *
from training.networks import utils


def extract_robot_xy_np(observation):
    """Return robot (px, py) from vectorized or unvectorized robot_node observations."""
    robot_node = observation['robot_node']
    if torch.is_tensor(robot_node):
        robot_node = robot_node.detach().cpu().numpy()

    robot_node = np.asarray(robot_node)
    if robot_node.ndim >= 3:
        return robot_node[0, 0, 0:2]
    if robot_node.ndim == 2:
        return robot_node[0, 0:2]
    if robot_node.ndim == 1:
        return robot_node[0:2]

    raise ValueError(f"Unexpected robot_node shape: {robot_node.shape}")


def estimate_spl_np(observation, base_env=None):
    try:
        robot = base_env.robot
        return float(np.linalg.norm([robot.px - robot.gx, robot.py - robot.gy]))
    except Exception:
        pass

    robot_node = observation['robot_node']
    if torch.is_tensor(robot_node):
        robot_node = robot_node.detach().cpu().numpy()

    robot_node = np.asarray(robot_node)
    if robot_node.ndim >= 3:
        robot_state = robot_node[0, 0]
    elif robot_node.ndim == 2:
        robot_state = robot_node[0]
    elif robot_node.ndim == 1:
        robot_state = robot_node
    else:
        return 1.0

    if robot_state.shape[0] >= 4:
        return float(np.linalg.norm(robot_state[0:2] - robot_state[2:4]))
    if robot_state.shape[0] >= 2:
        return float(np.linalg.norm(robot_state[0:2]))
    return 1.0


def evaluate(actor_critic, eval_envs, num_processes, device, config, logging, test_args):
    """Evaluate the policy model (actor_critic) in multiple testing episodes.

        Parameters:
        actor_critic : torch.nn.Module
            The policy model to evaluate.
        eval_envs : VecEnv
            The vectorized environments for evaluation.
        num_processes : int
            Number of parallel environments to run.
        device : torch.device
            Device for running evaluation (CPU or CUDA).
        config : Config
            Configuration object with environment and training settings.
        logging : logging.Logger
            Logger for evaluation information.
        test_args : argparse.Namespace
            Additional testing arguments like visualization options.
        """

    test_size = config.env.test_size

    eval_episode_rewards = []

    # initialize the RNN hidden states
    eval_recurrent_hidden_states = {}
    if config.robot.policy in ['srnn', 'dsrnn_obs_pc', 'dsrnn_obs_vertex']:
        node_num = 1
        edge_num = actor_critic.base.human_num + 1 + actor_critic.base.obs_num
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num,
                                                                     config.SRNN.human_node_rnn_size,
                                                                     device=device)

        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                           config.SRNN.human_node_rnn_size,
                                                                           device=device)

    else:
        eval_recurrent_hidden_states['rnn'] = torch.zeros(num_processes, 1, config.SRNN.human_node_rnn_size,
                                                          device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    # initialize testing metrics
    success_times = []
    collision_times = []
    timeout_times = []
    path_lengths = []
    success_path_lengths = []
    path_length_ratios = []
    success_path_length_ratios = []

    success = 0
    collision = 0
    collision_human = 0
    collision_obs = 0
    collision_wall = 0

    timeout = 0
    too_close_ratios = []
    min_dist = []
    cumulative_rewards = []

    collision_cases = []
    collision_human_cases = []
    collision_obs_cases = []
    collision_wall_cases = []

    timeout_cases = []
    unknown = 0
    unknown_cases = []
    gamma = 0.99
    baseEnv = eval_envs.venv.envs[0].env

    t = 0

    obs = eval_envs.reset()

    # the main testing loop
    for k in range(test_size):
        t += 1
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0.0

        path = 0.0
        too_close = 0

        last_pos = extract_robot_xy_np(obs)  # robot px, py
        spl = estimate_spl_np(obs, baseEnv)

        while not done:
            stepCounter = stepCounter + 1
            # given observation, forward the robot policy to get action
            if not test_args.dwa:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)
            else: # for DWA
                u, predicted_trajectory, curr_state, action = actor_critic.predict(eval_envs.venv.envs[0].env)

            if test_args.visualize:
                eval_envs.render()

            # step the environment to get reward and next obs
            obs, rew, done, infos = eval_envs.step(action)

            step_info = infos[0].get('terminal_info', infos[0])

            if done[0] and 'terminal_observation' in infos[0]:
                curr_pos = extract_robot_xy_np(infos[0]['terminal_observation'])
            else:
                curr_pos = extract_robot_xy_np(obs)
            path = path + np.linalg.norm(curr_pos - last_pos)

            last_pos = curr_pos

            rewards.append(rew)

            instant_cost = float(step_info.get('instant_cost', step_info.get('cost', 0.0)))
            danger_indicator = float(step_info.get('danger_indicator', 0.0))
            event_obj_step = step_info.get('info', None)

            if instant_cost > 0.0 or danger_indicator > 0.0:
                too_close = too_close + 1
                if isinstance(event_obj_step, Danger):
                    min_dist.append(event_obj_step.min_dist)

            episode_rew += float(rew[0].item() if hasattr(rew[0], 'item') else rew[0])

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        current_nav_time = stepCounter * config.env.time_step
        path_lengths.append(path)
        plr = path / spl if spl > 0 else 1.0
        path_length_ratios.append(plr)
        too_close_ratios.append(too_close / stepCounter * 100 if stepCounter > 0 else 0.0)

        terminal_info = infos[0].get('terminal_info', infos[0])
        info_obj = terminal_info.get('info', None)

        if isinstance(info_obj, ReachGoal):
            success += 1
            success_times.append(current_nav_time)
            success_path_lengths.append(path)
            success_path_length_ratios.append(plr)
            logging.info(
                f"Episode {k+1}: Success   | Rew: {episode_rew:>6.2f} | "
                f"Steps: {stepCounter:>3} | Time: {current_nav_time:>5.2f}s | "
                f"SPL: {spl:>5.2f}m | Path Len: {path:>5.2f}m | PLR: {plr:>5.3f}"
            )
        elif isinstance(info_obj, CollisionHuman):
            collision += 1
            collision_cases.append(k)
            collision_times.append(current_nav_time)
            collision_human += 1
            collision_human_cases.append(k)
            logging.info(
                f"Episode {k+1}: CollisionHuman | Rew: {episode_rew:>6.2f} | "
                f"Steps: {stepCounter:>3} | Time: {current_nav_time:>5.2f}s | "
                f"SPL: {spl:>5.2f}m | Path Len: {path:>5.2f}m"
            )
        elif isinstance(info_obj, CollisionObs):
            collision += 1
            collision_cases.append(k)
            collision_times.append(current_nav_time)
            collision_obs += 1
            collision_obs_cases.append(k)
            logging.info(
                f"Episode {k+1}: CollisionObs | Rew: {episode_rew:>6.2f} | "
                f"Steps: {stepCounter:>3} | Time: {current_nav_time:>5.2f}s | "
                f"SPL: {spl:>5.2f}m | Path Len: {path:>5.2f}m"
            )
        elif isinstance(info_obj, CollisionWall):
            collision += 1
            collision_cases.append(k)
            collision_times.append(current_nav_time)
            collision_wall += 1
            collision_wall_cases.append(k)
            logging.info(
                f"Episode {k+1}: CollisionWall | Rew: {episode_rew:>6.2f} | "
                f"Steps: {stepCounter:>3} | Time: {current_nav_time:>5.2f}s | "
                f"SPL: {spl:>5.2f}m | Path Len: {path:>5.2f}m"
            )
        elif isinstance(info_obj, Timeout):
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(baseEnv.time_limit)
            logging.info(
                f"Episode {k+1}: Timeout   | Rew: {episode_rew:>6.2f} | "
                f"Steps: {stepCounter:>3} | Time: {current_nav_time:>5.2f}s | "
                f"SPL: {spl:>5.2f}m | Path Len: {path:>5.2f}m"
            )
        else:
            unknown += 1
            unknown_cases.append(k)
            logging.warning(
                f"Episode {k+1}: Unknown terminal info={type(info_obj).__name__} | "
                f"Rew: {episode_rew:>6.2f} | Steps: {stepCounter:>3} | "
                f"Time: {current_nav_time:>5.2f}s | SPL: {spl:>5.2f}m | "
                f"Path Len: {path:>5.2f}m"
            )

        cumulative_rewards.append(sum([pow(gamma, t * baseEnv.robot.time_step * baseEnv.robot.v_pref)
                                       * reward for t, reward in enumerate(rewards)]))

    # after all testing episodes are done,
    # calculate and log results
    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size

    collision_human_rate = collision_human / test_size
    collision_obs_rate = collision_obs / test_size
    collision_wall_rate = collision_wall / test_size
    unknown_rate = unknown / test_size
    assert success + collision + timeout + unknown == test_size
    avg_path_length_all = np.mean(path_lengths) if path_lengths else 0.0
    avg_path_length_success = np.mean(success_path_lengths) if success_path_lengths else 0.0
    avg_plr_all = np.mean(path_length_ratios) if path_length_ratios else 0.0
    avg_plr_success = np.mean(success_path_length_ratios) if success_path_length_ratios else 0.0
    avg_intrusion_ratio = np.mean(too_close_ratios) if too_close_ratios else 0.0
    avg_min_dist = np.average(min_dist) if min_dist else 0.0

    model_label = getattr(config.training, 'model_name', config.robot.policy)
    protocol_label = getattr(config.training, 'baseline_protocol', 'n/a')

    print("\n" + "=" * 40)
    print(f"TEST RESULTS (Model: {model_label}, Protocol: {protocol_label})")
    print("=" * 40)
    print(f"Total Episodes:  {test_size}")
    print(f"Success Rate:    {success_rate:.2%}")
    print(f"Collision Rate:  {collision_rate:.2%}")
    print(f"Timeout Rate:    {timeout_rate:.2%}")
    print(f"Unknown Rate:    {unknown_rate:.2%}")
    print(f"Avg Path Length: {avg_path_length_all:.2f} m (All Episodes)")
    print(f"Avg Path Length: {avg_path_length_success:.2f} m (Success Only)")
    print(f"Avg PLR:         {avg_plr_all:.3f} (All Episodes)")
    print(f"Avg PLR:         {avg_plr_success:.3f} (Success Only)")
    print(f"Avg ITR:         {avg_intrusion_ratio:.2f}%")
    print("=" * 40)

    logging.info("\n" + "=" * 40)
    logging.info(f"TEST RESULTS (Model: {model_label}, Protocol: {protocol_label})")
    logging.info("=" * 40)
    logging.info(f"Total Episodes:  {test_size}")
    logging.info(f"Success Rate:    {success_rate:.2%}")
    logging.info(f"Collision Rate:  {collision_rate:.2%}")
    logging.info(f"Timeout Rate:    {timeout_rate:.2%}")
    logging.info(f"Unknown Rate:    {unknown_rate:.2%}")
    if success_times:
        logging.info(f"Avg Nav Time:    {np.mean(success_times):.2f} s")
    else:
        logging.info("Avg Nav Time:    N/A")
    logging.info(
        'collision rate with humans: {:.2f}, with obstacles: {:.2f}, with walls: {:.2f}'.
        format(collision_human_rate, collision_obs_rate, collision_wall_rate))
    logging.info(
        'average intrusion ratio: {:.2f} and average minimal distance during intrusions: {:.2f}'.
        format(avg_intrusion_ratio, avg_min_dist))
    logging.info(f"Avg Path Length: {avg_path_length_all:.2f} m (All Episodes)")
    logging.info(f"Avg Path Length: {avg_path_length_success:.2f} m (Success Only)")
    logging.info(f"Avg Path Length Ratio (PLR): {avg_plr_all:.3f} (All Episodes)")
    logging.info(f"Avg Path Length Ratio (PLR): {avg_plr_success:.3f} (Success Only)")
    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Collision with Human cases: ' + ' '.join([str(x) for x in collision_human_cases]))
    logging.info('Collision with Obstacle cases: ' + ' '.join([str(x) for x in collision_obs_cases]))
    logging.info('Collision with Wall cases: ' + ' '.join([str(x) for x in collision_wall_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
    logging.info('Unknown cases: ' + ' '.join([str(x) for x in unknown_cases]))
    logging.info("=" * 40)

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
