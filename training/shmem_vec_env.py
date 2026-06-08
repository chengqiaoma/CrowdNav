"""
An interface for asynchronous vectorized environments.
"""

import multiprocessing as mp
import numpy as np
from baselines.common.vec_env.vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars
import ctypes
import os
import sys
import traceback
from baselines import logger

from baselines.common.vec_env.util import dict_to_obs, obs_space_info, obs_to_dict

_NP_TO_CT = {np.float32: ctypes.c_float,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             bool: ctypes.c_bool,
             np.bool_: ctypes.c_bool}


class ShmemVecEnv(VecEnv):
    """
    Optimized version of SubprocVecEnv that uses shared variables to communicate observations.
    """

    def __init__(self, env_fns, spaces=None, context='spawn'):
        """
        If you don't specify observation_space, we'll have to create a dummy
        environment to get it.
        """
        ctx = mp.get_context(context)
        if spaces:
            observation_space, action_space = spaces
        else:
            logger.log('Creating dummy env object to get spaces')
            with logger.scoped_configure(format_strs=[]):
                dummy = env_fns[0]()
                observation_space, action_space = dummy.observation_space, dummy.action_space
                dummy.close()
                del dummy
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(observation_space)
        self.obs_bufs = [
            {k: ctx.Array(_NP_TO_CT[self.obs_dtypes[k].type], int(np.prod(self.obs_shapes[k]))) for k in self.obs_keys}
            for _ in env_fns]
        self.parent_pipes = []
        self.procs = []
        with clear_mpi_env_vars():
            for env_fn, obs_buf in zip(env_fns, self.obs_bufs):
                wrapped_fn = CloudpickleWrapper(env_fn)
                parent_pipe, child_pipe = ctx.Pipe()
                proc = ctx.Process(target=_subproc_worker,
                            args=(child_pipe, parent_pipe, wrapped_fn, obs_buf, self.obs_shapes, self.obs_dtypes, self.obs_keys))
                proc.daemon = True
                self.procs.append(proc)
                self.parent_pipes.append(parent_pipe)
                proc.start()
                child_pipe.close()
        self.waiting_step = False
        self.viewer = None

    def reset(self):
        if self.waiting_step:
            logger.warn('Called reset() while waiting for the step to complete')
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        return self._decode_obses([pipe.recv() for pipe in self.parent_pipes])

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        for pipe, act in zip(self.parent_pipes, actions):
            pipe.send(('step', act))
        self.waiting_step = True

    def step_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting_step = False
        obs, rews, dones, infos = zip(*outs)
        return self._decode_obses(obs), np.array(rews), np.array(dones), infos

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    def get_images(self, mode='human'):
        for pipe in self.parent_pipes:
            pipe.send(('render', None))
        return [pipe.recv() for pipe in self.parent_pipes]

    def _decode_obses(self, obs):
        result = {}
        for k in self.obs_keys:

            bufs = [b[k] for b in self.obs_bufs]
            o = [np.frombuffer(b.get_obj(), dtype=self.obs_dtypes[k]).reshape(self.obs_shapes[k]) for b in bufs]
            result[k] = np.array(o)
        return dict_to_obs(result)

    def env_method(self, method_name, *method_args, **method_kwargs):
        """调用每个子环境的方法"""
        for pipe in self.parent_pipes:
            pipe.send(('env_method', (method_name, method_args, method_kwargs)))
        return [pipe.recv() for pipe in self.parent_pipes]

    def get_attr(self, attr_name):
        """获取所有子环境的属性"""
        for pipe in self.parent_pipes:
            pipe.send(('get_attr', attr_name))
        return [pipe.recv() for pipe in self.parent_pipes]

    def set_attr(self, attr_name, value):
        """设置所有子环境的属性"""
        for pipe in self.parent_pipes:
            pipe.send(('set_attr', (attr_name, value)))
        return [pipe.recv() for pipe in self.parent_pipes]


def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_bufs, obs_shapes, obs_dtypes, keys):
    """
    Control a single environment instance using IPC and
    shared memory.
    """
    def _write_obs(maybe_dict_obs):
        flatdict = obs_to_dict(maybe_dict_obs)
        for k in keys:
            dst = obs_bufs[k].get_obj()
            dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(obs_shapes[k])  # pylint: disable=W0212
            np.copyto(dst_np, flatdict[k])

    env = None
    cmd = None
    parent_pipe.close()
    try:
        env = env_fn_wrapper.x()
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                pipe.send(_write_obs(env.reset()))
            elif cmd == 'step':
                obs, reward, done, info = env.step(data)

                if done:
                    # 保留终局时刻的 observation 和 info
                    terminal_observation = obs
                    terminal_info = info

                    # 自动 reset，返回下一局初始 obs 供训练继续使用
                    obs = env.reset()

                    # 复制一份 info，避免原地修改带来副作用
                    if info is None:
                        info = {}
                    else:
                        info = dict(info)

                    info['terminal_observation'] = terminal_observation
                    info['terminal_info'] = terminal_info

                pipe.send((_write_obs(obs), reward, done, info))
            elif cmd == 'render':
                pipe.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                pipe.send(None)
                break
            
            # === [新增代码 START] ===
            elif cmd == 'env_method':
                method_name, method_args, method_kwargs = data
                target_attr = getattr(env, method_name)
                result = target_attr(*method_args, **method_kwargs)
                pipe.send(result)
            elif cmd == 'get_attr':
                attr_name = data
                if hasattr(env, attr_name):
                    pipe.send(getattr(env, attr_name))
                else:
                    pipe.send(None)
            elif cmd == 'set_attr':
                attr_name, attr_value = data
                setattr(env, attr_name, attr_value)
                pipe.send(None)
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    except KeyboardInterrupt:
        print('ShmemVecEnv worker: got KeyboardInterrupt', flush=True)
    except Exception:
        print(
            f"ShmemVecEnv worker {mp.current_process().name} "
            f"(pid={os.getpid()}) crashed while handling cmd={cmd!r}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            env.close()
