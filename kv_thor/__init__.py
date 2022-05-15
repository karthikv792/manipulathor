"""lets start with an env that, at the beginning the robot is facing a table and holding a cup (or any pickable
object in ManipulaTHOR), and the task of the robot is to place the cup at a specified position on the table. """
from manipulathor_baselines.armpointnav_baselines.experiments.ithor.armpointnav_ithor_base import \
    ArmPointNaviThorBaseConfig
from kv_thor.kv_thor_config import KVThorConfig
from ithor_arm.ithor_arm_viz import ImageVisualizer, TestMetricLogger
import random
import gym
"""
Initial State:
    - robot is at the table
    - robot is holding a cup (or any pickable object in ManipulaTHOR)
Goal State:
    - robot is at the table
    - robot places the cup at the specified position on the table
"""

class KVThorEnv():
    def __init__(self):
        self.config = KVThorConfig()
        self.scene_id = random.choice(range(len(self.config.TRAIN_SCENES)))
        kwargs = self.get_sampler_args()
        self.sampler = self.config.make_sampler_fn(**kwargs)
        self.task = self.sampler.next_task()

    def get_sampler_args(self):
        res = {}
        res["scenes"] = [self.config.TRAIN_SCENES[self.scene_id]]
        res["max_steps"] = self.config.MAX_STEPS
        res["sensors"] = self.config.SENSORS
        res["action_space"] = gym.spaces.Discrete(
            len(self.config.TASK_SAMPLER._TASK_TYPE.class_action_names())
        )
        res["seed"] = self.scene_id
        res["deterministic_cudnn"] = False
        res["rewards_config"] = self.config.REWARD_CONFIG
        res["scene_period"] = "manual"
        res["sampler_mode"] = "train"
        res["cap_training"] = self.config.CAP_TRAINING
        res["env_args"] = {}
        res["env_args"].update(self.config.ENV_ARGS)
        res["env_args"]["x_display"] = None
        return res

    def reset(self):
        self.sampler.reset()
        self.task = self.sampler.next_task()

    def step(self, action):
        return self.task.step(action)

    def render(self):
        return self.task.render()

    def close(self):
        return self.task.close()

    def action_space(self):
        return self.task.action_space

    def _increment_num_steps_taken(self):
        self.task._increment_num_steps_taken()

    def reached_max_steps(self):
        return self.task.reached_max_steps()

    def reached_terminal_state(self):
        return self.task.reached_terminal_state()

    def is_done(self):
        return self.task.is_done()

    def metrics(self):
        return self.task.metrics()

    def action_names(self):
        return self.task.action_names()

    def cumulative_reward(self):
        return self.task.cumulative_reward()
