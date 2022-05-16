import platform
from ithor_arm.ithor_arm_constants import TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.ithor_arm_constants import ENV_ARGS
from kv_thor.dropoff_obj_task import SimpleArmDropNavSampler
from typing import Dict, Any, List, Optional, Sequence
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import TaskSampler
from ithor_arm.ithor_arm_sensors import (
    RelativeAgentArmToObjectSensor,
    RelativeObjectToGoalSensor,
    PickedUpObjSensor,
)
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
import gym
import torch
class KVThorConfig():
    def __init__(self):
        super(KVThorConfig, self).__init__()
        self.REWARD_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "pickup_success_reward": 10.0,
            "failed_stop_reward": 0.0,
            "shaping_weight": 1.0,  # we are not using this
            "failed_action_penalty": -0.03,
        }
        self.ENV_ARGS = ENV_ARGS
        self.TOTAL_NUMBER_SCENES = 30

        self.TRAIN_SCENES = [
            "FloorPlan{}_physics".format(str(i))
            for i in range(1, self.TOTAL_NUMBER_SCENES + 1)
            if (i % 3 == 1 or i % 3 == 0) and i != 28
        ]  # last scenes are really bad
        print(self.TRAIN_SCENES)
        self.TEST_SCENES = [
            "FloorPlan{}_physics".format(str(i))
            for i in range(1, self.TOTAL_NUMBER_SCENES + 1)
            if i % 3 == 2 and i % 6 == 2
        ]
        self.VALID_SCENES = [
            "FloorPlan{}_physics".format(str(i))
            for i in range(1, self.TOTAL_NUMBER_SCENES + 1)
            if i % 3 == 2 and i % 6 == 5
        ]

        self.ALL_SCENES = self.TRAIN_SCENES + self.TEST_SCENES + self.VALID_SCENES


        self.OBJECT_TYPES = tuple(sorted(TRAIN_OBJECTS))

        self.UNSEEN_OBJECT_TYPES = tuple(sorted(TEST_OBJECTS))

        self.TASK_SAMPLER = SimpleArmDropNavSampler
        self.VISUALIZE = False
        if platform.system() == "Darwin":
            self.VISUALIZE = True

        self.NUM_PROCESSES: Optional[int] = None
        self.TRAIN_GPU_IDS = list(range(torch.cuda.device_count()))
        self.SAMPLER_GPU_IDS = self.TRAIN_GPU_IDS
        self.VALID_GPU_IDS = [torch.cuda.device_count() - 1]
        self.TEST_GPU_IDS = [torch.cuda.device_count() - 1]

        self.TRAIN_DATASET_DIR: Optional[str] = None
        self.VAL_DATASET_DIR: Optional[str] = None

        self.CAP_TRAINING = None

        self.VALID_SAMPLES_IN_SCENE = 1
        self.TEST_SAMPLES_IN_SCENE = 1

        self.NUMBER_OF_TEST_PROCESS = 10

        self.ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None

        self.STEP_SIZE = 0.25
        self.ROTATION_DEGREES = 45.0
        self.VISIBILITY_DISTANCE = 1.0
        self.STOCHASTIC = False

        self.CAMERA_WIDTH = 224*2
        self.CAMERA_HEIGHT = 224*2
        self.SCREEN_SIZE = 224*2
        self.MAX_STEPS = 200

        self.SENSORS = [
            RGBSensorThor(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_resnet_normalization=True,
                uuid="rgb_lowres",
            ),
            RelativeAgentArmToObjectSensor(),
            RelativeObjectToGoalSensor(),
            PickedUpObjSensor(),
        ]

    def make_sampler_fn(self,**kwargs) -> TaskSampler:
        from datetime import datetime
        now = datetime.now()
        exp_name_w_time = "KVTHORCONFIG_" + now.strftime("%m_%d_%Y_%H_%M_%S_%f")
        if self.VISUALIZE:
            visualizers = [
                ImageVisualizer(exp_name=exp_name_w_time),
                TestMetricLogger(exp_name=exp_name_w_time),
            ]

            kwargs["visualizers"] = visualizers
        kwargs["objects"] = self.OBJECT_TYPES
        kwargs["exp_name"] = exp_name_w_time
        return self.TASK_SAMPLER(**kwargs)

