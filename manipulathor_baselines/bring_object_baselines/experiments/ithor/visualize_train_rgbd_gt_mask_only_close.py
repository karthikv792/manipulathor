import platform

import gym
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor, TempAllMasksSensor, TempEpisodeNumber, TempObjectCategorySensor
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.ithor_arm_constants import ENV_ARGS, TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.ithor_arm_sensors import (
    InitialAgentArmToObjectSensor,
    InitialObjectToGoalSensor,
    PickedUpObjSensor,
    DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor,
)
from ithor_arm.ithor_arm_viz import MaskImageVisualizer
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.predict_mask_rgbd import PredictMaskNoNoiseRGBQueryObjGTMaskSimpleDiverseBringObject
from manipulathor_baselines.bring_object_baselines.experiments.ithor.rgbd_gt_mask_only_close import NoPickUpRGBDMaskOnlyClose
from manipulathor_baselines.bring_object_baselines.models.rgbd_w_predict_mask_small_bring_object_model import PredictMaskSmallBringObjectWQueryObjRGBDModel


class VisualizeTrainMaskClose(
    NoPickUpRGBDMaskOnlyClose
):
    NUMBER_OF_TEST_PROCESS = 1
    VISUALIZE = True
    # TEST_SCENES = NoPickUpRGBDMaskOnlyClose.TRAIN_SCENES TODO
