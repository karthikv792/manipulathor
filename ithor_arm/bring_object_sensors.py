"""Utility classes and functions for sensory inputs used by the models."""
import glob
import os
import random
from typing import Any, Union, Optional

import gym
import numpy as np
import torch
from PIL import Image

# from allenact.base_abstractions.sensor import DepthSensor, Sensor, RGBSensor
from allenact.embodiedai.sensors.vision_sensors import DepthSensor, Sensor, RGBSensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor

from ithor_arm.arm_calculation_utils import (
    convert_world_to_agent_coordinate,
    convert_state_to_tensor,
    diff_position,
)
from ithor_arm.ithor_arm_constants import DONT_USE_ALL_POSSIBLE_OBJECTS_EVER
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from manipulathor_baselines.bring_object_baselines.models.detection_model import ConditionalDetectionModel
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.thor_category_names import thor_possible_objects
import torchvision.transforms as transforms

class NoGripperRGBSensorThor(RGBSensorThor):
    def frame_from_env(
            self, env: IThorEnvironment, task: Task[IThorEnvironment]
    ) -> np.ndarray:  # type:ignore
        env.controller.step('ToggleMagnetVisibility')
        frame = env.current_frame.copy()
        env.controller.step('ToggleMagnetVisibility')
        return frame

class CategorySampleSensor(Sensor):
    def __init__(self, type: str, uuid: str = "category_object", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        uuid = '{}_{}'.format(uuid, type)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        if self.type == 'source':
            info_to_search = 'source_object_query'
        elif self.type == 'destination':
            info_to_search = 'goal_object_query'
        else:
            raise Exception('Not implemented', self.type)
        image = task.task_info[info_to_search]
        return image


class NoisyObjectMask(Sensor):
    def __init__(self, type: str,noise,  uuid: str = "object_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        uuid = '{}_{}'.format(uuid, type)
        self.noise = noise
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        if self.type == 'source':
            info_to_search = 'source_object_id'
        elif self.type == 'destination':
            info_to_search = 'goal_object_id'
        else:
            raise Exception('Not implemented', self.type)

        target_object_id = task.task_info[info_to_search]
        all_visible_masks = env.controller.last_event.instance_masks
        if target_object_id in all_visible_masks:
            mask_frame = all_visible_masks[target_object_id]
        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        result = (np.expand_dims(mask_frame.astype(np.float),axis=-1))
        if len(env.controller.last_event.instance_masks) == 0:
            fake_mask = np.zeros(env.controller.last_event.frame[:,:,0].shape)
        else:
            fake_mask = random.choice([v for v in env.controller.last_event.instance_masks.values()])
        fake_mask = (np.expand_dims(fake_mask.astype(np.float),axis=-1))

        return add_mask_noise(result, fake_mask, noise=self.noise)


def add_mask_noise(result, fake_mask, noise):
    TURN_OFF_RATE = noise
    REMOVE_RATE = noise
    REPLACE_WITH_FAKE = noise

    random_prob = random.random()
    if random_prob < REMOVE_RATE:
        result[:] = 0.
    elif random_prob < REMOVE_RATE + REPLACE_WITH_FAKE:
        result = fake_mask
    else:
        w, h, d = result.shape
        mask = np.random.rand(w, h, d)
        mask = mask < TURN_OFF_RATE
        mask = mask & (result == 1)
        result[mask] = 0

    return result



class TempObjectCategorySensor(Sensor):
    def __init__(self, type: str, uuid: str = "temp_category_code", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        uuid = '{}_{}'.format(uuid, type)
        self.type = type
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        if self.type == 'source':
            info_to_search = 'source_object_id'
        elif self.type == 'destination':
            info_to_search = 'goal_object_id'
        else:
            raise Exception('Not implemented', self.type)
        object_type = task.task_info[info_to_search].split('|')[0]
        object_type_categ_ind = DONT_USE_ALL_POSSIBLE_OBJECTS_EVER.index(object_type)
        return torch.tensor(object_type_categ_ind)
class TempEpisodeNumber(Sensor):
    def __init__(self, uuid: str = "temp_episode_number", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return torch.tensor(task.task_info['episode_number'])

class TempAllMasksSensor(Sensor):
    def __init__(self, uuid: str = "all_masks_sensor", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        result = torch.zeros((224,224))
        result[:] = -1
        for (obj, mask) in env.controller.last_event.instance_masks.items():
            object_type = obj.split('|')[0]
            if object_type not in DONT_USE_ALL_POSSIBLE_OBJECTS_EVER:
                continue
            object_type_categ_ind = DONT_USE_ALL_POSSIBLE_OBJECTS_EVER.index(object_type)
            result[mask] = object_type_categ_ind
        return result


class DestinationObjectSensor(Sensor):
    def __init__(self, uuid: str = "destination_object", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return task.task_info['goal_object_id'].split('|')[0]
class TargetObjectMask(Sensor):
    def __init__(self, uuid: str = "target_object_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['source_object_id']
        all_visible_masks = env.controller.last_event.instance_masks
        if target_object_id in all_visible_masks:
            mask_frame = all_visible_masks[target_object_id]
        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        return (np.expand_dims(mask_frame.astype(np.float),axis=-1))



class TargetObjectBBox(Sensor):
    def __init__(self, uuid: str = "target_object_mask", **kwargs: Any): # maybe we should change this name but since i wanted to use the same model
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['source_object_id']
        all_visible_masks = env.controller.last_event.instance_detections2D
        box_as_mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)
        if target_object_id in all_visible_masks:
            x1, y1, x2, y2 = all_visible_masks[target_object_id]
            box_as_mask_frame[y1:y2, x1:x2] = 1.
        return (np.expand_dims(box_as_mask_frame.astype(np.float),axis=-1))



class TargetObjectType(Sensor):
    def __init__(self, uuid: str = "target_object_type", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['source_object_id']
        target_object_type = target_object_id.split('|')[0]
        return thor_possible_objects.index(target_object_type)

class RawRGBSensorThor(Sensor):
    def __init__(self, uuid: str = "raw_rgb_lowres", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return env.current_frame.copy()



class TargetLocationMask(Sensor): #
    def __init__(self, uuid: str = "target_location_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['goal_object_id']
        all_visible_masks = env.controller.last_event.instance_masks
        if target_object_id in all_visible_masks:
            mask_frame = all_visible_masks[target_object_id]
        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        return (np.expand_dims(mask_frame.astype(np.float),axis=-1))

class TargetLocationType(Sensor): #
    def __init__(self, uuid: str = "target_location_type", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['goal_object_id']
        target_location_type = target_object_id.split('|')[0]
        return thor_possible_objects.index(target_location_type)


class TargetLocationBBox(Sensor): #
    def __init__(self, uuid: str = "target_location_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['goal_object_id']

        all_visible_masks = env.controller.last_event.instance_detections2D
        box_as_mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)
        if target_object_id in all_visible_masks:
            x1, y1, x2, y2 = all_visible_masks[target_object_id]
            box_as_mask_frame[y1:y2, x1:x2] = 1.

        return (np.expand_dims(box_as_mask_frame.astype(np.float),axis=-1))


#
# class NoisyTargetLocationMask(Sensor): #
#     def __init__(self, uuid: str = "target_location_mask", **kwargs: Any):
#         observation_space = gym.spaces.Box(
#             low=0, high=1, shape=(1,), dtype=np.float32
#         )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
#         super().__init__(**prepare_locals_for_super(locals()))
#
#
#     def get_observation(
#             self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
#     ) -> Any:
#         target_object_id = task.task_info['goal_object_id']
#         all_visible_masks = env.controller.last_event.instance_masks
#         if target_object_id in all_visible_masks:
#             mask_frame = all_visible_masks[target_object_id]
#         else:
#             mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)
#
#         result = (np.expand_dims(mask_frame.astype(np.float),axis=-1))
#         fake_mask = random.choice([v for v in env.controller.last_event.instance_masks.values()])
#         fake_mask = (np.expand_dims(fake_mask.astype(np.float),axis=-1))
#
#         return add_mask_noise(result, fake_mask)
#
# class NoisyTargetObjectMask(Sensor):
#
#     def __init__(self, uuid: str = "target_object_mask", **kwargs: Any):
#         observation_space = gym.spaces.Box(
#             low=0, high=1, shape=(1,), dtype=np.float32
#         )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
#         super().__init__(**prepare_locals_for_super(locals()))
#
#     def get_observation(
#             self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
#     ) -> Any:
#         target_object_id = task.task_info['source_object_id']
#         all_visible_masks = env.controller.last_event.instance_masks
#         if target_object_id in all_visible_masks:
#             mask_frame = all_visible_masks[target_object_id]
#         else:
#             mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)
#
#         result = (np.expand_dims(mask_frame.astype(np.float),axis=-1))
#         fake_mask = random.choice([v for v in env.controller.last_event.instance_masks.values()])
#         fake_mask = (np.expand_dims(fake_mask.astype(np.float),axis=-1))
#         return add_mask_noise(result, fake_mask)



class PredictedTargetObjectMask(Sensor): #LATER_TODO if this is too slow then change it
    def __init__(self, uuid: str = "target_object_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))
        self.mask_predictor = ConditionalDetectionModel() #LATER_TODO can we have only one for both of target object and location and also save it in one place. this might explode
        self.mask_predictor.eval()
        print('resolve todos?')
        ForkedPdb().set_trace()
        #LATER_TODO load weights into this

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        with torch.no_grad():
            target_object_id = task.task_info['source_object_id'] #LATER_TODO just have one image instead of this
            target_image = get_target_image(target_object_id)
            target_image = mean_subtract_normalize_image(target_image)
            current_image = torch.Tensor(env.controller.last_event.frame.copy()).float()
            current_image = mean_subtract_normalize_image(current_image)
            #LATER_TODO double check beacause the load and the other shit might be different

            #LATER_TODO visualize the outputs
            predictions = self.mask_predictor(dict(rgb=current_image.unsqueeze(0), target_cropped_object=target_image.unsqueeze(0)))
            probs_mask = predictions['object_mask'].squeeze(0) #Remove batch size
            mask = probs_mask.argmax(dim=0).float().unsqueeze(-1) # To add a channel
            return mask


def get_target_image(object_id): #LATER_TODO implement
    return torch.zeros((224,224, 3))

def mean_subtract_normalize_image(image):
    #LATER_TODO implement
    return image.permute(2,0,1)












def add_noise(result):
    TURN_OFF_RATE = 0.05 # TODO good?
    ADD_PIXEL_RATE = 0.0005 # TODO good?
    REMOVE_RATE = 0.1 # TODO what do you think?
    if random.random() < REMOVE_RATE:
        result[:] = 0.
    else:
        w, h, d = result.shape
        mask = np.random.rand(w, h, d)
        mask = mask < TURN_OFF_RATE
        mask = mask & (result == 1)
        result[mask] = 0


        mask = np.random.rand(w, h, d)
        mask = mask < ADD_PIXEL_RATE
        mask = mask & (result == 0)
        result[mask] = 1

    return result

class NoisyTargetLocationBBox(Sensor): #
    def __init__(self, uuid: str = "target_location_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['goal_object_id']

        all_visible_masks = env.controller.last_event.instance_detections2D
        box_as_mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)
        if target_object_id in all_visible_masks:
            x1, y1, x2, y2 = all_visible_masks[target_object_id]
            box_as_mask_frame[y1:y2, x1:x2] = 1.
        result = (np.expand_dims(box_as_mask_frame.astype(np.float),axis=-1))

        return add_noise(result)



class NoisyTargetObjectBBox(Sensor):
    def __init__(self, uuid: str = "target_object_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['source_object_id']
        all_visible_masks = env.controller.last_event.instance_detections2D
        box_as_mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)
        if target_object_id in all_visible_masks:
            x1, y1, x2, y2 = all_visible_masks[target_object_id]
            box_as_mask_frame[y1:y2, x1:x2] = 1.
        result = (np.expand_dims(box_as_mask_frame.astype(np.float),axis=-1))

        return add_noise(result)
