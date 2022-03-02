import copy

import torch
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from typing import Any, Union, Optional

import gym
import numpy as np
# from allenact.base_abstractions.sensor import DepthSensor, Sensor, RGBSensor
from allenact.embodiedai.sensors.vision_sensors import DepthSensor, Sensor, RGBSensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
import cv2


from ithor_arm.arm_calculation_utils import (
    convert_world_to_agent_coordinate,
    convert_state_to_tensor,
    diff_position,
)
from ithor_arm.bring_object_sensors import NoisyObjectMask
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_sensors import DepthSensorThor
from ithor_arm.near_deadline_sensors import calc_world_coordinates
from manipulathor_utils.debugger_util import ForkedPdb
from utils.noise_in_motion_util import squeeze_bool_mask


class DepthSensorStretch(
    DepthSensorThor
):
    """Sensor for Depth images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        depth = (env.controller.last_event.depth_frame.copy())
        depth = clip_frame(depth)
        #TODO the ratio of image is slightly different in the real stretch tho
        return depth


class DepthSensorStretchIntel(
    DepthSensorThor
):
    """Sensor for Depth images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        depth = (env.controller.last_event.depth_frame.copy())
        return depth


class DepthSensorStretchKinect(
    DepthSensorThor
):
    """Sensor for Depth images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        depth = env.controller.last_event.third_party_depth_frames[0].copy()
        return depth


class DepthSensorStretchKinectZero(
    DepthSensorThor
):
    """Sensor for Depth images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        depth = env.controller.last_event.third_party_depth_frames[0].copy()
        depth[:] = 0
        return depth

class RGBSensorStretchKinect(
    RGBSensorThor
):
    """Sensor for RGB images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        rgb = env.controller.last_event.third_party_camera_frames[0].copy()
        return rgb


class RGBSensorStretchKinectZero(
    RGBSensorThor
):
    """Sensor for RGB images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        rgb = env.controller.last_event.third_party_camera_frames[0].copy()
        rgb[:] = 0
        return rgb

class RGBSensorStretchIntel(
    RGBSensorThor
):
    """Sensor for RGB images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        rgb = (env.controller.last_event.frame.copy())

        return rgb#cv2.resize(rgb, (224,224))

class NoisyObjectMaskStretch(NoisyObjectMask): #TODO double check correctness of this

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        mask = super().get_observation(env, task, *args, **kwargs)
        return clip_frame(mask)
class RGBSensorStretch(
    RGBSensorThor
):
    """Sensor for RGB images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        rgb = (env.controller.last_event.frame.copy())
        rgb = clip_frame(rgb) #TODO we should add more noise to this as well
        #TODO this is very dorehami
        return rgb
#TODO we need to crop our segmentation masks as well.
MASK_FRAMES = None

def clip_frame(frame):
    #TODO should we swap this w and h?
    if len(frame.shape) == 2:
        w, h = frame.shape
    if len(frame.shape) == 3:
        w, h, c = frame.shape
    if MASK_FRAMES is None or MASK_FRAMES.shape[0] != w or MASK_FRAMES.shape[1] != h:
        set_mask_frames(w, h)
    frame[(1 - MASK_FRAMES).astype(bool)] = 0
    return frame


def set_mask_frames(w, h):
    original_w, original_h = 640, 576
    w_up_left, h_up_left = 150, 270
    w_up_right, h_up_right = 120, 210
    w_down_left, h_down_left = 150,250
    w_down_right, h_down_right = 120, 200

    init = [(0, 0), (original_w, 0), (0, original_h), (original_w, original_h)]
    ws = [w_up_left, -w_up_right, w_down_left, -w_down_right]
    hs = [h_up_left, h_up_right, -h_down_left, -h_down_right]

    mask = np.ones((original_h, original_w))

    for i in range(4):
        pt1 = init[i]
        pt2 = (pt1[0] + ws[i], pt1[1])
        pt3 = (pt1[0], pt1[1] + hs[i])
        triangle_cnt = np.array( [pt1, pt2, pt3] )
        mask = cv2.drawContours(mask, [triangle_cnt], 0, (0), -1)
    global MASK_FRAMES
    MASK_FRAMES = cv2.resize(mask, (h, w))



class AgentBodyPointNavSensor(Sensor):

    def __init__(self, type: str, noise=0, uuid: str = "point_nav_real", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.noise = noise
        assert self.noise == 0
        uuid = '{}_{}'.format(uuid, type)

        super().__init__(**prepare_locals_for_super(locals()))

    def get_accurate_locations(self, env):
        metadata = copy.deepcopy(env.controller.last_event.metadata['agent'])
        return metadata


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        if self.type == 'source':
            info_to_search = 'source_object_id'
        elif self.type == 'destination':
            info_to_search = 'goal_object_id'
        goal_obj_id = task.task_info[info_to_search]
        real_object_info = env.get_object_by_id(goal_obj_id)
        real_agent_state = self.get_accurate_locations(env)
        relative_goal_obj = convert_world_to_agent_coordinate(real_object_info, real_agent_state)
        result = convert_state_to_tensor(dict(position=relative_goal_obj['position']))
        return result
    

class AgentBodyPointNavEmulSensor(Sensor):

    def __init__(self, type: str, mask_sensor:Sensor, depth_sensor:Sensor, uuid: str = "point_nav_emul", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.mask_sensor = mask_sensor
        self.depth_sensor = depth_sensor
        uuid = '{}_{}'.format(uuid, type)

        self.min_xyz = np.zeros((3))
        self.dummy_answer = torch.zeros(3)
        self.dummy_answer[:] = 4 # is this good enough?
        self.device = torch.device("cpu")


        super().__init__(**prepare_locals_for_super(locals()))
    def get_accurate_locations(self, env):
        metadata = copy.deepcopy(env.controller.last_event.metadata)
        camera_xyz = np.array([metadata["cameraPosition"][k] for k in ["x", "y", "z"]])
        camera_rotation=metadata["agent"]["rotation"]["y"]
        camera_horizon=metadata["agent"]["cameraHorizon"]
        arm_state = env.get_absolute_hand_state()
        fov = env.controller.last_event.metadata['fov']
        return dict(camera_xyz=camera_xyz, camera_rotation=camera_rotation, camera_horizon=camera_horizon, arm_state=arm_state, fov=fov)

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        mask = squeeze_bool_mask(self.mask_sensor.get_observation(env, task, *args, **kwargs))
        depth_frame = self.depth_sensor.get_observation(env, task, *args, **kwargs)#env.controller.last_event.depth_frame.copy()
        depth_frame[~mask] = -1
        if task.num_steps_taken() == 0:
            self.pointnav_history_aggr = []
            self.real_prev_location = None
            self.belief_prev_location = None

        agent_locations = self.get_accurate_locations(env)
        camera_xyz = agent_locations['camera_xyz']
        camera_rotation = agent_locations['camera_rotation']
        camera_horizon = agent_locations['camera_horizon']
        arm_state = agent_locations['arm_state']
        fov = agent_locations['fov']

        #TODO we have to rewrite this such that it rotates the object not the agent
        if mask.sum() != 0:
            world_space_point_cloud = calc_world_coordinates(self.min_xyz, camera_xyz, camera_rotation, camera_horizon, fov, self.device, depth_frame)
            valid_points = (world_space_point_cloud == world_space_point_cloud).sum(dim=-1) == 3
            point_in_world = world_space_point_cloud[valid_points]
            middle_of_object = point_in_world.mean(dim=0)
            self.pointnav_history_aggr.append((middle_of_object.cpu(), len(point_in_world), task.num_steps_taken()))

        return self.average_so_far(camera_xyz, camera_rotation, arm_state, task.num_steps_taken())

    def average_so_far(self, camera_xyz, camera_rotation, arm_state, current_step_number):
        if len(self.pointnav_history_aggr) == 0:
            return self.dummy_answer
        else:
            # TODO do the averaging with number of pixels as well
            weights = [1. / (current_step_number + 1 - num_steps) for mid,num_pixels,num_steps in self.pointnav_history_aggr]
            total_weights = sum(weights)
            total_sum = [mid * (1. / (current_step_number + 1 - num_steps)) for mid,num_pixels,num_steps in self.pointnav_history_aggr]
            total_sum = sum(total_sum)
            midpoint = total_sum / total_weights
            agent_state = dict(position=dict(x=camera_xyz[0], y=camera_xyz[1], z=camera_xyz[2], ), rotation=dict(x=0, y=camera_rotation, z=0))
            midpoint_position_rotation = dict(position=dict(x=midpoint[0], y=midpoint[1], z=midpoint[2]), rotation=dict(x=0,y=0,z=0))
            midpoint_agent_coord = convert_world_to_agent_coordinate(midpoint_position_rotation, agent_state)

            distance_in_agent_coord = dict(x=midpoint_agent_coord['position']['x'], y=midpoint_agent_coord['position']['y'], z=midpoint_agent_coord['position']['z'])

            agent_centric_middle_of_object = torch.Tensor([distance_in_agent_coord['x'], distance_in_agent_coord['y'], distance_in_agent_coord['z']])

            agent_centric_middle_of_object = agent_centric_middle_of_object
            return agent_centric_middle_of_object


class ArmPointNavEmulSensor(Sensor):

    def __init__(self, type: str, mask_sensor:Sensor, depth_sensor:Sensor, uuid: str = "arm_point_nav_emul", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.mask_sensor = mask_sensor
        self.depth_sensor = depth_sensor
        uuid = '{}_{}'.format(uuid, type)

        self.min_xyz = np.zeros((3))

        self.dummy_answer = torch.zeros(3)
        self.dummy_answer[:] = 4 # is this good enough?
        self.device = torch.device("cpu")
        super().__init__(**prepare_locals_for_super(locals()))

    def get_accurate_locations(self, env): #TODO this is using camera coordinate be aware of that
        if len(env.controller.last_event.metadata['thirdPartyCameras']) != 1:
            print('Warning multiple cameras')
        metadata = copy.deepcopy(env.controller.last_event.metadata['thirdPartyCameras'][0])
        camera_xyz = np.array([metadata["position"][k] for k in ["x", "y", "z"]])
        # camera_rotation = np.array([metadata["rotation"][k] for k in ["x", "y", "z"]])
        camera_rotation = metadata['rotation']['y']
        camera_horizon = metadata['rotation']['x']
        assert abs(metadata['rotation']['z'] - 0) < 0.1
        arm_state = env.get_absolute_hand_state()
        fov = metadata['fieldOfView'] #TODO this needs to be changed when the cameras have different fov
        return dict(camera_xyz=camera_xyz, camera_rotation=camera_rotation, camera_horizon=camera_horizon, arm_state=arm_state, fov=fov)

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        mask = squeeze_bool_mask(self.mask_sensor.get_observation(env, task, *args, **kwargs))
        depth_frame = self.depth_sensor.get_observation(env, task, *args, **kwargs)#env.controller.last_event.depth_frame.copy()
        depth_frame[~mask] = -1
        if task.num_steps_taken() == 0:
            self.pointnav_history_aggr = []
            self.real_prev_location = None
            self.belief_prev_location = None

        agent_locations = self.get_accurate_locations(env)
        camera_xyz = agent_locations['camera_xyz']
        camera_rotation = agent_locations['camera_rotation']
        camera_horizon = agent_locations['camera_horizon']
        arm_state = agent_locations['arm_state']
        fov = agent_locations['fov']

        #TODO we have to rewrite this such that it rotates the object not the agent
        if mask.sum() != 0:
            world_space_point_cloud = calc_world_coordinates(self.min_xyz, camera_xyz, camera_rotation, camera_horizon, fov, self.device, depth_frame)
            valid_points = (world_space_point_cloud == world_space_point_cloud).sum(dim=-1) == 3
            point_in_world = world_space_point_cloud[valid_points]
            middle_of_object = point_in_world.mean(dim=0)
            self.pointnav_history_aggr.append((middle_of_object.cpu(), len(point_in_world), task.num_steps_taken()))

        return self.average_so_far(camera_xyz, camera_rotation, arm_state, task.num_steps_taken())

    def average_so_far(self, camera_xyz, camera_rotation, arm_state, current_step_number):
        if len(self.pointnav_history_aggr) == 0:
            return self.dummy_answer
        else:
            weights = [1. / (current_step_number + 1 - num_steps) for mid,num_pixels,num_steps in self.pointnav_history_aggr]
            total_weights = sum(weights)
            total_sum = [mid * (1. / (current_step_number + 1 - num_steps)) for mid,num_pixels,num_steps in self.pointnav_history_aggr]
            total_sum = sum(total_sum)
            midpoint = total_sum / total_weights
            agent_state = dict(position=dict(x=camera_xyz[0], y=camera_xyz[1], z=camera_xyz[2], ), rotation=dict(x=0, y=camera_rotation, z=0))
            midpoint_position_rotation = dict(position=dict(x=midpoint[0], y=midpoint[1], z=midpoint[2]), rotation=dict(x=0,y=0,z=0))
            midpoint_agent_coord = convert_world_to_agent_coordinate(midpoint_position_rotation, agent_state)

            arm_state_agent_coord = convert_world_to_agent_coordinate(arm_state, agent_state)
            distance_in_agent_coord = dict(x=midpoint_agent_coord['position']['x'] - arm_state_agent_coord['position']['x'],y=midpoint_agent_coord['position']['y'] - arm_state_agent_coord['position']['y'],z=midpoint_agent_coord['position']['z'] - arm_state_agent_coord['position']['z'])

            # distance_in_agent_coord = dict(x=midpoint_agent_coord['position']['x'], y=midpoint_agent_coord['position']['y'], z=midpoint_agent_coord['position']['z'])

            agent_centric_middle_of_object = torch.Tensor([distance_in_agent_coord['x'], distance_in_agent_coord['y'], distance_in_agent_coord['z']])

            # Removing this hurts the performance
            agent_centric_middle_of_object = agent_centric_middle_of_object #.abs() TODO investigate removing this again


            # # remove
            # TODO remove
            # if self.type == 'source':
            #
            #     obj_id = self.task.task_info['source_object_id']
            #     obj_real_location = self.env.get_object_by_id(obj_id)
            #     obj_real_relative = convert_world_to_agent_coordinate(obj_real_location, agent_state)
            #     arm_real_relative = arm_state_agent_coord
            #     real_distance_in_world_coord = torch.Tensor([obj_real_location['position']['x'] - arm_state['position']['x'],obj_real_location['position']['y'] - arm_state['position']['y'],obj_real_location['position']['z'] - arm_state['position']['z']])
            #     real_distance_in_agent_coord = torch.Tensor([obj_real_relative['position']['x'] - arm_real_relative['position']['x'],obj_real_relative['position']['y'] - arm_real_relative['position']['y'],obj_real_relative['position']['z'] - arm_real_relative['position']['z']])
            #     pred_distance_in_agent_coord = agent_centric_middle_of_object
            #
            #     print('real_distance_in_world_coord', real_distance_in_world_coord, real_distance_in_world_coord.norm())
            #     print('real_distance_in_agent_coord', real_distance_in_agent_coord, real_distance_in_agent_coord.norm())
            #     print('pred_distance_in_agent_coord', pred_distance_in_agent_coord, pred_distance_in_agent_coord.norm())
            #     # ForkedPdb().set_trace()

            return agent_centric_middle_of_object

# TODO we have to rewrite the noisy movement experiment ones?
# class AgentGTLocationSensor(Sensor):
#
#     def __init__(self, uuid: str = "agent_gt_loc", **kwargs: Any):
#         observation_space = gym.spaces.Box(
#             low=0, high=1, shape=(1,), dtype=np.float32
#         )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
#         super().__init__(**prepare_locals_for_super(locals()))
#
#     def get_observation(
#             self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
#     ) -> Any:
#         metadata = copy.deepcopy(env.controller.last_event.metadata['agent'])
#         return metadata


class IntelRawDepthSensor(Sensor):

    def __init__(self, uuid: str = "intel_raw_depth", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return env.controller.last_event.depth_frame.copy()

class KinectRawDepthSensor(Sensor):

    def __init__(self, uuid: str = "kinect_raw_depth", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return env.controller.last_event.third_party_depth_frames[0].copy()


class KinectNoisyObjectMask(Sensor):
    def __init__(self, type: str,noise, height, width,  uuid: str = "object_mask_kinect", distance_thr: float = -1, **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.height = height
        self.width = width
        uuid = '{}_{}'.format(uuid, type)
        self.noise = noise
        self.distance_thr = distance_thr
        super().__init__(**prepare_locals_for_super(locals()))
        assert self.noise == 0

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
        all_visible_masks = env.controller.last_event.third_party_instance_masks[0]
        if len(env.controller.last_event.third_party_instance_masks) != 1:
            print('Warning multiple cameras')
        # assert len(env.controller.last_event.third_party_instance_masks) == 1
        if target_object_id in all_visible_masks:
            mask_frame = all_visible_masks[target_object_id]

            if self.distance_thr > 0:

                agent_location = env.get_agent_location()
                object_location = env.get_object_by_id(target_object_id)['position']
                current_agent_distance_to_obj = sum([(object_location[k] - agent_location[k])**2 for k in ['x', 'z']]) ** 0.5
                #TODO remove this or put back?
                # if current_agent_distance_to_obj > self.distance_thr or mask_frame.sum() < 20: # objects that are smaller than this many pixels should be removed. High chance all spatulas will be removed
                #     mask_frame[:] = 0

        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        mask_frame = (np.expand_dims(mask_frame.astype(np.float),axis=-1))

        current_shape = mask_frame.shape
        if (current_shape[0], current_shape[1]) == (self.width, self.height):
            resized_mask = mask_frame
        else:
            resized_mask = cv2.resize(mask_frame, (self.height, self.width)).reshape(self.width, self.height, 1) # my gut says this is gonna be slow

        return resized_mask
