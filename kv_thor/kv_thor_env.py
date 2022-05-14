"""lets start with an env that, at the beginning the robot is facing a table and holding a cup (or any pickable
object in ManipulaTHOR), and the task of the robot is to place the cup at a specified position on the table. """
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_task_samplers import ArmPointNavTaskSampler
"""
Initial State:
    - robot is at the table
    - robot is holding a cup (or any pickable object in ManipulaTHOR)
Goal State:
    - robot is at the table
    - robot places the cup at the specified position on the table
"""


class KVThorEnv(object):
    def __init__(self,
                 x_display: Optional[str] = None,
                 docker_enabled: bool = False,
                 local_thor_build: Optional[str] = None,
                 visibility_distance: float = VISIBILITY_DISTANCE,
                 fov: float = FOV,
                 player_screen_width: int = 224,
                 player_screen_height: int = 224,
                 quality: str = "Very Low",
                 restrict_to_initially_reachable_points: bool = False,
                 make_agents_visible: bool = True,
                 object_open_speed: float = 1.0,
                 simplify_physics: bool = False,
                 verbose: bool = False,
                 env_args=None, ):
        self.env = ManipulaTHOREnvironment(x_display=x_display, docker_enabled=docker_enabled,
                                           local_thor_build=local_thor_build, visibility_distance=visibility_distance,
                                           fov=fov, player_screen_width=player_screen_width,
                                           player_screen_height=player_screen_height, quality=quality,
                                           restrict_to_initially_reachable_points=restrict_to_initially_reachable_points,
                                           make_agents_visible=make_agents_visible, object_open_speed=object_open_speed,
                                           simplify_physics=simplify_physics, verbose=verbose, env_args=env_args)



    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def close(self):
        pass
