from typing import Dict, List, Optional
from typing_extensions import Literal
import math

from manipulathor_utils.debugger_util import ForkedPdb
from omegaconf import DictConfig, OmegaConf
# from ai2thor.controller import Controller
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment
from ai2thor.util import metrics
from allenact.utils.cache_utils import DynamicDistanceCache
from allenact.utils.system import get_logger
from utils.procthor_utils.procthor_types import Vector3


# def get_reachable_positions_procthor(controller):
#     ForkedPdb().set_trace()
#     event = controller.step('GetReachablePositions')
#     reachable_positions = event.metadata['actionReturn']
#     return reachable_positions


def position_dist(
    p0: Vector3,
    p1: Vector3,
    ignore_y: bool = False,
    dist_fn: Literal["l1", "l2"] = "l2",
) -> float:
    """Distance between two points of the form {"x": x, "y": y, "z": z}."""
    if dist_fn == "l1":
        return (
            abs(p0["x"] - p1["x"])
            + (0 if ignore_y else abs(p0["y"] - p1["y"]))
            + abs(p0["z"] - p1["z"])
        )
    elif dist_fn == "l2":
        return math.sqrt(
            (p0["x"] - p1["x"]) ** 2
            + (0 if ignore_y else (p0["y"] - p1["y"]) ** 2)
            + (p0["z"] - p1["z"]) ** 2
        )
    else:
        raise NotImplementedError(
            'dist_fn must be in {"l1", "l2"}.' f" You gave {dist_fn}"
        )


def distance_to_object_id(
    env: StretchManipulaTHOREnvironment,
    distance_cache: DynamicDistanceCache,
    object_id: str,
    house_name: str,
) -> Optional[float]:
    """Minimal geodesic distance to object of given objectId from agent's
    current location.
    It might return -1.0 for unreachable targets.
    # TODO: return None for unreachable targets.
    """

    def path_from_point_to_object_id(
        point: Dict[str, float], object_id: str, allowed_error: float
    ) -> Optional[List[Dict[str, float]]]:
        event = controller.step(
            action="GetShortestPath",
            objectId=object_id,
            position=point,
            allowedError=allowed_error,
        )
        if event:
            return event.metadata["actionReturn"]["corners"]
        else:
            get_logger().debug(
                f"Failed to find path for {object_id} in {house_name}."
                f' Start point {point}, agent state {event.metadata["agent"]}.'
            )
            return None

    def distance_from_point_to_object_id(
        point: Dict[str, float], object_id: str, allowed_error: float
    ) -> float:
        """Minimal geodesic distance from a point to an object of the given
        type.
        It might return -1.0 for unreachable targets.
        """
        path = path_from_point_to_object_id(point, object_id, allowed_error)
        if path:
            # Because `allowed_error != 0` means that the path returned above might not start
            # at `point`, we explicitly add any offset there is.
            dist = position_dist(p0=point, p1=path[0], ignore_y=True)
            return metrics.path_distance(path) + dist
        return -1.0

    def retry_dist(position: Dict[str, float], object_id: str) -> float:
        allowed_error = 0.05
        debug_log = ""
        d = -1.0
        while allowed_error < 2.5:
            d = distance_from_point_to_object_id(position, object_id, allowed_error)
            if d < 0:
                debug_log = (
                    f"In house {house_name}, could not find a path from {position} to {object_id} with"
                    f" {allowed_error} error tolerance. Increasing this tolerance to"
                    f" {2 * allowed_error} any trying again."
                )
                allowed_error *= 2
            else:
                break
        if d < 0:
            get_logger().warning(
                f"In house {house_name}, could not find a path from {position} to {object_id}"
                f" with {allowed_error} error tolerance. Returning a distance of -1."
            )
        elif debug_log != "":
            get_logger().debug(debug_log)
        return d

    return distance_cache.find_distance(
        scene_name=house_name,
        position=controller.last_event.metadata["agent"]["position"],
        target=object_id,
        native_distance_function=retry_dist,
    )

def spl_metric(
    success: bool, optimal_distance: float, travelled_distance: float
) -> Optional[float]:
    # TODO: eventually should be -> float
    if optimal_distance < 0:
        # TODO: update when optimal_distance must be >= 0.
        # raise ValueError(
        #     f"optimal_distance must be >= 0. You gave: {optimal_distance}."
        # )
        # return None
        return 0.0
    elif not success:
        return 0.0
    elif optimal_distance == 0:
        return 1.0 if travelled_distance == 0 else 0.0
    else:
        return optimal_distance / max(travelled_distance, optimal_distance)
