from ithor_arm.ithor_arm_task_samplers import ArmPointNavTaskSampler

class ArmPointNavWithObjectTaskSampler(ArmPointNavTaskSampler):


    def get_source_target_indices(self):
        if self.sampler_mode == "train":
            valid_countertops = [
                k for (k, v) in self.countertop_object_to_data_id.items() if len(v) > 1
            ]
            countertop_id = random.choice(valid_countertops)
            indices = random.sample(self.countertop_object_to_data_id[countertop_id], 2)
            result = (
                self.all_possible_points[indices[0]],
                self.all_possible_points[indices[1]],
            )
            scene_name = result[0]["scene_name"]

            # selected_agent_init_loc = self.get_pickable_pose(self.possible_agent_reachable_poses[scene_name], result[0])
            # initial_agent_pose = {
            #     "name": "agent",
            #     "position": {
            #         "x": selected_agent_init_loc["x"],
            #         "y": selected_agent_init_loc["y"],
            #         "z": selected_agent_init_loc["z"],
            #     },
            #     "rotation": {
            #         "x": -0.0,
            #         "y": selected_agent_init_loc["rotation"],
            #         "z": 0.0,
            #     },
            #     "cameraHorizon": selected_agent_init_loc["horizon"],
            #     "isStanding": True,
            # }
            # result[0]["initial_agent_pose"] = initial_agent_pose
        else:  # we need to fix this for test set, agent init location needs to be fixed, therefore we load a fixed valid agent init that is previously randomized
            result = self.deterministic_data_list[self.sampler_index]["datapoint"]
            scene_name = self.deterministic_data_list[self.sampler_index]["scene"]
            # datapoint_original_index = self.deterministic_data_list[self.sampler_index][
            #     "index"
            # ]
            # selected_agent_init_loc = self.possible_agent_reachable_poses[scene_name][
            #     datapoint_original_index
            # ]
            # initial_agent_pose = {
            #     "name": "agent",
            #     "position": {
            #         "x": selected_agent_init_loc["x"],
            #         "y": selected_agent_init_loc["y"],
            #         "z": selected_agent_init_loc["z"],
            #     },
            #     "rotation": {
            #         "x": -0.0,
            #         "y": selected_agent_init_loc["rotation"],
            #         "z": 0.0,
            #     },
            #     "cameraHorizon": selected_agent_init_loc["horizon"],
            #     "isStanding": True,
            # }
            # result[0]["initial_agent_pose"] = initial_agent_pose
            # self.sampler_index += 1

        return result

    def get_pickable_pose(self, possible_agent_reachable_poses, datapoint):
        # get the closest reachable pose
        p1 = np.array(datapoint["x"], datapoint["y"], datapoint["z"])
        closest_pose = None
        closest_pose_distance = None
        for pose in possible_agent_reachable_poses:
            p2 = np.array(pose["x"], pose["y"], pose["z"])
            distance = self.get_distance(pose, datapoint)
            if closest_pose_distance is None or distance < closest_pose_distance:
                closest_pose = pose
                closest_pose_distance = distance
        return closest_pose
    def next_task(
            self, force_advance_scene: bool = False
    ) -> Optional[AbstractPickUpDropOffTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.sampler_mode != "train" and self.length <= 0:
            return None

        source_data_point, target_data_point = self.get_source_target_indices()

        scene = source_data_point["scene_name"]

        assert source_data_point["object_id"] == target_data_point["object_id"]
        assert source_data_point["scene_name"] == target_data_point["scene_name"]

        if self.env is None:
            self.env = self._create_environment()

        self.env.reset(
            scene_name=scene, agentMode="arm", agentControllerType="mid-level"
        )

        event1, event2, event3 = initialize_arm(self.env.controller)

        source_location = source_data_point
        target_location = dict(
            position=target_data_point["object_location"],
            rotation={"x": 0, "y": 0, "z": 0},
        )

        task_info = {
            "objectId": source_location["object_id"],
            "countertop_id": source_location["countertop_id"],
            "source_location": source_location,
            "target_location": target_location,
        }

        this_controller = self.env

        event = transport_wrapper(
            this_controller,
            source_location["object_id"],
            source_location["object_location"],
        )
        agent_state = source_location["agent_pose"]

        event = this_controller.step(
            dict(
                action="TeleportFull",
                standing=True,
                x=agent_state["position"]["x"],
                y=agent_state["position"]["y"],
                z=agent_state["position"]["z"],
                rotation=dict(
                    x=agent_state["rotation"]["x"],
                    y=agent_state["rotation"]["y"],
                    z=agent_state["rotation"]["z"],
                ),
                horizon=agent_state["cameraHorizon"],
            )
        )

        #Pickup Object
        event = this_controller.step(
            dict(
                action="PickupMidLevel",
                objectId=source_location["object_id"]
            )
        )



        should_visualize_goal_start = [
            x for x in self.visualizers if issubclass(type(x), ImageVisualizer)
        ]
        if len(should_visualize_goal_start) > 0:
            task_info["visualization_source"] = source_data_point
            task_info["visualization_target"] = target_data_point

        self._last_sampled_task = self._TASK_TYPE(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            visualizers=self.visualizers,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task


