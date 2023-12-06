from typing import Dict, Text, Tuple

import numpy as np
from torch import norm

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.vehicle.objects import Obstacle
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle


class ExitEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approaches an exit ramp.
    It is rewarded for maintaining merging into the exit lane and minimzing the time spent waiting until it reach the end of the ramp.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 2,
                "exit_length": 500,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "exit_lane_reward": 10,  # The reward received when entering the exit lane. This is scaled based on distance to the end of the exit ramp
                "missed_exit_reward": -5,  # punish missing the exit
                "lane_change_reward": 0.5,  # The reward received at each lane change action. Positive for being close to the exit lane, negative for moving away
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": True,
            }
        )
        return cfg

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the exit ramp has reached the end."""
        print("crash" + str(self.vehicle.crashed))
        print("over" + str(self.vehicle.position[0] > self._tot))
        return self.vehicle.crashed or bool(self.vehicle.position[0] > self._tot)

    def _is_success(self):
        lane_index = (
            self.vehicle.target_lane_index
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index
        )
        goal_reached = lane_index == (
            "1",
            "2",
            self.config["lanes_count"],
        ) or lane_index == ("2", "exit", 0)
        return goal_reached

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()
        num_lanes = self.config["lanes_count"]
        exit_length = self.config["exit_length"]

        # Highway lanes
        ends = [
            200,
            exit_length,
            200,
            50,
        ]
        self._tot = sum(ends)
        # exit_start, exit lane diverge,ramp length, end
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(self.config["lanes_count"]):
            # each tuple is x,y coordinate of the line to draw
            # add the first segment of two lanes
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [0, StraightLane.DEFAULT_WIDTH * i],
                    [sum(ends[:2]), StraightLane.DEFAULT_WIDTH * i],
                    line_types=line_type_merge[i],
                ),
            )

            # add final segment after exit lane
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(ends[:2]), StraightLane.DEFAULT_WIDTH * i],
                    [sum(ends), StraightLane.DEFAULT_WIDTH * i],
                    line_types=line_type[i],
                ),
            )

        # Exit lane
        amplitude = 3.25
        exit_start = StraightLane(
            [0, StraightLane.DEFAULT_WIDTH * num_lanes],
            [sum(ends[:2]), StraightLane.DEFAULT_WIDTH * num_lanes],
            line_types=[s, c],
        )
        exit_ramp = SineLane(
            exit_start.position(ends[0] + exit_length, amplitude),
            exit_start.position(ends[0] + exit_length + ends[2], amplitude),
            -amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        end_seg = StraightLane(
            exit_ramp.position(ends[2], 0),
            exit_ramp.position(ends[2], 0) + [ends[3], 0],
            line_types=[c, c],
            forbidden=True,
        )
        net.add_lane("a", "b", exit_start)
        net.add_lane("j", "k", exit_ramp)
        net.add_lane("k", "l", end_seg)

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        # road.objects.append(Obstacle(road, end_seg.position(ends[3], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 1)).position(0, 0), speed=30
        )
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # place cars on the main highway randomly
        for position, speed in [(90, 29), (70, 31), (5, 31.5)]:
            lane = road.network.get_lane(("a", "b", self.np_random.integers(3)))
            position = lane.position(position + self.np_random.uniform(-5, 5), 0)
            speed += self.np_random.uniform(-1, 1)
            road.vehicles.append(other_vehicles_type(road, position, speed=speed))

        for position in [110, 130, 150]:
            merging_v = other_vehicles_type(
                road,
                road.network.get_lane(("a", "b", 2)).position(position, 0),
                speed=20,
            )
            merging_v.target_speed = 15
        road.vehicles.append(merging_v)

        self.vehicle = ego_vehicle

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = sum(
            self.config.get(name, 0) * reward
            for name, reward in self._rewards(action).items()
        )
        print(reward)
        norm_reward = utils.lmap(
            reward,
            [
                self.config["collision_reward"] + self.config["missed_exit_reward"],
                self.config["lane_change_reward"]
                + self.config["exit_lane_reward"] * 1.75,
            ],
            [0, 1],
        )

        print(norm_reward)
        return norm_reward

    def _rewards(self, action: int) -> Dict[Text, float]:
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        # this map says how to scale the reward passed in the config
        rewards_dict: Dict[Text, float] = {}

        # if crashed then scale the collision reward by 1 if not then by 0
        rewards_dict["collision_reward"] = self.vehicle.crashed

        # scale the exit lane reward by if I'm in the exit lane and how far I am from the exit
        if self.vehicle.lane_index[2] == self.config["lanes_count"]:
            # exit lane index is the lanes count + 1
            # if in the exit lane, grab the distance to the next lane segment which is the exit ramp
            distance_to_exit = (
                self.road.network.get_lane(
                    ("a", "b", self.config["lanes_count"])
                ).position(self.config["exit_length"], 0)[0]
                - self.vehicle.position[0]
            )

            if distance_to_exit < self.config["exit_length"] * 1 / 4:
                # large reward for joining early at last minute
                rewards_dict["exit_lane_reward"] = 1.75
            elif distance_to_exit < self.config["exit_length"] / 2:
                # medium reward for joining in last half
                rewards_dict["exit_lane_reward"] = 1.5
            else:
                rewards_dict["exit_lane_reward"] = 1

        else:
            # if not in the exit lane then dont reward
            rewards_dict["exit_lane_reward"] = 0

        # scale the lane change reward by if I'm moving towards or away from the exit lane
        if action == 0:
            rewards_dict["lane_change_reward"] = -1
        elif action == 2:
            rewards_dict["lane_change_reward"] = 1
        else:
            rewards_dict["lane_change_reward"] = 0

        # scale the reward for missing the exit
        if self.vehicle.lane_index[0] == "b":
            rewards_dict["missed_exit_reward"] = 1
        else:
            rewards_dict["missed_exit_reward"] = 0

        return rewards_dict


'''
    def _create_road(
        self, road_length=1000, exit_position=400, exit_length=100
    ) -> None:
        net = RoadNetwork.straight_road_network(
            self.config["lanes_count"],
            start=0,
            length=exit_position,
            nodes_str=("0", "1"),
        )
        net = RoadNetwork.straight_road_network(
            self.config["lanes_count"] + 1,
            start=exit_position,
            length=exit_length,
            nodes_str=("1", "2"),
            net=net,
        )
        net = RoadNetwork.straight_road_network(
            self.config["lanes_count"],
            start=exit_position + exit_length,
            length=road_length - exit_position - exit_length,
            nodes_str=("2", "3"),
            net=net,
        )
        for _from in net.graph:
            for _to in net.graph[_from]:
                for _id in range(len(net.graph[_from][_to])):
                    net.get_lane((_from, _to, _id)).speed_limit = 26 - 3.4 * _id
        exit_position = np.array(
            [
                exit_position + exit_length,
                self.config["lanes_count"] * CircularLane.DEFAULT_WIDTH,
            ]
        )
        radius = 150
        exit_center = exit_position + np.array([0, radius])
        lane = CircularLane(
            center=exit_center,
            radius=radius,
            start_phase=3 * np.pi / 2,
            end_phase=2 * np.pi,
            forbidden=True,
        )
        net.add_lane("2", "exit", lane)

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for _ in range(self.config["controlled_vehicles"]):
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_from="0",
                lane_to="1",
                lane_id=0,
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            lanes = np.arange(self.config["lanes_count"])
            lane_id = self.road.np_random.choice(
                lanes, size=1, p=lanes / lanes.sum()
            ).astype(int)[0]
            lane = self.road.network.get_lane(("0", "1", lane_id))
            vehicle = vehicles_type.create_random(
                self.road,
                lane_from="0",
                lane_to="1",
                lane_id=lane_id,
                speed=lane.speed_limit,
                spacing=1 / self.config["vehicles_density"],
            ).plan_route_to("3")
            vehicle.enable_lane_change = False
            self.road.vehicles.append(vehicle)'''
