from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


class TollgateEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.2,
                "reward_speed_range": [20, 30],
                "tollgate_speed_reward": -0.5,
                "lane_change_reward": -0.05,
            }
        )
        return cfg

    def _reward(self, action: int) -> float:
        reward = sum(
            self.config.get(name, 0) * reward
            for name, reward in self._rewards(action).items()
        )
        return utils.lmap(
            reward,
            [
                self.config["collision_reward"] + self.config["tollgate_speed_reward"],
                self.config["high_speed_reward"] + self.config["right_lane_reward"],
            ],
            [0, 1],
        )

    def _rewards(self, action: int) -> Dict[Text, float]:
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": self.vehicle.lane_index[2] / 1,
            "high_speed_reward": scaled_speed,
            "lane_change_reward": action in [0, 2],
            "tollgate_speed_reward": sum(
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2)
                and isinstance(vehicle, ControlledVehicle)
            ),
        }

    def _is_terminated(self) -> bool:
        print("crash" + str(self.vehicle.crashed))
        print("over" + str(self.vehicle.position[0] > 370))
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, tollgate, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [i * StraightLane.DEFAULT_WIDTH for i in range(4)]  # 4 lanes
        line_type = [[c, s], [n, c]]
        line_type_tollgate = [[c, s], [n, s]]
        
        # a -> b 구간: 4개 차선
        for i in range(4):
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i % 2]),
            )
        
        y = [i * StraightLane.DEFAULT_WIDTH for i in range(8)]  # 8 lanes
        # b -> c 구간: 8개 차선
        for i in range(8):
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]],
                    line_types=line_type_tollgate[i % 2],
                ),
            )
        
        y = [i * StraightLane.DEFAULT_WIDTH for i in range(4)]  # 4 lanes
        # c -> d 구간: 다시 4개 차선
        for i in range(4):
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i % 2]
                ),
            )

        # Tollgate lane
        amplitude = 3.25
        ljk = StraightLane(
            [0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True
        )
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(ends[1], 0),
            lkb.position(ends[1], 0) + [ends[2], 0],
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 1)).position(30, 0), speed=30
        )
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        for position, speed in [(90, 29), (70, 31), (5, 31.5)]:
            lane = road.network.get_lane(("a", "b", self.np_random.integers(4)))
            position = lane.position(position + self.np_random.uniform(-5, 5), 0)
            speed += self.np_random.uniform(-1, 1)
            road.vehicles.append(other_vehicles_type(road, position, speed=speed))

        tollgate_v = other_vehicles_type(
            road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20
        )
        tollgate_v.target_speed = 30
        road.vehicles.append(tollgate_v)
        self.vehicle = ego_vehicle
