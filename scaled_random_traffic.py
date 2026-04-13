"""Scaled random SUMO traffic (commonroad_sumo RandomTrafficGenerator extension)."""

from __future__ import annotations

from commonroad.scenario.scenario import Scenario

from commonroad_sumo.cr2sumo.traffic_generator.demand_calculator import RandomDemandCalculator
from commonroad_sumo.cr2sumo.traffic_generator.flow_traffic_generator import RandomTrafficGenerator
from commonroad_sumo.sumolib.sumo_project import SumoProject


class ScaledRandomDemandCalculator(RandomDemandCalculator):
    """Same O/D matrix as RandomDemandCalculator; tune SUMO flow headways.

    commonroad_sumo passes this value into SUMO period=exp(...) as the mean
    inter-departure time in seconds (larger -> sparser). random_density_scale > 1
    divides that mean -> denser traffic.
    """

    def __init__(self, scenario: Scenario, seed: int, density_scale: float) -> None:
        super().__init__(scenario, seed)
        if density_scale <= 0:
            raise ValueError("density_scale must be positive")
        self._density_scale = float(density_scale)

    def get_origin_destination_spawn_rate(self, start_lanelet_id: int, end_lanelet_id: int) -> float:
        if (start_lanelet_id, end_lanelet_id) not in self._od_spawn_probability_matrix:
            return 0.0
        raw = self._od_spawn_probability_matrix[(start_lanelet_id, end_lanelet_id)]
        return round(raw / self._density_scale, 4)


class ScaledRandomTrafficGenerator(RandomTrafficGenerator):
    """RandomTrafficGenerator with density_scale > 1 for denser random traffic."""

    def __init__(
        self,
        map_matching_delta: int = 10,
        seed: int = 1234,
        density_scale: float = 1.0,
    ) -> None:
        super().__init__(map_matching_delta, seed)
        if density_scale <= 0:
            raise ValueError("density_scale must be positive")
        self._density_scale = float(density_scale)

    def generate_traffic(self, scenario: Scenario, sumo_project: SumoProject) -> bool:
        demand = ScaledRandomDemandCalculator(scenario, self._seed, self._density_scale)
        return self._create_flows_from_demand_calculator(scenario, sumo_project, demand)
