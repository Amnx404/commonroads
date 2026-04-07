"""Generate a CommonRoad merge scenario and render a simple simulation animation.

Usage:
    python generate_merge_simulation.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario, ScenarioID, Tag
from commonroad.scenario.state import InitialState, KSState
from commonroad.scenario.trajectory import Trajectory
from commonroad.geometry.shape import Rectangle
from commonroad.common.common_lanelet import LaneletType
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.visualization.mp_renderer import MPRenderer


OUTPUT_DIR = Path("outputs")
DT = 0.2
STEPS = 60


def straight_lanelet(lanelet_id: int, y_center: float, y_min: float, y_max: float, x_start: float, x_end: float, predecessor=None, successor=None):
    xs = np.linspace(x_start, x_end, 61)
    center = np.column_stack([xs, np.full_like(xs, y_center)])
    left = np.column_stack([xs, np.full_like(xs, y_max)])
    right = np.column_stack([xs, np.full_like(xs, y_min)])
    return Lanelet(
        left_vertices=left,
        center_vertices=center,
        right_vertices=right,
        lanelet_id=lanelet_id,
        predecessor=predecessor or [],
        successor=successor or [],
        lanelet_type={LaneletType.INTERSTATE},
    )


def merging_lanelet(lanelet_id: int, x_start: float, x_end: float, y_start: float, y_end: float, width: float, predecessor=None, successor=None):
    xs = np.linspace(x_start, x_end, 61)
    ys = np.linspace(y_start, y_end, 61)
    center = np.column_stack([xs, ys])
    left = np.column_stack([xs, ys + width / 2])
    right = np.column_stack([xs, ys - width / 2])
    return Lanelet(
        left_vertices=left,
        center_vertices=center,
        right_vertices=right,
        lanelet_id=lanelet_id,
        predecessor=predecessor or [],
        successor=successor or [],
        lanelet_type={LaneletType.ACCESS_RAMP},
    )


def make_obstacle(obstacle_id: int, positions: list[tuple[float, float]], velocity: float, orientation: float) -> DynamicObstacle:
    states = [
        KSState(
            time_step=t,
            position=np.array([x, y]),
            velocity=velocity,
            orientation=orientation,
            steering_angle=0.0,
        )
        for t, (x, y) in enumerate(positions)
    ]

    shape = Rectangle(length=4.7, width=2.0)
    initial_state = InitialState(
        time_step=0,
        position=np.array(positions[0]),
        velocity=velocity,
        orientation=orientation,
        acceleration=0.0,
        yaw_rate=0.0,
        slip_angle=0.0,
    )

    trajectory = Trajectory(initial_time_step=1, state_list=states[1:])
    prediction = TrajectoryPrediction(trajectory=trajectory, shape=shape)

    return DynamicObstacle(
        obstacle_id=obstacle_id,
        obstacle_type=ObstacleType.CAR,
        obstacle_shape=shape,
        initial_state=initial_state,
        prediction=prediction,
    )


def build_scenario() -> Scenario:
    scenario = Scenario(
        dt=DT,
        scenario_id=ScenarioID(country_id="USA", map_name="MERGE", map_id=1),
        author="Codex",
        source="Generated with commonroad-io",
        tags={Tag.INTERSTATE},
    )

    lane_main_a = straight_lanelet(1, y_center=0.0, y_min=-2.0, y_max=2.0, x_start=0.0, x_end=80.0, successor=[3])
    lane_merge = merging_lanelet(2, x_start=0.0, x_end=80.0, y_start=6.0, y_end=0.0, width=4.0, successor=[3])
    lane_main_b = straight_lanelet(3, y_center=0.0, y_min=-2.0, y_max=2.0, x_start=80.0, x_end=200.0, predecessor=[1, 2])

    network = LaneletNetwork.create_from_lanelet_list([lane_main_a, lane_merge, lane_main_b])
    scenario.add_objects(network)

    # Ego-like car on main lane
    main_positions = [(5 + 2.2 * t, 0.0) for t in range(STEPS)]
    main_car = make_obstacle(101, main_positions, velocity=11.0, orientation=0.0)

    # Merging car ramps down into the main lane
    merge_positions = [(2 + 2.0 * t, max(0.0, 6.0 - 0.12 * t * 5)) for t in range(STEPS)]
    merge_car = make_obstacle(102, merge_positions, velocity=10.0, orientation=0.0)

    scenario.add_objects(main_car)
    scenario.add_objects(merge_car)
    return scenario


def save_xml(scenario: Scenario, output_xml: Path) -> None:
    pps = PlanningProblemSet()
    writer = CommonRoadFileWriter(
        scenario=scenario,
        planning_problem_set=pps,
        author="Codex",
        affiliation="OpenAI",
        source="Procedural merge scenario",
        tags={Tag.INTERSTATE},
    )
    writer.write_to_file(str(output_xml), overwrite_existing_file=OverwriteExistingFile.ALWAYS)


def render_animation(scenario: Scenario, output_gif: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    def draw_frame(t: int):
        ax.clear()
        renderer = MPRenderer(plot_limits=[0, 200, -4, 8], ax=ax)
        draw_params = MPDrawParams()
        draw_params.time_begin = t
        scenario.draw(renderer, draw_params=draw_params)
        renderer.render()
        ax.set_title(f"Merge simulation (t={t*DT:.1f}s)")

    anim = animation.FuncAnimation(fig, draw_frame, frames=STEPS, interval=100)
    anim.save(output_gif, writer="pillow", fps=10)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenario = build_scenario()

    xml_path = OUTPUT_DIR / "USA_MERGE-1_1_T-1.xml"
    gif_path = OUTPUT_DIR / "merge_simulation.gif"

    save_xml(scenario, xml_path)
    render_animation(scenario, gif_path)

    print(f"Wrote scenario XML: {xml_path}")
    print(f"Wrote simulation GIF: {gif_path}")


if __name__ == "__main__":
    main()
