"""
Parameters of one Ball Estimation run.
"""
from dataclasses import dataclass

@dataclass
class SimulationParameters:
    number_of_balls: int = 3
    width: int = 50
    height: int = 50
    gravity: float = 9.8
    ball_radius: float = 1
    bounce_discount: float = 1
    air_discount: float = 1
    ground_discount: float = 1
    sensor_variance: tuple[float, float] = (5, 5)
    initial_velocity_variance: tuple[float, float] = (100, 100)

    assumed_number_of_balls: int = 3
    assumed_width: int = 50
    assumed_height: int = 50
    assumed_gravity: float = 9.8
    assumed_ball_radius: float = 1
    assumed_bounce_discount: float = 1
    assumed_air_discount: float = 1
    assumed_ground_discount: float = 1
    assumed_sensor_variance: tuple[float, float] = (5, 5)
    assumed_initial_velocity_variance: tuple[float, float] = (100, 100)
    transition_velocity_variance: tuple[float, float] = (2,2)
    
    measurements_per_second: int = 30
    number_of_particles: int = 2000

    seed: int = 0
    live_show: bool = True
    visualize_tail_length: int = 15
    max_steps: int = 1000
    show_particles: bool = True
    show_observations: bool = True
    show_actual_positions: bool = True
    show_summary_plots: bool = False
    
