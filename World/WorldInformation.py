"""
Actual or assumed world parameters.
"""

from dataclasses import dataclass

@dataclass
class BallWorldInformation:
    width: float
    height: float
    gravity: float
    ball_radius: float
    bounce_discount: float
    air_discount: float
    ground_discount: float
