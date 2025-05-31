from .BaseInitializer import BaseInitializer
from World import BallWorldInformation

import numpy as np

class UniformPositionNormalVelocityInitializer(BaseInitializer[np.ndarray]):
    velocity_variance: np.ndarray
    world: BallWorldInformation

    def __init__(self, velocity_variance: np.ndarray, world: BallWorldInformation):
        """
        Initialize ball positions uniformly but velocities according to a normal distribution.
        """
        self.velocity_variance = velocity_variance
        self.world = world

        if velocity_variance.shape != (2,2):
            raise RuntimeError(f"velocity variance must be shape (2,2), got {velocity_variance.shape}")

    def generate(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed = seed + n)
        position_x = rng.random() * self.world.width
        position_y = rng.random() * self.world.height
        vel = rng.multivariate_normal([0,0], self.velocity_variance)
        return np.array([position_x, position_y, vel[0], vel[1]])
