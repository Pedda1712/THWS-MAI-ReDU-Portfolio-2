import numpy as np

from World import BallWorldInformation
from .IdentityProcess import IdentityProcess

class BallArenaProcess(IdentityProcess[np.ndarray]):
    world_information: BallWorldInformation

    def __init__(self, world_information: BallWorldInformation, tol: float = 1e-4):
        """
        The actual Process of the ball world.
        
        This is where gravity, collision, friction happens.

        Parameters:
          world_information: actual world parameters
        """
        self.world_information = world_information
        self.tol = tol

    def _transition_one(self, state: np.ndarray, delta: float) -> np.ndarray:
        if state.shape != (4,):
            raise RuntimeWarning("ball state must consist of (pos_x, pos_y, vel_x, vel_y)")

        pos = state[:2].astype(float)
        vel = state[2:].astype(float)

        pos += vel * delta

        # top collision
        if pos[1] + self.world_information.ball_radius > self.world_information.height:
            pos[1] = self.world_information.height - self.world_information.ball_radius
            vel[1] = -vel[1] * self.world_information.bounce_discount

        # bottom collision
        if pos[1] - self.world_information.ball_radius < 0:
            pos[1] = self.world_information.ball_radius
            vel[1] = -vel[1] * self.world_information.bounce_discount

        # right collision
        if pos[0] + self.world_information.ball_radius > self.world_information.width:
            pos[0] = self.world_information.width - self.world_information.ball_radius
            vel[0] = -vel[0] * self.world_information.bounce_discount

        # left collision
        if pos[0] - self.world_information.ball_radius < 0:
            pos[0] = self.world_information.ball_radius
            vel[0] = -vel[0] * self.world_information.bounce_discount

        # air resistance
        vel = vel * (self.world_information.air_discount ** delta)
        vel[1] = vel[1] - self.world_information.gravity * delta

        # ground friction
        if np.abs(pos[1] - self.world_information.ball_radius - 0) < self.tol:
            vel = vel * (self.world_information.ground_discount ** delta)

        return np.concatenate((pos, vel))
        
    def transition(self, states: list[np.ndarray], delta: float = 1, seed: int = 0) -> list[np.ndarray]:
        """
        Transition ball states ([pos_x, pos_y, vel_x, vel_y]) for one time step.
        """
        return [self._transition_one(s, delta) for s in states]
