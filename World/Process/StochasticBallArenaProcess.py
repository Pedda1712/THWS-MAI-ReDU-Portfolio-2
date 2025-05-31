import numpy as np
from .BallArenaProcess import BallArenaProcess

class StochasticBallArenaProcess(BallArenaProcess):
    internal_process: BallArenaProcess
    vel_variance: np.ndarray

    def __init__(self, ball_arena_process: BallArenaProcess, velocity_variance: np.ndarray):
        """
        Like a BallArenaProcess, but noise is added to the velocity before each transition step.

        This is intended as a state transition model in a particle filter. The velocity noise is supposed to model uncertainty in the transition process.

        Parameters:
          ball_arena_process: The underlying deterministic ball arena process
          velocity_variance: variance of x velocity and y velocity
        """
        if velocity_variance.shape != (2,):
            raise RuntimeError(f"velocity variance is supposed to have shape (2,), got {velocity_variance.shape}")
        self.internal_process = ball_arena_process
        self.vel_variance = velocity_variance

    def transition(self, states: list[np.ndarray], delta: float = 1, seed: int = 0):
        # pre-modify velocities by normal dist
        velocities = np.array(states)[:,2:]
        
        rng = np.random.default_rng(seed = seed)

        incs = rng.multivariate_normal(np.zeros(2), np.diag(self.vel_variance), size = len(states))

        npstates = np.array(states)
        npstates[:,2:] = velocities + incs
        
        seed += len(states)

        return self.internal_process.transition(list(npstates), delta, seed)
