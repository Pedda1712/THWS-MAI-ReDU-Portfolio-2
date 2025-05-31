import numpy as np

class MultiBallSensor:
    variance: np.ndarray

    def __init__(self, variances: np.ndarray, seed: int = 0):
        """
        This simulates a noisy sensor by adding normally
        distributed values to true observations.

        Parameters:
          variances of ball positions picked up by this sensor
        """
        if variances.shape != (2,2):
            raise RuntimeError(f"positional variance must be shape (2,2) got {variances.shape}")
        self.variance = variances
        self.rng = np.random.default_rng(seed = seed)

    def sense(self, states: list[np.ndarray]) -> list[np.ndarray]:
        """
        Sense some states.

        Parameters:
          states: list of ball states (with velocity)

        Return:
          list of sensed POSITIONS (no velocities here!!)
        """
        positions = np.array(states)[:,:2]
        
        return list(positions + self.rng.multivariate_normal(np.zeros(2), self.variance, size = len(states)))
