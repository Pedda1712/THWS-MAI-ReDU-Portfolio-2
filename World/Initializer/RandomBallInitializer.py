from .BaseInitializer import BaseInitializer

import numpy as np

class RandomBallInitializer(BaseInitializer[np.ndarray]):
    mean: np.ndarray
    covs: np.ndarray

    def __init__(self, mean: np.ndarray, covs: np.ndarray):
        """
        Initialize Ball States ([pos_x, pos_y, vel_x, vel_y])
        according to multivariate gaussian distribution.

        Parameters:
          mean: mean vector of length 4
          covs: covariance matrix of shape 4x4
        """
        if mean.shape != (4,):
            raise RuntimeError(f"mean ball state vector must be of shape (4,), got {mean.shape}")
        if covs.shape != (4, 4):
            raise RuntimeError("ball state covariance matrix must be of shape (4,4)")
        self.mean = mean
        self.covs = covs

    def generate(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed = seed + n)
        return rng.multivariate_normal(self.mean, self.covs)
        

