import numpy as np
from typing import Callable, Optional
from .BaseObservationModel import BaseObservationModel

import scipy

class MultiBallObservationModel(BaseObservationModel):
    variances: np.ndarray
    reinitialization_variances: np.ndarray

    def __init__(self, variances: np.ndarray, velocity_reinitialization_variances: np.ndarray):
        """
        Multi-Ball Observation model.

        Parameters:
          variances: variance in the statistical classifier (assume: normal distribution with mean at particle position)
            larger values will result in more equal weightings
            (how much uncertainty we assume to be in the observation process)
          velocity_reinitialization_variance:
            the observation model will try to 'fix' degenerate situations (orphan observations) by reinitializing some particles
            these particles will have their velocity initialized with this variance
        """
        if variances.shape != (2,):
            raise RuntimeError(f"observation variance must have shape (2,). got {variances.shape}")
        if velocity_reinitialization_variances.shape != (2,):
            raise RuntimeError(f"velocity reinitialization variance must have shape (2,). got {velocity_reinitialization_variances.shape}")
        self.velocity_reinitialization_variances = np.diag(velocity_reinitialization_variances)
        self.variances = np.diag(variances)
        self.icov = np.linalg.pinv(self.variances)
        self.det = np.linalg.det(self.variances)

    def observe(self, states: list[np.ndarray], observation: list[np.ndarray], seed: int) -> list[float]:
        """
        observations are a list of (x,y) tuples (we do NOT observe velocity)

        Our observation model:
        - each observation point is allowed to 'give out' 1/N weight (N = number of observations)
        - how this works:
           - center normal dist at each particle and fetch pdf value of observations
           - normalize the values corresponding to each observation
           - average these values for each particle

        Returns:
          weights: weights for each state
        """
        def weight_observation(p: np.ndarray, o: np.ndarray):
            return (1/(np.sqrt(2*3.14159 * self.det)))*np.exp(-0.5*(o - p).dot(self.icov).dot(o - p))

        weights = np.array([[weight_observation(p[:2], o) for p in states] for o in observation])
        for observation_index in range(len(observation)): # normalize each row by itself
            weights[observation_index, :] = weights[observation_index, :] / weights[observation_index, :].sum()

        weights = np.mean(weights, axis=0)
            
        return list(weights)
