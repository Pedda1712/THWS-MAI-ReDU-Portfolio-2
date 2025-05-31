import numpy as np
from typing import Callable
from .BaseObservationModel import BaseObservationModel

import scipy
import line_profiler

class MultiBallObservationModel(BaseObservationModel):
    variances: np.ndarray

    def __init__(self, variances: np.ndarray):
        """
        Multi-Ball Observation model.

        Parameters:
          variances: variance in the statistical classifier (assume: normal distribution with mean at particle position)
            larger values will result in more equal weightings
            (how much uncertainty we assume to be in the observation process)
        """
        if variances.shape != (2,):
            raise RuntimeError(f"observation variance must have shape (2,). got {variances.shape}")
        self.variances = np.diag(variances)
        self.icov = np.linalg.pinv(self.variances)
        self.det = np.linalg.det(self.variances)

    def observe(self, states: list[np.ndarray], observation: list[np.ndarray]) -> list[float]:
        """
        observations are a list of (x,y) tuples (we do NOT observe velocity)

        Our observation model:
        - each observation point is allowed to 'give out' 1/N weight (N = number of observations)
        - each observation point will distribute its weight to the particles
          for which it is the nearest observation
        => This ensures that the number of particles allocated for each
           ball stays roughly the same over time 
        """
        @line_profiler.profile
        def weight(s: np.ndarray):
            max_w = 0
            norm_index = 0
            p = s[:2]
            for (o_idx, o) in enumerate(observation):
                # gauss centered at particle
                w = (1/(np.sqrt(2*3.14159 * self.det)))*np.exp(-0.5*(o - p).dot(self.icov).dot(o - p))
                if w > max_w:
                    max_w = w
                    norm_index = o_idx
            return (norm_index, max_w)

        unnormalized_weights = [weight(s) for s in states]
        just_weights = np.array([w[1] for w in unnormalized_weights])
        just_indices = np.array([w[0] for w in unnormalized_weights])
        # normalize the set of closest weights to observation to have weight 1/N in sum
        for (o_idx, _) in enumerate(observation):
            sum_of_weights = np.sum(np.where(just_indices == o_idx, just_weights, 0)) * len(observation)
            just_weights = np.where(just_indices == o_idx, just_weights / sum_of_weights, just_weights)
        return list(just_weights)
