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
        """
        @line_profiler.profile
        def weight(s: np.ndarray):
            max_w = 0
            p = s[:2]
            for o in observation:
                # Assume equal importance of each observation
                
                w = (1/(np.sqrt(2*3.14159 * self.det)))*np.exp(-0.5*(o - p).dot(self.icov).dot(o - p))
                if w > max_w:
                    max_w = w
            return max_w

        unnormalized_weights = [weight(s) for s in states]
        sum_of_weights = sum(unnormalized_weights)
        return [w / sum_of_weights for w in unnormalized_weights]
