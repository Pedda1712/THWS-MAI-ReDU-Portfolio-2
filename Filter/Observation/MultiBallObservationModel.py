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

    def observe(self, states: list[np.ndarray], observation: list[np.ndarray], seed: int) -> tuple[list[float], Optional[list[np.ndarray]]]:
        """
        observations are a list of (x,y) tuples (we do NOT observe velocity)

        Our observation model:
        - each observation point is allowed to 'give out' 1/N weight (N = number of observations)
        - each observation point will distribute its weight to the particles
          for which it is the nearest observation
        => This ensures that the number of particles allocated for each
           ball stays roughly the same over time

        Returns:
          weights: weights for each state
          lonely_observations: observations to which no particle was assigned
        """
        def weight_observation(p: np.ndarray, o: np.ndarray):
            return (1/(np.sqrt(2*3.14159 * self.det)))*np.exp(-0.5*(o - p).dot(self.icov).dot(o - p))
            
            
        def weight(s: np.ndarray):
            max_w = 0
            norm_index = 0
            p = s[:2]
            for (o_idx, o) in enumerate(observation):
                # gauss centered at particle
                w = weight_observation(p, o)    
                if w > max_w:
                    max_w = w
                    norm_index = o_idx
            return (norm_index, max_w)

        unnormalized_weights = [weight(s) for s in states]
        just_weights = np.array([w[1] for w in unnormalized_weights])
        just_indices = np.array([w[0] for w in unnormalized_weights])
        # normalize the set of closest weights to observation to have weight 1/N in sum
        lonely = []
        for (o_idx, _) in enumerate(observation):
            sum_of_weights = np.sum(np.where(just_indices == o_idx, just_weights, 0)) * len(observation)
            if sum_of_weights == 0:
                lonely.append(o_idx)
            else:
                just_weights = np.where(just_indices == o_idx, just_weights / sum_of_weights, just_weights)

        if not lonely:
            return list(just_weights), None

        # remove observations which are nonetheless close to some particles
        # we only consider observations lonely if they are far away from all particles
        def is_really_lonely(o: np.ndarray):
            total_pdf = sum([weight_observation(s[:2],o) for s in states])
            return total_pdf < 1e-5

        lonely = [l for l in lonely if is_really_lonely(observation[l])]
        
        # degenerated state: some obervations do not have particles for which they are the closest
        # grab the N least weighted particles, and reinitialize them at the lonely observations

        batch_size = int((1 / len(observation)) * len(states))
        particles_to_redistribute = len(lonely) * batch_size

        idx = np.argpartition(just_weights, particles_to_redistribute)[:particles_to_redistribute]

        states = np.array(states)
        
        start = 0
        for lonely_index in lonely:
            o = observation[lonely_index]
            deported_particle_indices = idx[start:(start+batch_size)]
            N = len(deported_particle_indices)

            rng = np.random.default_rng(seed = seed)
            # distribute positions around observation
            positions = rng.multivariate_normal(o, self.variances, size=N)
            # distribute velocities normally
            velocities = rng.multivariate_normal((0,0), self.velocity_reinitialization_variances, size=N)
            new_states = np.concatenate((positions, velocities), axis=1)

            states[deported_particle_indices, :] = new_states
            just_weights[deported_particle_indices] = 1 / (batch_size * len(observation))
            start += batch_size

        return list(just_weights), list(states)
