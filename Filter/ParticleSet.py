"""
Particle Filter implementation.
"""
import numpy as np
from typing import TypeVar, Generic
from World.Initializer import BaseInitializer
from World.Process import IdentityProcess
from .Observation import BaseObservationModel

S = TypeVar('S')
O = TypeVar('O')

class ParticleSet(Generic[S, O]):
    particles: list[S]
    weights: list[float]
    process: IdentityProcess
    deterministic_process: IdentityProcess
    observation_model: BaseObservationModel
    seed: int
    N: int

    def __init__(self, N: int, initializer: BaseInitializer, process: IdentityProcess, deterministic_process: IdentityProcess, observation_model: BaseObservationModel, seed: int = 0):
        """
        Initialize the Particle Filter.

        Draw N from initializer to get initial particles.
        Weights are initialized uniformly.

        First step of condensation algorithm.

        Parameters:
          N: the number of particles
          initializer: how to initialize the particles
          process: state transition model (non-determinism inside)
          deterministic_process: state transition model to use in
           case of missing observations
          observation_model: particle weighting mechanism
          seed: seed used for RNG for reproducibility
        """
        self.N = N
        self.weights = [1/N] * N
        self.particles = [initializer.generate(n, seed) for n in range(N)]
        self.process = process
        self.deterministic_process = deterministic_process
        self.seed = seed
        self.observation_model = observation_model

    def resample(self):
        """
        Second step of condensation algorithm.
        """
        rng = np.random.default_rng(self.seed)
        counts = rng.multinomial(self.N, self.weights)
        
        new_set = []
        for (idx, c) in enumerate(counts):
            new_set.extend([self.particles[idx]] * c)
            
        self.weights = [1/self.N] * self.N
        self.particles = new_set
        
        self.seed += self.N

    def transition(self, delta: float = 1, deterministic: bool = False):
        """
        Third step of condensation algorithm.

        Parameters:
          delta: time step length
          deterministic: wether or not to do a deterministic particle transition
           (use in case of missing observation for time step)
        """
        if not deterministic:
          self.particles = self.process.transition(self.particles, delta, self.seed)
        else:
          self.particles = self.deterministic_process.transition(self.particles, delta, self.seed)
        self.seed += self.N

    def observe(self, observation: O):
        """
        Fourth step of condensation algorithm.
        """
        self.weights, modified_states = self.observation_model.observe(self.particles, observation, self.seed)

        self.seed += self.N * 2

        if modified_states:
            self.particles = modified_states
