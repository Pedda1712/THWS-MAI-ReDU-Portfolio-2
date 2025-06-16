"""
The Observation Model is responsible for weighting a
particles likelihood given an observation.
"""

from typing import TypeVar, Generic, Optional

S = TypeVar('S')
O = TypeVar('O')

class BaseObservationModel(Generic[S,O]):

    def __init__(self):
        pass

    def observe(self, states: list[S], observation: O, seed: int) -> list[float]:
        """
        Take in states and observation and produce weight for each
        sample.

        Weights are guaranteed to add up to one.
        """
        return [1/len(states)] * len(states)
