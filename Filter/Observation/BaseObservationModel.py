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

    def observe(self, states: list[S], observation: O, seed: int) -> tuple[list[float], Optional[list[S]]]:
        """
        Take in states and observation and produce weight for each
        sample.

        Weights are guaranteed to add up to one.

        Our Observation model is also allowed to modify particle states,
        (used for escaping from degenerated states), so a new state may
        be returned
        """
        return [1/len(states)] * len(states)
