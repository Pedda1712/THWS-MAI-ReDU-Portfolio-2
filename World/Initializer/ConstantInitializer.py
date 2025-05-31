from .BaseInitializer import BaseInitializer, S
from typing import Generic

class ConstantInitializer(Generic[S], BaseInitializer[S]):
    default_states: list[S]

    def __init__(self, default_states: list[S]):
        """
        The ConstantInitializer outputs a list
        of default states sequentially.

        Parameters:
          default_states: list of states to output
        """
        self.default_states = default_states

    def generate(self, n: int, seed: int = 0) -> S:
        """
        Output the nth default state. If n is larger than
        the number of states passed in the constructor,
        the default states array is viewed as a ring buffer.
        """
        return self.default_states[n % len(self.default_states)]
