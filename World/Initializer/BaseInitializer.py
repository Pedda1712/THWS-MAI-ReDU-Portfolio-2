"""
The Initializer will be called by the world initialization to
generate the initial states of the objects.
"""

from typing import Generic, TypeVar

S = TypeVar('S')

class BaseInitializer(Generic[S]):
    default_state: S
    
    def __init__(self, default_state: S):
        """
        The BaseInitializer just outputs a default state.

        Parameters:
          default_state: the state that will be output each time
        """
        self.default_state = default_state

    def generate(self, n: int, seed: int = 0) -> S:
        """
        Generate a new state.

        The world simulation will ask the initializer for the initial
        states of the N objects.

        Parameters:
          n: Object index
          seed: seed to use if the initializer uses random elements
        """
        return self.default_state
