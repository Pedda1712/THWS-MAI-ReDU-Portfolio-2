"""
The World's 'Process' is the state transition function.
This is where the state is updated.
"""
from typing import Generic, TypeVar

S = TypeVar('S')

class IdentityProcess(Generic[S]):

    def __init__(self):
        """
        The Identity Process does nothing to each state.
        It is only used as a base class.
        """
        pass

    def transition(self, states: list[S], delta: float = 1, seed: int = 0) -> list[S]:
        """
        Takes in a list of states and a time delta and produces
        a list of the transitioned states.
        """
        return states
