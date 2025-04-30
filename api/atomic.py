from typing import TypeVar, Any
from .atomic_interface import AtomicInterface

T = TypeVar('T')

class Atomic(AtomicInterface[T]):
    """
    A class that owns a scalar object that is accessed atomically.
    """

    def __init__(self, dtype: Any):
        """
        Creates an `Atomic` object containing an object of `dtype`.

        The size of the machine representation of the dtype of the atomic's
        object shall be less than or equal to 16 bytes. [User Requirement]
        """
        ...