from typing import TypeVar, Any
from .atomic_interface import AtomicInterface

T = TypeVar('T')

class AtomicRef(AtomicInterface[T]):
    """
    A class that implements the `AtomicInterface` and represents
    the object at `array[index]`.

    Users shall preserve the lifetime of `array` for as long as the
    `AtomicRef` may be used. [User Requirement]
    """
    
    def __init__(self, array: Any, index: Any):
        """
        Creates an `AtomicRef` object representing the element at `array[index]`.
        
        Users shall preserve the lifetime of `array` for as long as the
        `AtomicRef` may be used. [User Requirement]
        """
        ...