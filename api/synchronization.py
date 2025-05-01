from typing import Callable, Tuple, TypeVar

T = TypeVar('T')

# Thread Block Synchronization
def syncthreads() -> None:
    """
    Synchronizes with all threads in the same thread block.
    
    This function shall be called by every thread in the thread block. [User Requirement]
    """
    ...

def syncthreads_count(pred: Callable[[], bool]) -> int:
    """
    Synchronizes with all threads in the same thread block.
    
    Returns the number of threads in the thread block for which
    ``predicate()`` == ``True``.
    
    This function shall be called by every thread in the thread block. [User Requirement]
    
    ``predicate`` parameters shall be a function that is callable with no
    arguments. [User Requirement]
    """
    ...

def syncthreads_and(pred: Callable[[], bool]) -> bool:
    """
    Synchronizes with all threads in the same thread block.
    
    Returns ``True`` if ``predicate() == True`` for all threads in the
    thread block.
    
    This function shall be called by every thread in the thread block. [User Requirement]
    
    ``predicate`` parameters shall be a function that is callable with no
    arguments. [User Requirement]
    """
    ...

def syncthreads_or(pred: Callable[[], bool]) -> bool:
    """
    Synchronizes with all threads in the same thread block.
    
    Returns ``True`` if ``predicate() == True`` for any thread in the thread
    block.
    
    This function shall be called by every thread in the thread block. [User Requirement]
    
    ``predicate`` parameters shall be a function that is callable with no
    arguments. [User Requirement]
    """
    ...

# Thread Warp Synchronization
class WarpMask(int):
    """
    A ``WarpMask`` object indicates which of the threads in a thread warp
    are participating in an operation.
    
    ``WarpMask`` is a subclass of ``int32``.
    """
    
    def __getitem__(self, i: int) -> bool:
        """
        Returns ``True`` if thread ID ``i`` is set in the mask.
        
        ``i >= 0 and i < 32`` [User Requirement]
        """
        ...
    
    def __setitem__(self, i: int, val: bool) -> None:
        """
        If ``val == True``, add thread ID ``i`` to the mask, otherwise, remove
        thread ID ``i`` from the mask.
        
        ``i >= 0 and i < 32`` [User Requirement]
        """
        ...

def activemask() -> WarpMask:
    """
    Returns a mask of all the currently active threads in the calling warp.
    """
    ...

def lanemask_lt() -> WarpMask:
    """
    Returns a mask of all threads (including inactive ones) with thread IDs
    less than the current lane (``lane_id``).
    """
    ...

def syncwarp(mask: WarpMask) -> None:
    """
    Synchronizes with all threads in ``mask`` within the thread warp.
    """
    ...

def all_sync(mask: WarpMask, pred: Callable[[], bool]) -> bool:
    """
    Returns ``True`` if ``pred() == True`` for all threads in ``mask``
    within the thread warp, and ``False`` otherwise.
    
    Note: This operation does not guarantee any memory ordering.
    
    ``predicate`` parameters shall be a function that is callable with no
    arguments. [User Requirement]
    """
    ...

def any_sync(mask: WarpMask, pred: Callable[[], bool]) -> bool:
    """
    Returns ``True`` if ``pred() == True`` for any threads in ``mask``
    within the thread warp, and ``False`` otherwise.
    
    Note: This operation does not guarantee any memory ordering.
    
    ``predicate`` parameters shall be a function that is callable with no
    arguments. [User Requirement]
    """
    ...

def eq_sync(mask: WarpMask, pred: Callable[[], bool]) -> bool:
    """
    Returns ``True`` if ``pred()`` has the same value for all threads in
    ``mask`` within the thread warp, and ``False`` otherwise.
    
    Note: This operation does not guarantee any memory ordering.
    
    ``predicate`` parameters shall be a function that is callable with no
    arguments. [User Requirement]
    """
    ...

def ballot_sync(mask: WarpMask, pred: Callable[[], bool]) -> WarpMask:
    """
    Returns all threads in ``mask`` within the warp for which
    ``pred() == True``.
    
    Note: This operation does not guarantee any memory ordering.
    
    ``predicate`` parameters shall be a function that is callable with no
    arguments. [User Requirement]
    """
    ...

def shfl_sync(mask: WarpMask, value: T, src_lane: int) -> T:
    """
    Returns ``value`` from thread ``src_lane``.
    
    ``mask[src_lane] == True`` [User Requirement]
    
    ``src_lane >= 0 and src_lane < 32`` [User Requirement]
    
    The size of the machine representation of ``value`` shall be less than
    or equal to 8 bytes. [User Requirement]
    
    Note: This operation does not guarantee any memory ordering.
    """
    ...

def shfl_up_sync(mask: WarpMask, value: T, delta: int) -> T:
    """
    Returns ``value`` from thread ``lane_id - delta``.
    
    ``mask[lane_id - delta] == True`` [User Requirement]
    
    ``src_lane >= 0 and src_lane < 32`` [User Requirement]
    
    The size of the machine representation of ``value`` shall be less than
    or equal to 8 bytes. [User Requirement]
    
    Note: This operation does not guarantee any memory ordering.
    """
    ...

def shfl_down_sync(mask: WarpMask, value: T, delta: int) -> T:
    """
    Returns ``value`` from thread ``lane_id + delta``.
    
    ``mask[lane_id + delta] == True`` [User Requirement]
    
    ``src_lane >= 0 and src_lane < 32`` [User Requirement]
    
    The size of the machine representation of ``value`` shall be less than
    or equal to 8 bytes. [User Requirement]
    
    Note: This operation does not guarantee any memory ordering.
    """
    ...

def shfl_xor_sync(mask: WarpMask, value: T, flag: int) -> T:
    """
    Returns ``value`` from thread ``lane_id ^ flag``.
    
    ``mask[lane_id ^ flag] == True`` [User Requirement]
    
    ``src_lane >= 0 and src_lane < 32`` [User Requirement]
    
    The size of the machine representation of ``value`` shall be less than
    or equal to 8 bytes. [User Requirement]
    
    Note: This operation does not guarantee any memory ordering.
    """
    ...

def match_any_sync(mask: WarpMask, value: T, flag: int) -> WarpMask:
    """
    Returns all threads in ``mask`` within the warp with the same ``value``
    as the caller.
    
    Note: This operation does not guarantee any memory ordering.
    """
    ...

def match_all_sync(mask: WarpMask, value: T, flag: int) -> Tuple[WarpMask, bool]:
    """
    Returns a tuple of ``(eq, pred)``, where ``eq`` is a mask of threads in
    ``mask`` that have the same value, and ``pred`` is ``True`` if all
    threads in the ``mask`` have the same value, and False otherwise.
    
    Note: This operation does not guarantee any memory ordering.
    """
    ... 