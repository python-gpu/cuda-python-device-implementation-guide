"""
CUDA Python Device API stubs.

This package contains stub files that document the CUDA Python Device API.
"""

from .atomic_interface import AtomicInterface, threadfence, MemoryOrder, ThreadScope
from .atomic import Atomic
from .atomic_ref import AtomicRef
from .synchronization import (
    WarpMask, 
    syncthreads, syncthreads_count, syncthreads_and, syncthreads_or,
    activemask, lanemask_lt, syncwarp,
    all_sync, any_sync, eq_sync, ballot_sync,
    shfl_sync, shfl_up_sync, shfl_down_sync, shfl_xor_sync,
    match_any_sync, match_all_sync
)

__all__ = [
    # Atomics
    'AtomicInterface',
    'Atomic',
    'AtomicRef',
    'threadfence',
    'MemoryOrder',
    'ThreadScope',
    
    # Thread Block Synchronization
    'syncthreads',
    'syncthreads_count',
    'syncthreads_and',
    'syncthreads_or',
    
    # Thread Warp Synchronization
    'WarpMask',
    'activemask',
    'lanemask_lt',
    'syncwarp',
    'all_sync',
    'any_sync',
    'eq_sync',
    'ballot_sync',
    'shfl_sync',
    'shfl_up_sync',
    'shfl_down_sync',
    'shfl_xor_sync',
    'match_any_sync',
    'match_all_sync',
] 