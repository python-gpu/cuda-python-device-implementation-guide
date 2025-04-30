from typing import TypeVar, Generic, Callable, Literal, Any, Tuple

# Type definitions
T = TypeVar('T')
MemoryOrder = Literal['relaxed', 'consume', 'acquire', 'release', 'acq_rel', 'seq_cst']
ThreadScope = Literal['system', 'device', 'block', 'thread']

class AtomicInterface(Generic[T]):
    """
    A class that represents a scalar object that can be accessed atomically.
    AtomicInterface types are usable in device code.
    """

    @property
    def dtype(self) -> Any:
        """The dtype of the atomic's object."""
        ...

    def load(self, memory: MemoryOrder = 'seq_cst', scope: ThreadScope = 'system') -> T:
        """
        Atomically returns the value of the atomic's object.

        The size of the machine representation of the dtype of the atomic's
        object shall be less than or equal to 16 bytes. [User Requirement]
        """
        ...

    def store(self, val: T, memory: MemoryOrder = 'seq_cst', scope: ThreadScope = 'system') -> None:
        """
        Atomically sets the value of the atomic's object to `val`.

        The size of the machine representation of the dtype of the atomic's
        object shall be less than or equal to 16 bytes. [User Requirement]
        """
        ...

    def exch(self, val: T, memory: MemoryOrder = 'seq_cst', scope: ThreadScope = 'system') -> T:
        """
        Atomically sets the value of the atomic's object to `val`.

        Returns the value of the atomic's object before this operation.

        The size of the machine representation of the dtype of the atomic's
        object shall be less than or equal to 8 bytes. [User Requirement]
        """
        ...

    def cas(self, old: T, val: T, memory: MemoryOrder = 'seq_cst', scope: ThreadScope = 'system') -> T:
        """
        Atomically performs `if (this == old) this = val`, where `this` is
        the value of the object.

        Returns the value of the atomic's object before this operation.

        The size of the machine representation of the dtype of the atomic's
        object shall be less than or equal to 8 bytes. [User Requirement]
        """
        ...

    def add(self, val: T, memory: MemoryOrder = 'seq_cst', scope: ThreadScope = 'system') -> T:
        """
        Atomically performs `this += val`, where `this` is the value of the
        object.

        Returns the value of the atomic's object before this operation.

        The dtype of the atomic's object shall be `uint32`, `int32`,
        `uint64`, `int64`, `float32`, or `float64`. [User Requirement]
        """
        ...

    def sub(self, val: T, memory: MemoryOrder = 'seq_cst', scope: ThreadScope = 'system') -> T:
        """
        Atomically performs `this -= val`, where `this` is the value of the
        object.

        Returns the value of the atomic's object before this operation.

        The dtype of the atomic's object shall be `uint32`, `int32`,
        `uint64`, `int64`, `float32`, or `float64`. [User Requirement]
        """
        ...

    def and_(self, val: T, memory: MemoryOrder = 'seq_cst', scope: ThreadScope = 'system') -> T:
        """
        Atomically performs `this &= val`, where `this` is the value of the
        object.

        Returns the value of the atomic's object before this operation.

        The dtype of the atomic's object shall be `uint32`, `int32`,
        `uint64`, or `int64`. [User Requirement]
        """
        ...

    def or_(self, val: T, memory: MemoryOrder = 'seq_cst', scope: ThreadScope = 'system') -> T:
        """
        Atomically performs `this |= val`, where `this` is the value of the
        object.

        Returns the value of the atomic's object before this operation.

        The dtype of the atomic's object shall be `uint32`, `int32`,
        `uint64`, or `int64`. [User Requirement]
        """
        ...

    def xor(self, val: T, memory: MemoryOrder = 'seq_cst', scope: ThreadScope = 'system') -> T:
        """
        Atomically performs `this ^= val`, where `this` is the value of the
        object.

        Returns the value of the atomic's object before this operation.

        The dtype of the atomic's object shall be `uint32`, `int32`,
        `uint64`, or `int64`. [User Requirement]
        """
        ...

    def max(self, val: T, memory: MemoryOrder = 'seq_cst', scope: ThreadScope = 'system') -> T:
        """
        Atomically perform `this = max(this, val)`, where `this` is the
        value of the object.

        Returns the value of the atomic's object before this operation.

        The dtype of the atomic's object shall be `uint32`, `int32`,
        `uint64`, `int64`, `float32`, or `float64`. [User Requirement]
        """
        ...

    def nanmax(self, val: T, memory: MemoryOrder = 'seq_cst', scope: ThreadScope = 'system') -> T:
        """
        Atomically perform `this = max(this, val)`, where `this` is the
        value of the object.

        NaN is treated as a missing value. Example:
        `assert(AtomicInterface.nanmax(a, NaN) == a)`. Example:
        `a = NaN; assert(AtomicInterface.nanmax(a, n) == n)`.

        Returns the value of the atomic's object before this operation.

        The dtype of the atomic's object shall be `uint32`, `int32`,
        `uint64`, `int64`, `float32`, or `float64`. [User Requirement]
        """
        ...

    def min(self, val: T, memory: MemoryOrder = 'seq_cst', scope: ThreadScope = 'system') -> T:
        """
        Atomically perform `this = min(this, val)`, where `this` is the
        value of the object.

        Returns the value of the atomic's object before this operation.

        The dtype of the atomic's object shall be `uint32`, `int32`,
        `uint64`, `int64`, `float32`, or `float64`. [User Requirement]
        """
        ...

    def nanmin(self, val: T, memory: MemoryOrder = 'seq_cst', scope: ThreadScope = 'system') -> T:
        """
        Atomically perform `this = min(this, val)`, where `this` is the
        value of the object.

        NaN is treated as a missing value. Example:
        `assert(AtomicInterface.nanmin(a, NaN) == a)`. Example:
        `a = NaN; assert(AtomicInterface.nanmin(a, n) == n)`.

        Returns the value of the atomic's object before this operation.

        The dtype of the atomic's object shall be `uint32`, `int32`,
        `uint64`, `int64`, `float32`, or `float64`. [User Requirement]
        """
        ...


def threadfence(memory: MemoryOrder = 'seq_cst', scope: ThreadScope = 'system') -> None:
    """
    Establishes the specified memory synchronization ordering of non-atomic
    and relaxed accesses.

    Note: This is the equivalent of CUDA C++'s `__threadfence`,
    `__threadfence_block`, and `__threadfence_system`.
    """
    ... 