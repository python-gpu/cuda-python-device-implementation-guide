Synchronization
---------------

Memory Orderings
~~~~~~~~~~~~~~~~~

A *memory order* specifies how atomic and non-atomic memory accesses are
ordered around a synchronization primitive. Memory orders are defined by
`Standard C++ (ISO/IEC
14882:2023) <https://timsong-cpp.github.io/cppwp/n4950/atomics.order>`__.

Functions that take a memory order have a parameter named ``memory``.

If the ``memory`` parameter is equal to one of the following, the
function shall have the behavior of the corresponding Standard C++
memory order:

======================= =============================
Python Parameter Equals Standard C++ Thread Scope
======================= =============================
``'relaxed'``           ``std::memory_order_relaxed``
``'consume'``           ``std::memory_order_consume``
``'acquire'``           ``std::memory_order_acquire``
``'release'``           ``std::memory_order_release``
``'acq_rel'``           ``std::memory_order_acq_rel``
``'seq_cst'``           ``std::memory_order_seq_cst``
======================= =============================

An error shall occur if the ``memory`` parameter is not equal to one of
the above values. [User Requirement]

Thread Scopes
~~~~~~~~~~~~~

A *thread scope* specifies the kind of threads that can synchronize with
each other using a synchronization primitive. ThreadThreads scopes are
defined by
`libcu++ <https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes>`__.

Functions that take a thread scope have a parameter named ``scope``.

If the ``scope`` parameter is equal to one of the following, the
function shall have the behavior of the corresponding CUDA C++ thread
scope:

======================= =============================
Python Parameter Equals CUDA C++ Thread Scope
======================= =============================
``'system'``            ``cuda::thread_scope_system``
``'device'``            ``cuda::thread_scope_device``
``'block'``             ``cuda::thread_scope_block``
``'thread'``            ``cuda::thread_scope_thread``
======================= =============================

An error shall occur if the ``scope`` parameter is not equal to one of
the above values. [User Requirement]

Atomics
~~~~~~~

``AtomicInterface`` Class
^^^^^^^^^^^^^^^^^^^^^^^^^

A class that subclasses ``AtomicInterface`` represents a scalar object
that can be accessed atomically. ``AtomicInterface`` types are usable in
device code.

Frameworks may define ``AtomicInterface`` as device only.

In this section, *atomic's object* means the scalar object that an
``AtomicInterface`` refers to.

``Atomic`` Class
^^^^^^^^^^^^^^^^

A class that owns a scalar object that is accessed atomically.

``Atomic.__init__(self, dtype)`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creates an ``Atomic`` object containing an object of ``dtype``.

The size of the machine representation of the dtype of the atomic's
object shall be less than or equal to 16 bytes. [User Requirement]

``Atomic.dtype`` Attribute
^^^^^^^^^^^^^^^^^^^^^^^^^^

The data type of the atomic's object.

``atomic_ref(array, index) -> AtomicInterface`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns an object that implements the ``AtomicInterface`` and represents
the object at ``array[index]``.

Users shall preserve the lifetime of ``array`` for as long as the
returned ``AtomicInterface`` may be used. [User Requirement]

``AtomicInterface.dtype`` Attribute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The dtype of the atomic's object.

``AtomicInterface.load(self, memory='seq_cst', scope='system')`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atomically returns the value of the atomic's object.

The size of the machine representation of the dtype of the atomic's
object shall be less than or equal to 16 bytes. [User Requirement]

``AtomicInterface.store(self, val, memory='seq_cst', scope='system')`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atomically sets the value of the atomic's object to ``val``.

The size of the machine representation of the dtype of the atomic's
object shall be less than or equal to 16 bytes. [User Requirement]

``AtomicInterface.exch(self, val, memory='seq_cst', scope='system')`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atomically sets the value of the atomic's object to ``val``.

Returns the value of the atomic's object before this operation.

The size of the machine representation of the dtype of the atomic's
object shall be less than or equal to 8 bytes. [User Requirement]

``AtomicInterface.cas(self, old, val, memory='seq_cst', scope='system')`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atomically performs ``if (this == old) this = val``, where ``this`` is
the value of the object.

Returns the value of the atomic's object before this operation.

The size of the machine representation of the dtype of the atomic's
object shall be less than or equal to 8 bytes. [User Requirement]

``AtomicInterface.add(self, val, memory='seq_cst', scope='system')`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atomically performs ``this += val``, where ``this`` is the value of the
object.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be ``uint32``, ``int32``,
``uint64``, ``int64``, ``float32``, or ``float64``. [User Requirement]

``AtomicInterface.sub(self, val, memory='seq_cst', scope='system')`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atomically performs ``this -= val``, where ``this`` is the value of the
object.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be ``uint32``, ``int32``,
``uint64``, ``int64``, ``float32``, or ``float64``. [User Requirement]

``AtomicInterface.and_(self, val, memory='seq_cst', scope='system')`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atomically performs ``this &= val``, where ``this`` is the value of the
object.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be ``uint32``, ``int32``,
``uint64``, or ``int64``. [User Requirement]

``AtomicInterface.or_(self, val, memory='seq_cst', scope='system')`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atomically performs ``this |= val``, where ``this`` is the value of the
object.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be ``uint32``, ``int32``,
``uint64``, or ``int64``. [User Requirement]

``AtomicInterface.xor(self, val, memory='seq_cst', scope='system')`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atomically performs ``this ^= val``, where ``this`` is the value of the
object.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be ``uint32``, ``int32``,
``uint64``, or ``int64``. [User Requirement]

``AtomicInterface.max(self, val, memory='seq_cst', scope='system')`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atomically perform ``this = max(this, val)``, where ``this`` is the
value of the object.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be ``uint32``, ``int32``,
``uint64``, ``int64``, ``float32``, or ``float64``. [User Requirement]

``AtomicInterface.nanmax(self, val, memory='seq_cst', scope='system')`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atomically perform ``this = max(this, val)``, where ``this`` is the
value of the object.

NaN is treated as a missing value. Example:
``assert(AtomicInterface.nanmax(a, NaN) == a)``. Example:
``a = NaN; assert(AtomicInterface.nanmax(a, n) == n)``.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be ``uint32``, ``int32``,
``uint64``, ``int64``, ``float32``, or ``float64``. [User Requirement]

``AtomicInterface.min(self, val, memory='seq_cst', scope='system')`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atomically perform ``this = min(this, val)``, where ``this`` is the
value of the object.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be ``uint32``, ``int32``,
``uint64``, ``int64``, ``float32``, or ``float64``. [User Requirement]

``AtomicInterface.nanmin(self, val, memory='seq_cst', scope='system')`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atomically perform ``this = min(this, val)``, where ``this`` is the
value of the object.

NaN is treated as a missing value. Example:
``assert(AtomicInterface.nanmin(a, NaN) == a)``. Example:
``a = NaN; assert(AtomicInterface.nanmin(a, n) == n)``.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be ``uint32``, ``int32``,
``uint64``, ``int64``, ``float32``, or ``float64``. [User Requirement]

``threadfence(memory='seq_cst', scope='system')`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Establishes the specified memory synchronization ordering of non-atomic
and relaxed accesses.

Note: This is the equivalent of CUDA C++'s ``__threadfence``,
``__threadfence_block``, and ``__threadfence_system``.

Thread Block Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following functions shall be called by every thread in the thread
block. [User Requirement]

``predicate`` parameters shall be a function that is callable with no
arguments. [User Requirement]

``syncthreads()`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^

Synchronizes with all threads in the same thread block.

``syncthreads_count(pred: Callable[[], bool]) -> int32`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Synchronizes with all threads in the same thread block.

Returns the number of threads in the thread block for which
``predicate()`` == ``True``.

``syncthreads_and(pred: Callable[[], bool]) -> bool`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Synchronizes with all threads in the same thread block.

Returns ``True`` if ``predicate() == True`` for all threads in the
thread block.

``syncthreads_or(pred: Callable[[], bool]) -> bool`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Synchronizes with all threads in the same thread block.

Returns ``True`` if ``predicate() == True`` for any thread in the thread
block.

Thread Warp Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``predicate`` parameters shall be a function that is callable with no
arguments. [User Requirement]

``WarpMask`` Class
^^^^^^^^^^^^^^^^^^

A ``WarpMask`` object indicates which of the threads in a thread warp
are participating in an operation.

``WarpMask`` is a subclass of ``int32``.

``WarpMask.__getitem__(self, i: int32) -> bool`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns ``True`` if thread ID ``i`` is set in the mask.

``i >= 0 and i < 32`` [User Requirement]

``WarpMask.__setitem__(self, i: int32, val: bool)`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``val == True``, add thread ID ``i`` to the mask, otherwise, remove
thread ID ``i`` from the mask.

``i >= 0 and i < 32`` [User Requirement]

``activemask() -> WarpMask`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a mask of all the currently active threads in the calling warp.

``lanemask_lt() -> WarpMask`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a mask of all threads (including inactive ones) with thread IDs
less than the current lane (``lane_id``).

``syncwarp(mask: WarpMask)`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Synchronizes with all threads in ``mask`` within the thread warp.

``all_sync(mask: WarpMask, pred: Callable[[], bool]) -> bool`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns ``True`` if ``pred() == True`` for all threads in ``mask``
within the thread warp, and ``False`` otherwise.

Note: This operation does not guarantee any memory ordering.

``any_sync(mask: WarpMask, pred: Callable[[], bool]) -> bool`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns ``True`` if ``pred() == True`` for any threads in ``mask``
within the thread warp, and ``False`` otherwise.

Note: This operation does not guarantee any memory ordering.

``eq_sync(mask: WarpMask, pred: Callable[[], bool]) -> bool`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns ``True`` if ``pred()`` has the same value for all threads in
``mask`` within the thread warp, and ``False`` otherwise.

Note: This operation does not guarantee any memory ordering.

``ballot_sync(mask: WarpMask, pred : function) -> WarpMask`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns all threads in ``mask`` within the warp for which
``pred() == True``.

Note: This operation does not guarantee any memory ordering.

``shfl_sync(mask: WarpMask, value, src_lane: int32)`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns ``value`` from thread ``src_lane``.

``mask[src_lane] == True`` [User Requirement]

``src_lane >= 0 and src_lane < 32`` [User Requirement]

The size of the machine representation of ``value`` shall be less than
or equal to 8 bytes. [User Requirement]

Note: This operation does not guarantee any memory ordering.

``shfl_up_sync(mask: WarpMask, value, delta: int32)`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns ``value`` from thread ``lane_id - delta``.

``mask[lane_id - delta] == True`` [User Requirement]

``src_lane >= 0 and src_lane < 32`` [User Requirement]

The size of the machine representation of ``value`` shall be less than
or equal to 8 bytes. [User Requirement]

Note: This operation does not guarantee any memory ordering.

``shfl_down_sync(mask: WarpMask, value, delta: int32)`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns ``value`` from thread ``lane_id + delta``.

``mask[lane_id + delta] == True`` [User Requirement]

``src_lane >= 0 and src_lane < 32`` [User Requirement]

The size of the machine representation of ``value`` shall be less than
or equal to 8 bytes. [User Requirement]

Note: This operation does not guarantee any memory ordering.

``shfl_xor_sync(mask: WarpMask, value, flag: WarpMask)`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns ``value`` from thread ``lane_id ^ flag``.

``mask[lane_id ^ flag] == True`` [User Requirement]

``src_lane >= 0 and src_lane < 32`` [User Requirement]

The size of the machine representation of ``value`` shall be less than
or equal to 8 bytes. [User Requirement]

Note: This operation does not guarantee any memory ordering.

``match_any_sync(mask: WarpMask, value, flag: int32) -> WarpMask`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns all threads in ``mask`` within the warp with the same ``value``
as the caller.

Note: This operation does not guarantee any memory ordering.

``match_all_sync(mask: WarpMask, value, flag: int32) -> tuple[WarpMask, bool]`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a tuple of ``(eq, pred)``, where ``eq`` is a mask of threads in
``mask`` that have the same value, and ``pred`` is ``True`` if all
threads in the ``mask`` have the same value, and False otherwise.

Note: This operation does not guarantee any memory ordering.

Numeric Intrinsics
~~~~~~~~~~~~~~~~~~

``popc(x)`` Function
^^^^^^^^^^^^^^^^^^^^

Returns the number of bits set in ``x``.

``x``'s type must be a heterogeneous integer type. [User Requirement]

``brev(x)`` Function
^^^^^^^^^^^^^^^^^^^^

Returns the reverse of the bit pattern of ``x``.

``x``'s type must be a heterogeneous integer type. [User Requirement]

``clz(x)`` Function
^^^^^^^^^^^^^^^^^^^

Returns the number of leading zeros in ``x``.

``x``'s type must be a heterogeneous integer type. [User Requirement]

``ffs(x)`` Function
^^^^^^^^^^^^^^^^^^^

Returns the position of the first (least significant) bit set in ``x``,
where the least significant bit position is 1.

``x``'s type must be a heterogeneous integer type. [User Requirement]

``cbrt(a)`` Function
^^^^^^^^^^^^^^^^^^^^

Returns ``a ** (1/3)``.

``a``'s type must be a heterogeneous floating point type. [User
Requirement]

``fma(a, b, c)`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^

Returns ``(a * b) + c``.

``a``, ``b``, and ``c``'s types must be heterogeneous floating point
types. [User Requirement] 