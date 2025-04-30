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

.. autoclass:: api.atomic_interface.AtomicInterface
   :members:
   :undoc-members:
   :special-members: __init__

.. autoclass:: api.atomic.Atomic
   :members:
   :undoc-members:
   :special-members: __init__

.. autoclass:: api.atomic_ref.AtomicRef
   :members:
   :undoc-members:
   :special-members: __init__

.. autofunction:: api.atomic_interface.threadfence 

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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