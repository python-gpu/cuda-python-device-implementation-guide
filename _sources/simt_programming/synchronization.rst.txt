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

.. currentmodule:: api.atomic_interface

.. autoclass:: AtomicInterface
   :members:
   :undoc-members:
   :special-members: __init__

.. currentmodule:: api.atomic

.. autoclass:: Atomic
   :members:
   :undoc-members:
   :special-members: __init__

.. currentmodule:: api.atomic_ref

.. autoclass:: AtomicRef
   :members:
   :undoc-members:
   :special-members: __init__

.. currentmodule:: api.atomic_interface

.. autofunction:: threadfence

Thread Block Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following functions shall be called by every thread in the thread
block. [User Requirement]

``predicate`` parameters shall be a function that is callable with no
arguments. [User Requirement]

.. currentmodule:: api.synchronization

.. autofunction:: syncthreads

.. autofunction:: syncthreads_count

.. autofunction:: syncthreads_and

.. autofunction:: syncthreads_or

Thread Warp Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``predicate`` parameters shall be a function that is callable with no
arguments. [User Requirement]

.. autoclass:: WarpMask
   :members:
   :undoc-members:
   :special-members: __getitem__, __setitem__

.. autofunction:: activemask

.. autofunction:: lanemask_lt

.. autofunction:: syncwarp

.. autofunction:: all_sync

.. autofunction:: any_sync

.. autofunction:: eq_sync

.. autofunction:: ballot_sync

.. autofunction:: shfl_sync

.. autofunction:: shfl_up_sync

.. autofunction:: shfl_down_sync

.. autofunction:: shfl_xor_sync

.. autofunction:: match_any_sync

.. autofunction:: match_all_sync

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