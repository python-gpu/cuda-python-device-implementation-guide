Thread Positioning
------------------

All entities in this section are device only.

``Dim3`` Class

``Dim3`` is a subclass of ``uint32x3``.

``thread_idx: Dim3`` Object
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The thread indices' in the current thread block.

Each index ``x``, ``y``, and ``z`` shall be greater than or equal to 0
and less than ``block_dim.x``, ``block_dim.y``, and ``block_dim.z``
respectively.

``tid(ndims): Shape`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The absolute position of the thread in the grid.

Note: Returns a single integer if ndims is 1, and a tuple of integers
otherwise.

Equivalent to:

.. code:: py

   def tid(ndims):
     pos = tuple(t + bi * bd for t, bi, bd in zip(thread_idx, block_idx, block_dim))
     return pos[0] if ndims == 1 else pos[0:ndims]

``ndims > 0 and ndims <= 3`` [User Requirement]

``block_idx: Dim3`` Object
^^^^^^^^^^^^^^^^^^^^^^^^^^

The block indices in the grid of thread blocks.

Each index ``x``, ``y``, and ``z`` shall be greater than or equal to 0
and less than ``grid_dim.x``, ``grid_dim.y``, and ``grid_dim.z``
respectively.

``block_dim: Dim3`` Object
^^^^^^^^^^^^^^^^^^^^^^^^^^

The shape of each block of threads.

``grid_dim: Dim3`` Object
^^^^^^^^^^^^^^^^^^^^^^^^^

The shape of the grid of blocks.

``grid_size(ndims): Shape`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The absolute size of the grid.

Note: Returns a single integer if ndims is 1, and a tuple of integers
otherwise.

Equivalent to:

.. code:: py

   def grid_size(ndims):
     size = tuple(b * g for b, g in zip(block_dim, thread_dim))
     return size[0] if ndims == 1 else size[0:ndims]

``ndims > 0 and ndims <= 3`` [User Requirement]

``warp_size: int32`` Object
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The number of threads per warp.

``lane_id: int32`` Object
^^^^^^^^^^^^^^^^^^^^^^^^^

The thread index in the current warp. It is greater than or equal to
``0`` and less than ``warp_size``. 