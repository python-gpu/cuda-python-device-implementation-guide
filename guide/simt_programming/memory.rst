Local, Shared, and Const Memory
-------------------------------

All entities in this section are device only.

``local_array(shape: ConstExpr, dtype, order='C': {'C','F'}, align: uint32)`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creates an array of the given shape, NumPy data type, and alignment in
memory private to each thread.

``shape`` shall be an integer or tuple of integers. [User Requirement]

``shared_array(shape: ConstExpr, dtype, order='C': {'C','F'}, align: uint32)`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creates an array of the given shape, NumPy data type, and alignment in
memory shared across the thread block.

``shape`` shall be an integer or tuple of integers. [User Requirement]

``dynamic_shared_array()`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a 1D array of ``uint8`` that references dynamic shared memory.

The size of the array is the size of the ``shared`` parameter of the
current kernel launch. 