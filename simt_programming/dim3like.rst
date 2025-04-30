*``Dim3Like``* Object (Exposition Only)
---------------------------------------

.. code:: py

   Dim3Like = tuple[int32, int32, int32] \
        | tuple[int32, int32] \
        | tuple[int32] \
        | int32

A type hint for parameters describing the shape of a level in the thread
hierarchy.

``Dim3Like`` is exposition only. 