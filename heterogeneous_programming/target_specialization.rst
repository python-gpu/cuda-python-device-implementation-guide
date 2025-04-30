Target Specialization
---------------------

+---------------------------------------+-----------------------------------+
| .. code-block:: py                    | .. code-block:: cu                |
|                                       |                                   |
|    if target(is_host):                |    #if !__CUDA_ARCH__             |
|      substatement                     |      substatement                 |
|                                       |    #endif                         |
+---------------------------------------+-----------------------------------+
| .. code-block:: py                    | .. code-block:: cu                |
|                                       |                                   |
|    if target(is_device):              |    #if __CUDA_ARCH__              |
|      substatement                     |      substatement                 |
|                                       |    #endif                         |
+---------------------------------------+-----------------------------------+
| .. code-block:: py                    | .. code-block:: cu                |
|                                       |                                   |
|    if target(is_exactly(smXX)         |    #if  __CUDA_ARCH__ == XX0 \    |
|            | is_exactly(smYY)         |      || __CUDA_ARCH__ == YY0 \    |
|            | is_exactly(smZZ)):       |      || __CUDA_ARCH__ == ZZ0      |
|      substatement                     |      substatement                 |
|                                       |    #endif                         |
+---------------------------------------+-----------------------------------+
| .. code-block:: py                    | .. code-block:: cu                |
|                                       |                                   |
|    if target(not provides(smXX)):     |    #if __CUDA_ARCH__ < XX0        |
|      substatement                     |      substatement                 |
|                                       |    #endif                         |
+---------------------------------------+-----------------------------------+
| .. code-block:: py                    | .. code-block:: cu                |
|                                       |                                   |
|    if target(not provides(smXX)       |    #if __CUDA_ARCH__ <= XX0       |
|            | is_exactly(smXX)):       |      substatement                 |
|      substatement                     |    #endif                         |
+---------------------------------------+-----------------------------------+
| .. code-block:: py                    | .. code-block:: cu                |
|                                       |                                   |
|    if target(provides(smXX)           |    #if __CUDA_ARCH__ > XX0        |
|            & ~is_exactly(smXX)):      |      substatement                 |
|      substatement                     |    #endif                         |
+---------------------------------------+-----------------------------------+
| .. code-block:: py                    | .. code-block:: cu                |
|                                       |                                   |
|    if target(provides(smXX)):         |    #if __CUDA_ARCH__ >= XX0       |
|      substatement                     |      substatement                 |
|                                       |    #endif                         |
+---------------------------------------+-----------------------------------+

*Target specialization* enables the conditional use of constructs that
are only usable in a certain execution space without constraining the
calling function's usability in different execution spaces.

If the condition of an ``if`` statement:

-  is a constant expression,
-  contains a call to ``target``, and
-  evaluates to ``False``

then the use of any constructs that require a certain execution space
shall not require that execution space.

``target(desc: ConstExpr[TargetDescription]) -> bool`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns ``True`` if the current target matches ``desc``.

``TargetDescription`` Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ``TargetDescription`` object describes an execution space.

``TargetDescription.__or__(self, desc: ConstExpr[TargetDescription]) -> TargetDescription`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a ``TargetDescription`` that matches all the targets in either
``self`` and ``desc``.

``TargetDescription.__and__(self, desc: ConstExpr[TargetDescription]) -> TargetDescription`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a ``TargetDescription`` that matches all the targets in both
``self`` and ``desc``.

``TargetDescription.__invert__(self) -> TargetDescription`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a ``TargetDescription`` that matches all the targets that are
not in ``self``.

``SMSelector`` Class
^^^^^^^^^^^^^^^^^^^^

An ``SMSelector`` represents a particular CUDA architecture.

``smXX: SMSelector`` Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each supported CUDA architecture *XX*, a ``SMSelector`` ``smXX``
shall be defined.

Example: ``sm80`` would be defined for Ampere.

``is_exactly(sm: ConstExpr[SMSelector]) -> TargetDescription`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a ``TargetDescription`` that matches only the target ``sm``.

``provides(sm: ConstExpr[SMSelector]) -> TargetDescription`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a ``TargetDescription`` that matches all targets that support
the capabilities of target ``sm``.

Example: ``provides(sm70)`` is a ``TargetDescription`` that includes
``sm70``, ``sm72``, ``sm75``, ``sm80``, and all newer CUDA
architectures.

``is_host: TargetDescription`` Object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ``TargetDescription`` that matches all host targets.

``is_device: TargetDescription`` Object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ``TargetDescription`` that matches all devices targets.

``any_target: TargetDescription`` Object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ``TargetDescription`` that always matches.

``no_target: TargetDescription`` Object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ``TargetDescription`` that never matches. 