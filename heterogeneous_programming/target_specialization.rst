Target Specialization
---------------------

+-----------------------------------+-----------------------------------+
| ``if                              | ``#if !__CU                       |
| target(is_host):   substatement`` | DA_ARCH__   substatement #endif`` |
+===================================+===================================+
| ``if ta                           | ``#if __CU                        |
| rget(is_device):   substatement`` | DA_ARCH__   substatement #endif`` |
+-----------------------------------+-----------------------------------+
| ``if t                            | ``#if                             |
| arget(is_exactly(smXX) \          |  __CUDA_ARCH__ == XX0 \  || __CUD |
| | is_exactly(smYY) \         | is | A_ARCH__ == YY0 \  || __CUDA_ARCH |
| _exactly(smZZ)):   substatement`` | __ == ZZ0   substatement #endif`` |
+-----------------------------------+-----------------------------------+
| ``if target(!                     | ``#if __CUDA_ARC                  |
| provides(smXX)):   substatement`` | H__ < XX0   substatement #endif`` |
+-----------------------------------+-----------------------------------+
| ``if target(!provides(smXX) | is  | ``#if __CUDA_ARCH                 |
| _exactly(smXX)):   substatement`` | __ <= XX0   substatement #endif`` |
+-----------------------------------+-----------------------------------+
| ``if target(provides(smXX) & ~is  | ``#if __CUDA_ARC                  |
| _exactly(smXX)):   substatement`` | H__ > XX0   substatement #endif`` |
+-----------------------------------+-----------------------------------+
| ``if target(                      | ``#if __CUDA_ARCH                 |
| provides(smXX)):   substatement`` | __ >= XX0   substatement #endif`` |
+-----------------------------------+-----------------------------------+

*Target specialization* enables the conditional use of constructs that
are only usable in a certain execution space without constraining the
calling function's usability in different execution spaces.

If the condition of an ``if`` statement:

-  is a constant expression,
-  contains a call to ``target``, and
-  evaluates to ``False``

then the use of any constructs that require a certain execution space
shall not require that execution space.

Note: This facility is based on `NVC++'s ``if target``
facility <https://docs.google.com/document/d/1BK7V_hS4-X35Ua9RzyQzRYDvXLR4Xh0T45Ke7uMsAsY/edit?tab=t.0#bookmark=id.41mghml>`__.

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