Constant Expressions
--------------------

Some facilities need certain parameters to be a value that is known
statically at JIT compilation time.

*Constant expressions* produce values suitable for such parameters.
Constant expressions are:

-  A literal value.
-  A local variable or parameter whose right-hand side is a literal
   value or constant expression.
-  A global variable that is defined at the time of compilation or
   launch.

``ConstExpr`` Class
^^^^^^^^^^^^^^^^^^^

A type hint that requires that the parameter shall come from a constant
expression.

``ConstExpr.__getitem__(type)`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a type hint that requires the parameter shall be of type
``type`` and come a constant expression. 