Heterogeneous Functions
-----------------------

.. code:: py

   @device.func
   def recip(a):
     return 1 / a
   # Can be called in device code.

   @device.func(interop=True)
   def diff(a, b):
     return abs(a - b)
   # Can be called in device code by other frameworks and languages.

A *heterogeneous function* is a function whose execution space includes
host and device code.

``@func`` and ``@func(interop=False, **kwargs)`` Function Decorator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A function with the ``@func`` decorator shall be a heterogeneous
function. [User Requirement]

Frameworks may produce an error if a heterogeneous function is not
decorated with either ``@func`` or ``@kernel``.

``@func`` returns a heterogeneous function that has an ``underlying``
attribute that produces the function that was decorated.

If ``@func`` is used with an interop argument that is ``True``, then the
function shall be interoperable.

Frameworks may define additional keyword arguments for ``@func``.
Frameworks shall produce an error if ``@func`` is invoked with any
unsupported or unknown keyword arguments. Example: A framework could
provide a link argument: ``@func(link=files)``. 