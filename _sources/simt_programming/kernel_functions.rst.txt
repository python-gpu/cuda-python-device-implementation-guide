Kernel Functions
----------------

.. code:: py

   @device.kernel
   def vec_add(a, b, c):
     c[device.tid(1)] = a[device.tid(1)] + b[device.tid(1)]

   dev = core.Device(0)
   dev.set_current()
   stm = dev.create_stream()

   N = 1024
   a = cupy.random.random(N)
   b = cupy.random.random(N)
   c = cupy.zeros_like(a)
   device.launch(vec_add, grid=4, block=256, stream=stm)

A *kernel function* is a function that shall be launched on a device,
where it will be executed simultaneously by a group of threads.

A kernel function is device only.

A kernel function shall return ``None``. [User Requirement]

Interoperable kernel functions shall have the same symbol and calling
convention as the machine representation of a ``__global__`` CUDA C++
function of the same name and same parameters. Note: An interoperable
kernel function is callable with ``cudaLaunchKernel``.

The machine representation of an interoperable kernel function shall
take the machine representation of each of its parameters as if they
were passed by value as a single parameter in CUDA C++.

The machine representation of a non-interoperable kernel function may
take additional parameters.

``@kernel`` and ``@kernel(interop=False, **kwargs)`` Function Decorator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A function with the ``@kernel`` decorator shall be a device only
function. [User Requirement]

A function with the ``@kernel`` decorator shall not have the ``@func``
decorator. [User Requirement]

``@kernel`` returns an object that has an ``underlying`` attribute that
produces the function that was decorated.

If ``@kernel`` is used with an interop argument that is ``True``, then
the function shall be interoperable.

Frameworks may define additional keyword arguments for ``@kernel``.
Frameworks shall produce an error if ``@kernel`` is invoked with any
unsupported or unknown keyword arguments. Example: A framework could
provide a link argument: ``@kernel(link=files)``

``launch(f, *args, grid: Shape, block: Shape, stream: core.Stream, shared=0: int32)`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Asynchronously executes on ``stream`` a thread grid of ``grid`` blocks
each with ``block`` threads and ``shared`` bytes of dynamic shared
memory, each thread of which executes ``f(args)``.

``f`` shall be a kernel function. [User Requirement]

All of ``args`` shall be heterogeneous. [User Requirement] 