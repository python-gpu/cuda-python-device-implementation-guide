Heterogeneous Programming
=========================

This section defines:

-  The dialect of the Python programming language that can be evaluated
   on NVIDIA CUDA devices.
-  The ABI for said subset, including:

   -  The size, layout, alignment, conversion, and promotion of types.
   -  Object lifetime and reference model.
   -  The calling conventions for functions.

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

Execution Spaces
----------------

A program is executed on one or more *targets*, which are distinct
execution environments that are distinguished by different hardware
resources or programming models.

A function is *usable* if it can be called. A type or object is *usable*
if its attributes are accessible (can be read and written) and its
methods are callable.

Some functions, types, and objects are only usable on certain targets.

The set of targets that such a construct is usable on is called its
*execution space*.

*Host code* is the execution space that includes all CPU targets.

*Device code* is the execution space that includes all GPU targets.

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

Heterogeneous Types and Objects
-------------------------------

*Heterogeneous types* are types whose execution space includes host code
and device code. Note: Not all instances of a heterogeneous type are
usable in both host code and device code.

*Heterogeneous objects* are objects of a heterogeneous type whose
execution space includes both host code and device code.

Attributes and methods cannot be dynamically added in device code to the
heterogeneous types or objects defined in this document. Implementations
may define additional types and objects that support these operations.
[User Requirement]

``None``
~~~~~~~~

The ``NoneType`` is heterogeneous. The ``None`` object is heterogeneous.

When ``NoneType`` appears in a call to an interoperable function, it
shall have the same machine representation as ``void*`` in CUDA C++.
When None appears in a call to an interoperable function, it shall have
the same machine representation as a CUDA C++ object of type ``void*``
with a value of ``nullptr``.

Builtin Numbers
~~~~~~~~~~~~~~~

The *builtin numeric types* ``bool``, ``int``, ``float``, and
``complex`` are heterogeneous. Literal objects of these types are
heterogeneous.

The builtin numeric types shall have the following formats in device
code, and when they appear in a call to an interoperable function, they
shall have the same machine representation as the corresponding CUDA C++
types:

+-----------------------+-----------------------+-----------------------+
| Python Type           | CUDA C++ Type         | Format                |
+=======================+=======================+=======================+
| ``bool``              | ``bool``              | A boolean (either     |
|                       |                       | ``True`` or           |
|                       |                       | ``False``).           |
+-----------------------+-----------------------+-----------------------+
| ``int``               | `                     | A 32-bit signed       |
|                       | `cuda::std::int32_t`` | integer whose values  |
|                       |                       | exist on the interval |
|                       |                       | [−2,147,483,648,      |
|                       |                       | +2,147,483,647].      |
+-----------------------+-----------------------+-----------------------+
| ``float``             | ``c                   | IEEE 754              |
|                       | uda::std::float32_t`` | single-precision      |
|                       |                       | (32-bit) binary       |
|                       |                       | floating-point number |
|                       |                       | (see IEEE 754-2019).  |
+-----------------------+-----------------------+-----------------------+
| ``complex``           | ``cud                 | Single-precision      |
|                       | a::std::complex<  cud | (64-bit) complex      |
|                       | a::std::float32_t >`` | floating-point number |
|                       |                       | whose real and        |
|                       |                       | imaginary components  |
|                       |                       | must be IEEE 754      |
|                       |                       | single-precision      |
|                       |                       | (32-bit) binary       |
|                       |                       | floating-point        |
|                       |                       | numbers (see IEEE     |
|                       |                       | 754-2019).            |
+-----------------------+-----------------------+-----------------------+

In device code, if the value of a builtin number cannot be represented
by its corresponding format, the behavior is undefined.

Fixed-Format Numbers
~~~~~~~~~~~~~~~~~~~~

*Fixed-format numbers* represent a single number stored in a specific
machine format. They are heterogeneous.

Fixed-format numbers behave as if they are zero dimensional arrays. They
have the same attributes and methods as heterogeneous Python arrays.

The following fixed-format number types shall be defined, and when they
appear in a call to an interoperable function, they shall have the same
machine representation as the corresponding CUDA C++ types:

+-----------------------+-----------------------+-----------------------+
| Python Type           | CUDA C++ Type         | Format                |
+=======================+=======================+=======================+
| ``int8``              | ``cuda::std::int8_t`` | An 8-bit signed       |
|                       |                       | integer whose values  |
|                       |                       | exist on the interval |
|                       |                       | [-128, +127].         |
+-----------------------+-----------------------+-----------------------+
| ``int16``             | `                     | A 16-bit signed       |
|                       | `cuda::std::int16_t`` | integer whose values  |
|                       |                       | exist on the interval |
|                       |                       | [−32,768, +32,767].   |
+-----------------------+-----------------------+-----------------------+
| ``int32``             | `                     | A 32-bit signed       |
|                       | `cuda::std::int32_t`` | integer whose values  |
|                       |                       | exist on the interval |
|                       |                       | [−2,147,483,648,      |
|                       |                       | +2,147,483,647].      |
+-----------------------+-----------------------+-----------------------+
| ``int64``             | `                     | A 64-bit signed       |
|                       | `cuda::std::int64_t`` | integer whose values  |
|                       |                       | exist on the interval |
|                       |                       | [−9,223               |
|                       |                       | ,372,036,854,775,808, |
|                       |                       | +9,223,               |
|                       |                       | 372,036,854,775,807]. |
+-----------------------+-----------------------+-----------------------+
| ``uint8``             | `                     | An 8-bit unsigned     |
|                       | `cuda::std::uint8_t`` | integer whose values  |
|                       |                       | exist on the interval |
|                       |                       | [0, +255].            |
+-----------------------+-----------------------+-----------------------+
| ``uint16``            | ``                    | A 16-bit unsigned     |
|                       | cuda::std::uint16_t`` | integer whose values  |
|                       |                       | exist on the interval |
|                       |                       | [0, +65,535].         |
+-----------------------+-----------------------+-----------------------+
| ``uint32``            | ``                    | A 32-bit unsigned     |
|                       | cuda::std::uint32_t`` | integer whose values  |
|                       |                       | exist on the interval |
|                       |                       | [0, +4,294,967,295].  |
+-----------------------+-----------------------+-----------------------+
| ``uint64``            | ``                    | A 64-bit unsigned     |
|                       | cuda::std::uint64_t`` | integer whose values  |
|                       |                       | exist on the interval |
|                       |                       | [0,                   |
|                       |                       | +18,446,              |
|                       |                       | 744,073,709,551,615]. |
+-----------------------+-----------------------+-----------------------+
| ``float16``           | ``c                   | IEEE 754              |
|                       | uda::std::float16_t`` | half-precision        |
|                       |                       | (16-bit) binary       |
|                       |                       | floating-point number |
|                       |                       | (see IEEE 754-2019).  |
+-----------------------+-----------------------+-----------------------+
| ``float32``           | ``c                   | IEEE 754              |
|                       | uda::std::float32_t`` | single-precision      |
|                       |                       | (32-bit) binary       |
|                       |                       | floating-point number |
|                       |                       | (see IEEE 754-2019).  |
+-----------------------+-----------------------+-----------------------+
| ``float64``           | ``c                   | IEEE 754              |
|                       | uda::std::float64_t`` | double-precision      |
|                       |                       | (64-bit) binary       |
|                       |                       | floating-point number |
|                       |                       | (see IEEE 754-2019).  |
+-----------------------+-----------------------+-----------------------+
| ``complex64``         | ``cud                 | Single-precision      |
|                       | a::std::complex<  cud | (64-bit) complex      |
|                       | a::std::float32_t >`` | floating-point number |
|                       |                       | whose real and        |
|                       |                       | imaginary components  |
|                       |                       | must be IEEE 754      |
|                       |                       | single-precision      |
|                       |                       | (32-bit) binary       |
|                       |                       | floating-point        |
|                       |                       | numbers (see IEEE     |
|                       |                       | 754-2019).            |
+-----------------------+-----------------------+-----------------------+
| ``complex128``        | ``cud                 | Double-precision      |
|                       | a::std::complex<  cud | (128-bit) complex     |
|                       | a::std::float64_t >`` | floating-point number |
|                       |                       | whose real and        |
|                       |                       | imaginary components  |
|                       |                       | must be IEEE 754      |
|                       |                       | double-precision      |
|                       |                       | (64-bit) binary       |
|                       |                       | floating-point        |
|                       |                       | numbers (see IEEE     |
|                       |                       | 754-2019).            |
+-----------------------+-----------------------+-----------------------+

The following fixed-format types shall be defined in device code, and
may be defined in host code; if they are not, the types are not
heterogeneous. When they appear in a call to an interoperable function,
they shall have the same machine representation as the corresponding
CUDA C++ types:

+-----------------------+-----------------------+-----------------------+
| ``float8e4m3``        | ``__nv_fp8_e4m3``     | 8-bit floating-point  |
|                       |                       | number with 1 sign    |
|                       |                       | bit, 4 exponent bits, |
|                       |                       | and 3 mantissa bits.  |
+=======================+=======================+=======================+
| ``float8e5m2``        | ``__nv_fp8_e5m2``     | 8-bit floating-point  |
|                       |                       | number with 1 sign    |
|                       |                       | bit, 5 exponent bits, |
|                       |                       | and 2 mantissa bits.  |
+-----------------------+-----------------------+-----------------------+
| ``bfloat16``          | ``cu                  | 16-bit floating-point |
|                       | da::std::bfloat16_t`` | number with 1 sign    |
|                       |                       | bit, 8 exponent bits, |
|                       |                       | and 7 mantissa bits.  |
+-----------------------+-----------------------+-----------------------+

If the value of a fixed-format number cannot be represented by its
corresponding format, the behavior is undefined.

Vectors
~~~~~~~

``Vector`` Interface
^^^^^^^^^^^^^^^^^^^^

*Vectors* are collections of 1, 2, 3, or 4 objects of a single
heterogeneous type.

Vectors shall be immutable; any operation that modifies one shall return
a new object. Note: This is the same semantics as builtin numbers.

``Vector.size: int32`` and ``Vector.__len__: int32`` Attribute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The number of elements in the vector.

``Vector.dtype`` Attribute
^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Python Array API Standard
v2023.12 <https://data-apis.org/array-api/2023.12/API_specification/type_promotion.html>`__
data type of the vector’s elements.

``Vector.__init__(self, *args)`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Initializes the vector’s elements with ``args``.

An object of type ``self.dtype`` shall be constructible from each of the
args. [User Requirement]

``len(args) == len(self.size)``. [User Requirement]

``Vector.__getitem__(self, i: int32) -> bool`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns the ``i``\ th element of the vector.

``i < self.size`` [User Requirement]

``Vector.__setitem__(self, i: int32, val)`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assign ``val`` to the ``i``\ th element of the vector.

val shall be assignable to an object of ``self.dtype``. [User
Requirement]

``i < self.size`` [User Requirement]

``Vector.__iter__(self)`` Function

Returns an iterator to the elements of the vector.

``Vector.x`` Attribute
^^^^^^^^^^^^^^^^^^^^^^

The first element of the vector.

``Vector.y`` Attribute
^^^^^^^^^^^^^^^^^^^^^^

The second element of the vector.

This attribute shall only be defined if ``self.size > 1``.

``Vector.z`` Attribute
^^^^^^^^^^^^^^^^^^^^^^

The third element of the vector.

This attribute shall only be defined if ``self.size > 2``.

``Vector.w`` Attribute
^^^^^^^^^^^^^^^^^^^^^^

The fourth element of the vector.

This attribute shall only be defined if ``self.size > 3``.

Vector Types
^^^^^^^^^^^^

The following vector types shall be defined for *N* = 1, *N* = 2, *N* =
3, *N* = 4, and when they appear in a call to an interoperable function,
they shall have the same machine representation as the corresponding
CUDA C++ types:

================ ===================
Python Type      CUDA C++ Type
================ ===================
``int8xN``       ``charN``
``int16xN``      ``shortN``
``int32xN``      ``intN``
``int64xN``      ``longlongN``
``uint8xN``      ``ucharN``
``uint16xN``     ``ushortN``
``uint32xN``     ``uintN``
``uint64xN``     ``ulonglongN``
``float8e4m3xN`` ``__nv_fp8xN_e4m3``
``float8e5m2xN`` ``__nv_fp8xN_e5m2``
``float16xN``    ``__halfN``
``bfloat16xN``   ``__nvbfloat16N``
``float32xN``    ``floatN``
``float64xN``    ``doubleN``
================ ===================

Numeric Promotion
~~~~~~~~~~~~~~~~~

When performing arithmetic operations on two or more builtin numbers,
fixed-format numbers, and/or arrays, the type of the resulting object
shall be determined by the type promotion rules defined in the `Python
Array API Standard
v2023.12 <https://data-apis.org/array-api/2023.12/API_specification/type_promotion.html>`__.

Tuples
~~~~~~

+-----------------------------------+-----------------------------------+
| ``(8, 8, 8)``                     | ``struct __ano                    |
|                                   | nymous_tuple_0 {   cuda::std::int |
|                                   | 32_t __0;   cuda::std::int32_t __ |
|                                   | 1;   cuda::std::int32_t __2; };`` |
+===================================+===================================+
+-----------------------------------+-----------------------------------+

A *heterogeneous tuple* is a tuple of heterogeneous elements.

When a heterogeneous tuple appears in a call to an interoperable
function, it shall have the same machine representation as a CUDA C++
standard layout class with a public data member corresponding to each
element of the tuple ordered from first to last. Each such data member
shall have the machine representation that it would have if it was a
stand alone object.

Note: ``cuda::std::tuple`` is not used as the machine representation as
it does not have a specified layout in memory and discovering that
layout could be challenging for frameworks.

User-Defined Types
~~~~~~~~~~~~~~~~~~

+-----------------------------------+-----------------------------------+
| ``@device.struct class point:     | ``struct point {   cuda::st       |
|    x: int     y: int     z: int`` | d::int32_t x;   cuda::std::int32_ |
|                                   | t y;   cuda::std::int32_t z; };`` |
+===================================+===================================+
| ``@device                         | ``struct alignas(16) comple       |
| .struct(align=16) class complex:  | x {   cuda::std::float32_t real;  |
|     real: float     imag: float`` |   cuda::std::float32_t imag; };`` |
+-----------------------------------+-----------------------------------+
| ``@device.st                      |                                   |
| ruct class ticket_mutex:     line |                                   |
| : device.Atomic(int, align=16)    |                                   |
|   current: device.Atomic(int, ali |                                   |
| gn=16)     @device.func     def l |                                   |
| ock(self):         my = self.line |                                   |
| .add(1)         while True:       |                                   |
|      now = self.current.load()    |                                   |
|         if (now == my) break      |                                   |
|       self.current.wait(now)      |                                   |
| @device.func      def unlock(self |                                   |
| ):         self.current.add(1)    |                                   |
|       self.current.notify_all()`` |                                   |
+-----------------------------------+-----------------------------------+

A class decorated with ``@struct`` is a *heterogeneous struct type*,
which is a heterogeneous type. An instance of such a class is a
*heterogeneous struct*.

Each attribute of a heterogeneous struct shall have a type hint that is
heterogeneous. When a heterogeneous struct migrates to the device, the
attribute’s type shall match the type hint. [User Requirement]

Attributes shall not be dynamically added to a heterogeneous struct.
[User Requirement]

Heterogeneous structs shall be immutable; any operation that modifies
one shall return a new object. Note: This is the same semantics as
builtin numbers.

When a heterogeneous struct appears in a call to an interoperable
function, it shall have the same machine representation as a CUDA C++
standard layout class with a public data member corresponding to each
attribute of the heterogeneous struct ordered as they lexically appear
in Python. Each such data member shall have the machine representation
that it would have if it was a stand alone object.

``@struct`` and ``@struct(align: int)`` Class Decorator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``@struct`` is used without arguments, the heterogeneous struct will
have default alignment. If ``@struct`` is used with an ``align``
argument, the heterogeneous struct shall be aligned to at least that
many bytes.

``@struct`` returns a type that has an ``underlying`` attribute that
produces the class that was decorated.

``align(t: type, n: int) -> type`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a new type that is equivalent to ``t`` and is aligned to at
least ``n`` bytes.

Arrays
~~~~~~

A *heterogeneous array* is a tensor of heterogeneous objects.

A heterogeneous array shall implement the `Python Array API Standard
v2023.12 <https://data-apis.org/array-api/2023.12/API_specification/type_promotion.html>`__
in host code and the following subset of said API in device code:

-  Indexing.
-  Slicing.
-  Striding.
-  Reading attributes whose type is a heterogeneous Python type.
   Example: ``.dtype``, ``.shape``, ``.strides``, ``.size, and``
   ``.ndims`` can be read in device code, but ``.flags`` cannot because
   it is not a heterogeneous Python type.
-  Calls to ``view``.
-  Calls to ``reshape`` that do not allocate.
-  Calls to ``astype`` with copy=False that do not allocate.

A heterogeneous array shall support the following data types:

-  The data types for the Builtin Numbers and Fixed Format Numbers
   defined in this document.
-  `NumPy Structured Data
   Types <https://numpy.org/devdocs/user/basics.rec.html#structured-arrays>`__
   that are compositions of the data types mentioned above.
-  ``@struct`` classes.

A heterogeneous array shall implement either the `DLPack Python
Specification <https://dmlc.github.io/dlpack/latest/python_spec.html>`__
or the `CUDA Array Interface Version
3 <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`__.
Note: This explicit opt-in prevents arrays from accidentally being
treated as heterogeneous. [User Requirement]

If a heterogeneous array implements both the DLPack Specification and
the CUDA Array Interface, the DLPack Specification shall be used.

When a heterogeneous array ``x`` appears in a call to an interoperable
function, it shall have the same machine representation as the following
CUDA C++ type:

.. code:: c

   template <typename dtype, cuda::std::uint64_t ndim>
   struct cuda::interoperable_array_descriptor {
     dtype* data;
     cuda::std::uint64_t shape[ndim];
     cuda::std::uint64_t strides[ndim];

     template <typename mdspan>
     interoperable_array_descriptor(mdspan&& ms);

     operator auto() const {
       return cuda::std::mdspan{data,
         cuda::std::layout_stride::mapping{cuda::std::dims<ndim>(shape), strides}};
     }
   };

where:

-  ``dtype`` is the machine representation of ``x.dtype``.
-  ``ndim`` is ``x.ndim``.
-  ``data`` is ``x.data``.
-  ``size`` is ``x.size``.
-  ``shape`` is ``x.shape``.
-  ``strides`` is ``x.strides``.

Note: ``cuda::std::mdspan`` is not used as the machine representation as
it does not have a specified layout in memory and discovering that
layout could be challenging for frameworks.

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
calling function’s usability in different execution spaces.

If the condition of an ``if`` statement:

-  is a constant expression,
-  contains a call to ``target``, and
-  evaluates to ``False``

then the use of any constructs that require a certain execution space
shall not require that execution space.

Note: This facility is based on `NVC++’s ``if target``
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
