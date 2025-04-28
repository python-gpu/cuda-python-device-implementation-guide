# Introduction

## Scope, Motivation, and Goals

This document defines how to write SIMT code for CUDA devices in Python, how *Python frameworks (*Python libraries, compilers, interpreters, domain specific languages) evaluate such *device code* on the CUDA platform, and how such frameworks interact with each other and with CUDA C++.

This specification has been developed to:

* Adoption: Encourage the use of CUDA parallelism in Python.   
* Assurance: Give users confidence that CUDA parallelism in Python can be relied on and will be supported.  
* Leadership: Establish trust, respect, and influence for NVIDIA in the Python community.  
* Consistency: Provide a uniform and coherent experience across different Python frameworks.

We want to make CUDA Python programming:

* Easy.  
* Performant.  
* Modern.  
* Pythonic.

We want to enable:

* Source Portability: Use the same Python and CUDA C++ source code with different Python frameworks without rewriting.  
  * Example: Compile the same Python code with Numba and CuPy.  
  * Example: Use the same Python user-defined type and reduction operator with Numba and Warp.  
  * Example: Write Python bindings once that will support multiple frameworks for a CUDA C++ library that contains host code, device code, and C++ templates.   
* Interoperability: Use different Python frameworks and/or CUDA C++ together, either within a single kernel (intra-kernel) or between different kernels (inter-kernel).  
  * Example: Launch a Numba kernel followed by a Warp kernel on the same data.  
  * Example: Call a Python user-defined load operation in CUB or cuFFTDx.

## Conformance

This document defines two kinds of requirements:

* Framework Requirements: Requirements on the behavior of Python frameworks.  
* User Requirements: Requirements on Python source code. If such a requirement is violated, the program is ill-formed, and there is a framework requirement to produce an error. Such requirements are explicitly annotated in this document.

The first framework requirement is to implement the entities and semantics defined in this document.

This document places no requirements on the structure of Python frameworks. Conforming frameworks are only required to emulate the observable behavior defined by this document. A framework is free to disregard any requirement of this document as long as the result is as if the requirement had been obeyed, as far as can be determined from the observable behavior of the program.

## General Requirements

All entities defined in this document shall be in the `cuda.device` namespace.

`core.` entities referenced in this document are from `cuda.core`.

When calling any function defined in this document, the type of an argument shall match the type hint of the corresponding parameter if it has one. \[User Requirement\]

*Exposition only* entities are only described to help define other entities.

Exposition only entities are not part of the interface and shall not be used. \[User Requirement\]

# Python and CUDA C++

This section defines interoperability and bindings between Python and CUDA C++. 

## Machine Representation

Python frameworks execute Python code on devices by translating the Python code into a *machine representation* that can be executed by CUDA devices. 

Likewise, CUDA C++ evaluates CUDA C++ code on devices by translating the CUDA C++ code into a machine representation.

#### `machine_representation() -> str` Function

Returns an unspecified description of the machine representation format.

Example: `"itanium"` on Linux, `"msvc"` on Windows.

## Interoperable Functions

Python and CUDA C++ interoperate together by calling certain *interoperable functions* defined by the other language.

Python interoperable functions shall have the same symbol and calling convention as the machine representation of an `extern "C" __host__ __device__` CUDA C++ function of the same name. 

The machine representation of a Python interoperable function shall take the machine representation of each of its parameters as if they were passed by value in CUDA C++. 

This document defines requirements for the machine representation of types and objects when they are taken as parameters to and returned from interoperable functions. These requirements only apply when calling interoperable functions.

# Python Heterogeneous Programming

This section defines:

* The dialect of the Python programming language that can be evaluated on NVIDIA CUDA devices.  
* The ABI for said subset, including:  
  * The size, layout, alignment, conversion, and promotion of types.  
  * Object lifetime and reference model.  
  * The calling conventions for functions.

## Constant Expressions

Some facilities need certain parameters to be a value that is known statically at JIT compilation time.

*Constant expressions* produce values suitable for such parameters. Constant expressions are:

* A literal value.  
* A local variable or parameter whose right-hand side is a literal value or constant expression.  
* A global variable that is defined at the time of compilation or launch.

#### `ConstExpr` Class

A type hint that requires that the parameter shall come from a constant expression.

#### `ConstExpr.__getitem__(type)` Function

Returns a type hint that requires the parameter shall be of type `type` and come a constant expression.

## Execution Spaces

A program is executed on one or more *targets*, which are distinct execution environments that are distinguished by different hardware resources or programming models.

A function is *usable* if it can be called. A type or object is *usable* if its attributes are accessible (can be read and written) and its methods are callable.

Some functions, types, and objects are only usable on certain targets.

The set of targets that such a construct is usable on is called its *execution space*.

*Host code* is the execution space that includes all CPU targets.

*Device code* is the execution space that includes all GPU targets.

## Heterogeneous Functions

```py
@device.func
def recip(a):
  return 1 / a
# Can be called in device code.

@device.func(interop=True)
def diff(a, b):
  return abs(a - b)
# Can be called in device code by other frameworks and languages.
```

A *heterogeneous function* is a function whose execution space includes host and device code.

#### `@func` and `@func(interop=False, **kwargs)` Function Decorator

A function with the `@func` decorator shall be a heterogeneous function. \[User Requirement\]

Frameworks may produce an error if a heterogeneous function is not decorated with either `@func` or `@kernel`.

`@func` returns a heterogeneous function that has an `underlying` attribute that produces the function that was decorated.

If `@func` is used with an interop argument that is `True`, then the function shall be interoperable.

Frameworks may define additional keyword arguments for `@func`. Frameworks shall produce an error if `@func` is invoked with any unsupported or unknown keyword arguments. Example: A framework could provide a link argument: `@func(link=files)`.

## Heterogeneous Types and Objects

*Heterogeneous types* are types whose execution space includes host code and device code. Note: Not all instances of a heterogeneous type are usable in both host code and device code.

*Heterogeneous objects* are objects of a heterogeneous type whose execution space includes both host code and device code.

Attributes and methods cannot be dynamically added in device code to the heterogeneous types or objects defined in this document. Implementations may define additional types and objects that support these operations. \[User Requirement\]

### `None`

The `NoneType` is heterogeneous. The `None` object is heterogeneous.

When `NoneType` appears in a call to an interoperable function, it shall have the same machine representation as `void*` in CUDA C++. When None appears in a call to an interoperable function, it shall have the same machine representation as a CUDA C++ object of type `void*` with a value of `nullptr`.

### Builtin Numbers

The *builtin numeric types* `bool`, `int`, `float`, and `complex` are heterogeneous. Literal objects of these types are heterogeneous.

The builtin numeric types shall have the following formats in device code, and when they appear in a call to an interoperable function, they shall have the same machine representation as the corresponding CUDA C++ types:

| Python Type | CUDA C++ Type | Format |
| :---- | :---- | :---- |
| `bool` | `bool` | A boolean (either `True` or `False`). |
| `int` | `cuda::std::int32_t` | A 32-bit signed integer whose values exist on the interval \[−2,147,483,648, \+2,147,483,647\]. |
| `float` | `cuda::std::float32_t` | IEEE 754 single-precision (32-bit) binary floating-point number (see IEEE 754-2019). |
| `complex` | `cuda::std::complex<  cuda::std::float32_t >` | Single-precision (64-bit) complex floating-point number whose real and imaginary components must be IEEE 754 single-precision (32-bit) binary floating-point numbers (see IEEE 754-2019). |

In device code, if the value of a builtin number cannot be represented by its corresponding format, the behavior is undefined.

### Fixed-Format Numbers

*Fixed-format numbers* represent a single number stored in a specific machine format. They are heterogeneous.

Fixed-format numbers behave as if they are zero dimensional arrays. They have the same attributes and methods as heterogeneous Python arrays.

The following fixed-format number types shall be defined, and when they appear in a call to an interoperable function, they shall have the same machine representation as the corresponding CUDA C++ types:

| Python Type | CUDA C++ Type | Format |
| :---- | :---- | :---- |
| `int8` | `cuda::std::int8_t` | An 8-bit signed integer whose values exist on the interval \[-128, \+127\]. |
| `int16` | `cuda::std::int16_t` | A 16-bit signed integer whose values exist on the interval \[−32,768, \+32,767\]. |
| `int32` | `cuda::std::int32_t` | A 32-bit signed integer whose values exist on the interval \[−2,147,483,648, \+2,147,483,647\]. |
| `int64` | `cuda::std::int64_t` | A 64-bit signed integer whose values exist on the interval \[−9,223,372,036,854,775,808, \+9,223,372,036,854,775,807\]. |
| `uint8` | `cuda::std::uint8_t` | An 8-bit unsigned integer whose values exist on the interval \[0, \+255\]. |
| `uint16` | `cuda::std::uint16_t` | A 16-bit unsigned integer whose values exist on the interval \[0, \+65,535\]. |
| `uint32` | `cuda::std::uint32_t` | A 32-bit unsigned integer whose values exist on the interval \[0, \+4,294,967,295\]. |
| `uint64` | `cuda::std::uint64_t` | A 64-bit unsigned integer whose values exist on the interval \[0, \+18,446,744,073,709,551,615\]. |
| `float16` | `cuda::std::float16_t` | IEEE 754 half-precision (16-bit) binary floating-point number (see IEEE 754-2019). |
| `float32` | `cuda::std::float32_t` | IEEE 754 single-precision (32-bit) binary floating-point number (see IEEE 754-2019). |
| `float64` | `cuda::std::float64_t` | IEEE 754 double-precision (64-bit) binary floating-point number (see IEEE 754-2019). |
| `complex64` | `cuda::std::complex<  cuda::std::float32_t >` | Single-precision (64-bit) complex floating-point number whose real and imaginary components must be IEEE 754 single-precision (32-bit) binary floating-point numbers (see IEEE 754-2019). |
| `complex128` | `cuda::std::complex<  cuda::std::float64_t >` | Double-precision (128-bit) complex floating-point number whose real and imaginary components must be IEEE 754 double-precision (64-bit) binary floating-point numbers (see IEEE 754-2019). |

The following fixed-format types shall be defined in device code, and may be defined in host code; if they are not, the types are not heterogeneous. When they appear in a call to an interoperable function, they shall have the same machine representation as the corresponding CUDA C++ types:

| `float8e4m3` | `__nv_fp8_e4m3` | 8-bit floating-point number with 1 sign bit, 4 exponent bits, and 3 mantissa bits. |
| :---- | :---- | :---- |
| `float8e5m2` | `__nv_fp8_e5m2` | 8-bit floating-point number with 1 sign bit, 5 exponent bits, and 2 mantissa bits. |
| `bfloat16` | `cuda::std::bfloat16_t` | 16-bit floating-point number with 1 sign bit, 8 exponent bits, and 7 mantissa bits. |

If the value of a fixed-format number cannot be represented by its corresponding format, the behavior is undefined.

### Vectors

#### `Vector` Interface

*Vectors* are collections of 1, 2, 3, or 4 objects of a single heterogeneous type.

Vectors shall be immutable; any operation that modifies one shall return a new object. Note: This is the same semantics as builtin numbers.

#### `Vector.size: int32` and `Vector.__len__: int32` Attribute

The number of elements in the vector.

#### `Vector.dtype` Attribute

The [Python Array API Standard v2023.12](https://data-apis.org/array-api/2023.12/API_specification/type_promotion.html) data type of the vector's elements.

#### `Vector.__init__(self, *args)` Function

Initializes the vector's elements with `args`.

An object of type `self.dtype` shall be constructible from each of the args. \[User Requirement\]

`len(args) == len(self.size)`. \[User Requirement\]

#### `Vector.__getitem__(self, i: int32) -> bool` Function

Returns the `i`th element of the vector.

`i < self.size` \[User Requirement\]

#### `Vector.__setitem__(self, i: int32, val)` Function

Assign `val` to the `i`th element of the vector.

val shall be assignable to an object of `self.dtype`. \[User Requirement\]

`i < self.size` \[User Requirement\]

`Vector.__iter__(self)` Function

Returns an iterator to the elements of the vector.

#### `Vector.x` Attribute

The first element of the vector.

#### `Vector.y` Attribute

The second element of the vector.

This attribute shall only be defined if `self.size > 1`. 

#### `Vector.z` Attribute

The third element of the vector.

This attribute shall only be defined if `self.size > 2`. 

#### `Vector.w` Attribute

The fourth element of the vector.

This attribute shall only be defined if `self.size > 3`. 

#### Vector Types

The following vector types shall be defined for *N* \= 1, *N* \= 2, *N* \= 3, *N* \= 4, and when they appear in a call to an interoperable function, they shall have the same machine representation as the corresponding CUDA C++ types:

| Python Type | CUDA C++ Type |
| :---- | :---- |
| `int8xN` | `charN` |
| `int16xN` | `shortN` |
| `int32xN` | `intN` |
| `int64xN` | `longlongN` |
| `uint8xN` | `ucharN` |
| `uint16xN` | `ushortN` |
| `uint32xN` | `uintN` |
| `uint64xN` | `ulonglongN` |
| `float8e4m3xN` | `__nv_fp8xN_e4m3` |
| `float8e5m2xN` | `__nv_fp8xN_e5m2` |
| `float16xN` | `__halfN` |
| `bfloat16xN` | `__nvbfloat16N` |
| `float32xN` | `floatN` |
| `float64xN` | `doubleN` |

### Numeric Promotion

When performing arithmetic operations on two or more builtin numbers, fixed-format numbers, and/or arrays, the type of the resulting object shall be determined by the type promotion rules defined in the [Python Array API Standard v2023.12](https://data-apis.org/array-api/2023.12/API_specification/type_promotion.html). 

### Tuples

|  `(8, 8, 8)`  |  `struct __anonymous_tuple_0 {   cuda::std::int32_t __0;   cuda::std::int32_t __1;   cuda::std::int32_t __2; };`  |
| :---- | :---- |

A *heterogeneous tuple* is a tuple of heterogeneous elements.

When a heterogeneous tuple appears in a call to an interoperable function, it shall have the same machine representation as a CUDA C++ standard layout class with a public data member corresponding to each element of the tuple ordered from first to last. Each such data member shall have the machine representation that it would have if it was a stand alone object. 

Note: `cuda::std::tuple` is not used as the machine representation as it does not have a specified layout in memory and discovering that layout could be challenging for frameworks. 

### User-Defined Types

|  `@device.struct class point:     x: int     y: int     z: int`  |  `struct point {   cuda::std::int32_t x;   cuda::std::int32_t y;   cuda::std::int32_t z; };`  |
| :---- | :---- |
|  `@device.struct(align=16) class complex:     real: float     imag: float`  |  `struct alignas(16) complex {   cuda::std::float32_t real;   cuda::std::float32_t imag; };`  |
|  `@device.struct class ticket_mutex:     line: device.Atomic(int, align=16)     current: device.Atomic(int, align=16)     @device.func     def lock(self):         my = self.line.add(1)         while True:           now = self.current.load()           if (now == my) break           self.current.wait(now)     @device.func      def unlock(self):         self.current.add(1)         self.current.notify_all()`  |  |

A class decorated with `@struct` is a *heterogeneous struct type*, which is a heterogeneous type. An instance of such a class is a *heterogeneous struct*.

Each attribute of a heterogeneous struct shall have a type hint that is heterogeneous. When a heterogeneous struct migrates to the device, the attribute's type shall match the type hint. \[User Requirement\] 

Attributes shall not be dynamically added to a heterogeneous struct. \[User Requirement\]

Heterogeneous structs shall be immutable; any operation that modifies one shall return a new object. Note: This is the same semantics as builtin numbers.

When a heterogeneous struct appears in a call to an interoperable function, it shall have the same machine representation as a CUDA C++ standard layout class with a public data member corresponding to each attribute of the heterogeneous struct ordered as they lexically appear in Python. Each such data member shall have the machine representation that it would have if it was a stand alone object. 

#### `@struct` and `@struct(align: int)` Class Decorator

If `@struct` is used without arguments, the heterogeneous struct will have default alignment. If `@struct` is used with an `align` argument, the heterogeneous struct shall be aligned to at least that many bytes. 

`@struct` returns a type that has an `underlying` attribute that produces the class that was decorated.

#### `align(t: type, n: int) -> type` Function

Returns a new type that is equivalent to `t` and is aligned to at least `n` bytes. 

### Arrays

A *heterogeneous array* is a tensor of heterogeneous objects. 

A heterogeneous array shall implement the [Python Array API Standard v2023.12](https://data-apis.org/array-api/2023.12/API_specification/type_promotion.html) in host code and the following subset of said API in device code:

* Indexing.  
* Slicing.  
* Striding.  
* Reading attributes whose type is a heterogeneous Python type. Example: `.dtype`, `.shape`, `.strides`, `.size, and` `.ndims` can be read in device code, but `.flags` cannot because it is not a heterogeneous Python type.  
* Calls to `view`.  
* Calls to `reshape` that do not allocate.  
* Calls to `astype` with copy=False that do not allocate.

A heterogeneous array shall support the following data types:

* The data types for the Builtin Numbers and Fixed Format Numbers defined in this document.  
* [NumPy Structured Data Types](https://numpy.org/devdocs/user/basics.rec.html#structured-arrays) that are compositions of the data types mentioned above.  
* `@struct` classes.

A heterogeneous array shall implement either the [DLPack Python Specification](https://dmlc.github.io/dlpack/latest/python_spec.html) or the [CUDA Array Interface Version 3](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html). Note: This explicit opt-in prevents arrays from accidentally being treated as heterogeneous. \[User Requirement\]

If a heterogeneous array implements both the DLPack Specification and the CUDA Array Interface, the DLPack Specification shall be used. 

When a heterogeneous array `x` appears in a call to an interoperable function, it shall have the same machine representation as the following CUDA C++ type: 

```c
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
```

where:

* `dtype` is the machine representation of `x.dtype`.  
* `ndim` is `x.ndim`.  
* `data` is `x.data`.  
* `size` is `x.size`.  
* `shape` is `x.shape`.  
* `strides` is `x.strides`.

Note: `cuda::std::mdspan` is not used as the machine representation as it does not have a specified layout in memory and discovering that layout could be challenging for frameworks.

## Target Specialization

|  `if target(is_host):   substatement`  |  `#if !__CUDA_ARCH__   substatement #endif`  |
| :---- | :---- |
|  `if target(is_device):   substatement`  |  `#if __CUDA_ARCH__   substatement #endif`  |
|  `if target(is_exactly(smXX) \         | is_exactly(smYY) \         | is_exactly(smZZ)):   substatement`  |  `#if __CUDA_ARCH__ == XX0 \  || __CUDA_ARCH__ == YY0 \  || __CUDA_ARCH__ == ZZ0   substatement #endif`  |
|  `if target(!provides(smXX)):   substatement`  |  `#if __CUDA_ARCH__ < XX0   substatement #endif`  |
|  `if target(!provides(smXX) | is_exactly(smXX)):   substatement`  |  `#if __CUDA_ARCH__ <= XX0   substatement #endif`  |
|  `if target(provides(smXX) & ~is_exactly(smXX)):   substatement`  |  `#if __CUDA_ARCH__ > XX0   substatement #endif`  |
|  `if target(provides(smXX)):   substatement`  |  `#if __CUDA_ARCH__ >= XX0   substatement #endif`  |

*Target specialization* enables the conditional use of constructs that are only usable in a certain execution space without constraining the calling function's usability in different execution spaces.

If the condition of an `if` statement:

* is a constant expression,  
* contains a call to `target`, and  
* evaluates to `False`

then the use of any constructs that require a certain execution space shall not require that execution space.

Note: This facility is based on [NVC++'s `if target` facility](https://docs.google.com/document/d/1BK7V_hS4-X35Ua9RzyQzRYDvXLR4Xh0T45Ke7uMsAsY/edit?tab=t.0#bookmark=id.41mghml).

#### `target(desc: ConstExpr[TargetDescription]) -> bool` Function

Returns `True` if the current target matches `desc`.

#### `TargetDescription` Class

A `TargetDescription` object describes an execution space.

#### `TargetDescription.__or__(self, desc: ConstExpr[TargetDescription]) -> TargetDescription` Function

Returns a `TargetDescription` that matches all the targets in either `self` and `desc`.

#### `TargetDescription.__and__(self, desc: ConstExpr[TargetDescription]) -> TargetDescription` Function

Returns a `TargetDescription` that matches all the targets in both `self` and `desc`.

#### `TargetDescription.__invert__(self) -> TargetDescription` Function

Returns a `TargetDescription` that matches all the targets that are not in `self`.

#### `SMSelector` Class

An `SMSelector` represents a particular CUDA architecture.

#### `smXX: SMSelector` Objects

For each supported CUDA architecture *XX*, a `SMSelector` `smXX` shall be defined.

Example: `sm80` would be defined for Ampere.

#### `is_exactly(sm: ConstExpr[SMSelector]) -> TargetDescription` Function

Returns a `TargetDescription` that matches only the target `sm`.

#### `provides(sm: ConstExpr[SMSelector]) -> TargetDescription` Function

Returns a `TargetDescription` that matches all targets that support the capabilities of target `sm`.

Example: `provides(sm70)` is a `TargetDescription` that includes `sm70`, `sm72`, `sm75`, `sm80`, and all newer CUDA architectures.

#### `is_host: TargetDescription` Object

A `TargetDescription` that matches all host targets.

#### `is_device: TargetDescription` Object

A `TargetDescription` that matches all devices targets.

#### `any_target: TargetDescription` Object

A `TargetDescription` that always matches.

#### `no_target: TargetDescription` Object

A `TargetDescription` that never matches.

# CUDA Python Programming

This section defines Python interfaces for CUDA SIMT programming \- writing device code that explicitly controls cooperative threads, warps, and blocks.

## Device Only Entities

*Device only* functions, types, and objects whose execution space is device code only.

Device only entities shall not be used in host code. \[User Requirement\]

## *`Dim3Like`* Object (Exposition Only)

```py
Dim3Like = tuple[int32, int32, int32] \
     | tuple[int32, int32] \
     | tuple[int32] \
     | int32
```

A type hint for parameters describing the shape of a level in the thread hierarchy.

`Dim3Like` is exposition only.

## Kernel Functions

```py
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
```

A *kernel function* is a function that shall be launched on a device, where it will be executed simultaneously by a group of threads.

A kernel function is device only.

A kernel function shall return `None`. \[User Requirement\]

Interoperable kernel functions shall have the same symbol and calling convention as the machine representation of a `__global__` CUDA C++ function of the same name and same parameters. Note: An interoperable kernel function is callable with `cudaLaunchKernel`.

The machine representation of an interoperable kernel function shall take the machine representation of each of its parameters as if they were passed by value as a single parameter in CUDA C++.

The machine representation of a non-interoperable kernel function may take additional parameters.

#### `@kernel` and `@kernel(interop=False, **kwargs)` Function Decorator

A function with the `@kernel` decorator shall be a device only function. \[User Requirement\]

A function with the `@kernel` decorator shall not have the `@func` decorator. \[User Requirement\]

`@kernel` returns an object that has an `underlying` attribute that produces the function that was decorated.

If `@kernel` is used with an interop argument that is `True`, then the function shall be interoperable.

Frameworks may define additional keyword arguments for `@kernel`. Frameworks shall produce an error if `@kernel` is invoked with any unsupported or unknown keyword arguments. Example: A framework could provide a link argument: `@kernel(link=files)`

#### `launch(f, *args, grid: Dim3Like, block: Dim3Like, stream: core.Stream, shared=0: int32)` Function

Asynchronously executes on `stream` a thread grid of `grid` blocks each with `block` threads and `shared` bytes of dynamic shared memory, each thread of which executes `f(args)`.

f shall be a kernel function. \[User Requirement\]

All of `args` shall be heterogeneous. \[User Requirement\]

## Thread Positioning

All entities in this section are device only.

`Dim3` Class

`Dim3` is a subclass of `uint32x3`.

#### `thread_idx: Dim3` Object

The thread indices' in the current thread block.

Each index `x`, `y`, and `z` shall be greater than or equal to 0 and less than `block_dim.x`, `block_dim.y`, and `block_dim.z` respectively.

#### `tid(ndims): Dim3Like` Function

The absolute position of the thread in the grid.

Note: Returns a single integer if ndims is 1, and a tuple of integers otherwise.

Equivalent to:

```py
def tid(ndims):
  pos = tuple(t + bi * bd for t, bi, bd in zip(thread_idx, block_idx, block_dim))
  return pos[0] if ndims == 1 else pos[0:ndims]
```

`ndims > 0 and ndims <= 3` \[User Requirement\]

#### `block_idx: Dim3` Object

The block indices in the grid of thread blocks.

Each index `x`, `y`, and `z` shall be greater than or equal to 0 and less than `grid_dim.x`, `grid_dim.y`, and `grid_dim.z` respectively.

#### `block_dim: Dim3` Object

The shape of each block of threads.

#### `grid_dim: Dim3` Object

The shape of the grid of blocks.

#### `grid_size(ndims): Dim3Like` Function

The absolute size of the grid.

Note: Returns a single integer if ndims is 1, and a tuple of integers otherwise.

Equivalent to:

```py
def grid_size(ndims):
  size = tuple(b * g for b, g in zip(block_dim, thread_dim))
  return size[0] if ndims == 1 else size[0:ndims]
```

`ndims > 0 and ndims <= 3` \[User Requirement\]

#### `warp_size: int32` Object

The number of threads per warp.

#### `lane_id: int32` Object

The thread index in the current warp. It is greater than or equal to `0` and less than `warp_size`.

## Local, Shared, and Const Memory

All entities in this section are device only.

#### `local_array(shape: ConstExpr, dtype, order='C': {'C','F'}, align: uint32)` Function

Creates an array of the given shape, NumPy data type, and alignment in memory private to each thread.

`shape` shall be an integer or tuple of integers. \[User Requirement\]

#### `shared_array(shape: ConstExpr, dtype, order='C': {'C','F'}, align: uint32)` Function

Creates an array of the given shape, NumPy data type, and alignment in memory shared across the thread block.

`shape` shall be an integer or tuple of integers. \[User Requirement\]

#### `dynamic_shared_array()` Function

Returns a 1D array of `uint8` that references dynamic shared memory.

The size of the array is the size of the `shared` parameter of the current kernel launch.

## Synchronization

### Memory Orderings

A *memory order* specifies how atomic and non-atomic memory accesses are ordered around a synchronization primitive. Memory orders are defined by [Standard C++ (ISO/IEC 14882:2023)](https://timsong-cpp.github.io/cppwp/n4950/atomics.order).

Functions that take a memory order have a parameter named `memory`.

If the `memory` parameter is equal to one of the following, the function shall have the behavior of the corresponding Standard C++ memory order:

| Python Parameter Equals | Standard C++ Thread Scope |
| :---- | :---- |
| `'relaxed'` | `std::memory_order_relaxed` |
| `'consume'` | `std::memory_order_consume` |
| `'acquire'` | `std::memory_order_acquire` |
| `'release'` | `std::memory_order_release` |
| `'acq_rel'` | `std::memory_order_acq_rel` |
| `'seq_cst'` | `std::memory_order_seq_cst` |

An error shall occur if the `memory` parameter is not equal to one of the above values. \[User Requirement\]

### Thread Scopes

A *thread scope* specifies the kind of threads that can synchronize with each other using a synchronization primitive. ThreadThreads scopes are defined by [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes).

Functions that take a thread scope have a parameter named `scope`.

If the `scope` parameter is equal to one of the following, the function shall have the behavior of the corresponding CUDA C++ thread scope:

| Python Parameter Equals | CUDA C++ Thread Scope |
| :---- | :---- |
| `'system'` | `cuda::thread_scope_system` |
| `'device'` | `cuda::thread_scope_device` |
| `'block'` | `cuda::thread_scope_block` |
| `'thread'` | `cuda::thread_scope_thread` |

An error shall occur if the `scope` parameter is not equal to one of the above values. \[User Requirement\]

### Atomics

#### `AtomicInterface` Class

A class that subclasses `AtomicInterface` represents a scalar object that can be accessed atomically. `AtomicInterface` types are usable in device code.

Frameworks may define `AtomicInterface` as device only.

In this section, *atomic's object* means the scalar object that an `AtomicInterface` refers to.

#### `Atomic` Class

A class that owns a scalar object that is accessed atomically.

#### `Atomic.__init__(self, dtype)` Function

Creates an `Atomic` object containing an object of `dtype`.

The size of the machine representation of the dtype of the atomic's object shall be less than or equal to 16 bytes. \[User Requirement\]

#### `Atomic.dtype` Attribute

The data type of the atomic's object. 

#### `atomic_ref(array, index) -> AtomicInterface` Function

Returns an object that implements the `AtomicInterface`  and represents the object at `array[index]`.

Users shall preserve the lifetime of `array` for as long as the returned `AtomicInterface` may be used. \[User Requirement\]

#### `AtomicInterface.dtype` Attribute

The dtype of the atomic's object.

#### `AtomicInterface.load(self, memory='seq_cst', scope='system')` Function

Atomically returns the value of the atomic's object.

The size of the machine representation of the dtype of the atomic's object shall be less than or equal to 16 bytes. \[User Requirement\]

#### `AtomicInterface.store(self, val, memory='seq_cst', scope='system')` Function

Atomically sets the value of the atomic's object to `val`.

The size of the machine representation of the dtype of the atomic's object shall be less than or equal to 16 bytes. \[User Requirement\]

#### `AtomicInterface.exch(self, val, memory='seq_cst', scope='system')` Function

Atomically sets the value of the atomic's object to `val`.

Returns the value of the atomic's object before this operation.

The size of the machine representation of the dtype of the atomic's object shall be less than or equal to 8 bytes. \[User Requirement\]

#### `AtomicInterface.cas(self, old, val, memory='seq_cst', scope='system')` Function

Atomically performs `if (this == old) this = val`, where `this` is the value of the object.

Returns the value of the atomic's object before this operation.

The size of the machine representation of the dtype of the atomic's object shall be less than or equal to 8 bytes. \[User Requirement\]

#### `AtomicInterface.add(self, val, memory='seq_cst', scope='system')` Function

Atomically performs `this += val`, where `this` is the value of the object.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be `uint32`, `int32`, `uint64`, `int64`, `float32`, or `float64`. \[User Requirement\]

#### `AtomicInterface.sub(self, val, memory='seq_cst', scope='system')` Function

Atomically performs `this -= val`, where `this` is the value of the object.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be `uint32`, `int32`, `uint64`, `int64`, `float32`, or `float64`. \[User Requirement\]

#### `AtomicInterface.and_(self, val, memory='seq_cst', scope='system')` Function

Atomically performs `this &= val`, where `this` is the value of the object.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be `uint32`, `int32`, `uint64`, or `int64`. \[User Requirement\]

#### `AtomicInterface.or_(self, val, memory='seq_cst', scope='system')` Function

Atomically performs `this |= val`, where `this` is the value of the object.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be `uint32`, `int32`, `uint64`, or `int64`. \[User Requirement\]

#### `AtomicInterface.xor(self, val, memory='seq_cst', scope='system')` Function

Atomically performs `this ^= val`, where `this` is the value of the object.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be `uint32`, `int32`, `uint64`, or `int64`. \[User Requirement\]

#### `AtomicInterface.max(self, val, memory='seq_cst', scope='system')` Function

Atomically perform `this = max(this, val)`, where `this` is the value of the object.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be `uint32`, `int32`, `uint64`, `int64`, `float32`, or `float64`. \[User Requirement\]

#### `AtomicInterface.nanmax(self, val, memory='seq_cst', scope='system')` Function

Atomically perform `this = max(this, val)`, where `this` is the value of the object.

NaN is treated as a missing value. Example: `assert(AtomicInterface.nanmax(a, NaN) == a)`. Example: `a = NaN; assert(AtomicInterface.nanmax(a, n) == n)`.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be `uint32`, `int32`, `uint64`, `int64`, `float32`, or `float64`. \[User Requirement\]

#### `AtomicInterface.min(self, val, memory='seq_cst', scope='system')` Function

Atomically perform `this = min(this, val)`, where `this` is the value of the object.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be `uint32`, `int32`, `uint64`, `int64`, `float32`, or `float64`. \[User Requirement\]

#### `AtomicInterface.nanmin(self, val, memory='seq_cst', scope='system')` Function

Atomically perform `this = min(this, val)`, where `this` is the value of the object.

NaN is treated as a missing value. Example: `assert(AtomicInterface.nanmin(a, NaN) == a)`. Example: `a = NaN; assert(AtomicInterface.nanmin(a, n) == n)`.

Returns the value of the atomic's object before this operation.

The dtype of the atomic's object shall be `uint32`, `int32`, `uint64`, `int64`, `float32`, or `float64`. \[User Requirement\]

#### `threadfence(memory='seq_cst', scope='system')` Function

Establishes the specified memory synchronization ordering of non-atomic and relaxed accesses. 

Note: This is the equivalent of CUDA C++'s `__threadfence`, `__threadfence_block`, and `__threadfence_system`. 

### Thread Block Synchronization

The following functions shall be called by every thread in the thread block. \[User Requirement\]

`predicate` parameters shall be a function that is callable with no arguments. \[User Requirement\]

#### `syncthreads()` Function

Synchronizes with all threads in the same thread block.

#### `syncthreads_count(pred: Callable[[], bool]) -> int32` Function

Synchronizes with all threads in the same thread block.

Returns the number of threads in the thread block for which `predicate()` \== `True`.

#### `syncthreads_and(pred: Callable[[], bool]) -> bool` Function

Synchronizes with all threads in the same thread block.

Returns `True` if `predicate() == True` for all threads in the thread block.

#### `syncthreads_or(pred: Callable[[], bool]) -> bool` Function

Synchronizes with all threads in the same thread block.

Returns `True` if `predicate() == True` for any thread in the thread block.

### Thread Warp Synchronization

`predicate` parameters shall be a function that is callable with no arguments. \[User Requirement\]

#### `WarpMask` Class

A `WarpMask` object indicates which of the threads in a thread warp are participating in an operation.

`WarpMask` is a subclass of `int32`.

#### `WarpMask.__getitem__(self, i: int32) -> bool` Function

Returns `True` if thread ID `i` is set in the mask.

`i >= 0 and i < 32` \[User Requirement\]

#### `WarpMask.__setitem__(self, i: int32, val: bool)` Function

If `val == True`, add thread ID `i` to the mask, otherwise, remove thread ID `i` from the mask.

`i >= 0 and i < 32` \[User Requirement\]

#### `activemask() -> WarpMask` Function

Returns a mask of all the currently active threads in the calling warp.

#### `lanemask_lt() -> WarpMask` Function

Returns a mask of all threads (including inactive ones) with thread IDs less than the current lane (`lane_id`).

#### `syncwarp(mask: WarpMask)` Function

Synchronizes with all threads in `mask` within the thread warp.

#### `all_sync(mask: WarpMask, pred: Callable[[], bool]) -> bool` Function

Returns `True` if `pred() == True` for all threads in `mask` within the thread warp, and `False` otherwise.

Note: This operation does not guarantee any memory ordering.

#### `any_sync(mask: WarpMask, pred: Callable[[], bool]) -> bool` Function

Returns `True` if `pred() == True` for any threads in `mask` within the thread warp, and `False` otherwise.

Note: This operation does not guarantee any memory ordering.

#### `eq_sync(mask: WarpMask, pred: Callable[[], bool]) -> bool` Function

Returns `True` if `pred()` has the same value for all threads in `mask` within the thread warp, and `False` otherwise.

Note: This operation does not guarantee any memory ordering.

#### `ballot_sync(mask: WarpMask, pred : function) -> WarpMask` Function

Returns all threads in `mask` within the warp for which `pred() == True`.

Note: This operation does not guarantee any memory ordering.

#### `shfl_sync(mask: WarpMask, value, src_lane: int32)` Function

Returns `value` from thread `src_lane`.

`mask[src_lane] == True` \[User Requirement\]

`src_lane >= 0 and src_lane < 32` \[User Requirement\]

The size of the machine representation of `value` shall be less than or equal to 8 bytes. \[User Requirement\]

Note: This operation does not guarantee any memory ordering.

#### `shfl_up_sync(mask: WarpMask, value, delta: int32)` Function

Returns `value` from thread `lane_id - delta`.

`mask[lane_id - delta] == True` \[User Requirement\]

`src_lane >= 0 and src_lane < 32` \[User Requirement\]

The size of the machine representation of `value` shall be less than or equal to 8 bytes. \[User Requirement\]

Note: This operation does not guarantee any memory ordering.

#### `shfl_down_sync(mask: WarpMask, value, delta: int32)` Function

Returns `value` from thread `lane_id + delta`.

`mask[lane_id + delta] == True` \[User Requirement\]

`src_lane >= 0 and src_lane < 32` \[User Requirement\]

The size of the machine representation of `value` shall be less than or equal to 8 bytes. \[User Requirement\]

Note: This operation does not guarantee any memory ordering.

#### `shfl_xor_sync(mask: WarpMask, value, flag: WarpMask)` Function

Returns `value` from thread `lane_id ^ flag`.

`mask[lane_id ^ flag] == True` \[User Requirement\]

`src_lane >= 0 and src_lane < 32` \[User Requirement\]

The size of the machine representation of `value` shall be less than or equal to 8 bytes. \[User Requirement\]

Note: This operation does not guarantee any memory ordering.

#### `match_any_sync(mask: WarpMask, value, flag: int32) -> WarpMask` Function

Returns all threads in `mask` within the warp with the same `value` as the caller.

Note: This operation does not guarantee any memory ordering.

#### `match_all_sync(mask: WarpMask, value, flag: int32) -> tuple[WarpMask, bool]` Function

Returns a tuple of `(eq, pred)`, where `eq` is a mask of threads in `mask` that have the same value, and `pred` is `True` if all threads in the `mask` have the same value, and False otherwise.

Note: This operation does not guarantee any memory ordering.

### Numeric Intrinsics

#### `popc(x)` Function

Returns the number of bits set in `x`.

`x`'s type must be a heterogeneous integer type. \[User Requirement\]

#### `brev(x)` Function

Returns the reverse of the bit pattern of `x`.

`x`'s type must be a heterogeneous integer type. \[User Requirement\]

#### `clz(x)` Function

Returns the number of leading zeros in `x`.

`x`'s type must be a heterogeneous integer type. \[User Requirement\]

#### `ffs(x)` Function

Returns the position of the first (least significant) bit set in `x`, where the least significant bit position is 1\.

`x`'s type must be a heterogeneous integer type. \[User Requirement\]

#### `cbrt(a)` Function

Returns `a ** (1/3)`.

`a`'s type must be a heterogeneous floating point type. \[User Requirement\]

#### `fma(a, b, c)` Function

Returns `(a * b) + c`.

`a`, `b`, and `c`'s types must be heterogeneous floating point types. \[User Requirement\]

# References

[cuda.core](https://docs.google.com/document/d/1_aLAk9azax5zsaCbqh2LokFiBlTdBNovkWFUq3q4AfM/edit?usp=sharing)

[Python Array API Standard v2023.12](https://data-apis.org/array-api/2023.12/API_specification/type_promotion.html)

[DLPack](https://dmlc.github.io/dlpack/latest/python_spec.html)

[CUDA Array Interface Version 3](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html)

[NVC++ `if target`](https://docs.google.com/document/d/1BK7V_hS4-X35Ua9RzyQzRYDvXLR4Xh0T45Ke7uMsAsY/edit?tab=t.0#bookmark=id.41mghml)

[ISO House Style](https://www.iso.org/ISO-house-style.html)

[ISO Verbal Forms](https://www.iso.org/sites/directives/current/part2/index.xhtml#_idTextAnchor078)

Meeting Notes on [CUDA Python Device Guide](https://drive.google.com/drive/folders/1WoXX0s5hGaQUpzL9EPJuCk7yWK6wr6Oj)

