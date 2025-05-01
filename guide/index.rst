.. toctree::
   :hidden:

   self
   python_and_cuda_cpp
   heterogeneous_programming
   simt_programming
   references

Introduction
============

Scope, Motivation, and Goals
----------------------------

This document defines how to write SIMT code for CUDA devices in Python,
how *Python frameworks* (Python libraries, compilers, interpreters,
domain specific languages) evaluate such *device code* on the CUDA
platform, and how such frameworks interact with each other and with CUDA
C++.

This specification has been developed to:

-  Adoption: Encourage the use of CUDA parallelism in Python.
-  Assurance: Give users confidence that CUDA parallelism in Python can
   be relied on and will be supported.
-  Leadership: Establish trust, respect, and influence for NVIDIA in the
   Python community.
-  Consistency: Provide a uniform and coherent experience across
   different Python frameworks.

We want to make CUDA Python programming:

-  Easy.
-  Performant.
-  Modern.
-  Pythonic.

We want to enable:

-  Source Portability: Use the same Python and CUDA C++ source code with
   different Python frameworks without rewriting.

   -  Example: Compile the same Python code with Numba and CuPy.
   -  Example: Use the same Python user-defined type and reduction
      operator with Numba and Warp.
   -  Example: Write Python bindings once that will support multiple
      frameworks for a CUDA C++ library that contains host code, device
      code, and C++ templates.

-  Interoperability: Use different Python frameworks and/or CUDA C++
   together, either within a single kernel (intra-kernel) or between
   different kernels (inter-kernel).

   -  Example: Launch a Numba kernel followed by a Warp kernel on the
      same data.
   -  Example: Call a Python user-defined load operation in CUB or
      cuFFTDx.

Conformance
-----------

This document defines two kinds of requirements:

-  Framework Requirements: Requirements on the behavior of Python
   frameworks.
-  User Requirements: Requirements on Python source code. If such a
   requirement is violated, the program is ill-formed, and there is a
   framework requirement to produce an error. Such requirements are
   explicitly annotated in this document.

The first framework requirement is to implement the entities and
semantics defined in this document.

This document places no requirements on the structure of Python
frameworks. Conforming frameworks are only required to emulate the
observable behavior defined by this document. A framework is free to
disregard any requirement of this document as long as the result is as
if the requirement had been obeyed, as far as can be determined from the
observable behavior of the program.

General Requirements
--------------------

All entities defined in this document shall be in the ``cuda.device``
namespace.

``core.`` entities referenced in this document are from ``cuda.core``.

When calling any function defined in this document, the type of an
argument shall match the type hint of the corresponding parameter if it
has one. [User Requirement]

*Exposition only* entities are only described to help define other
entities.

Exposition only entities are not part of the interface and shall not be
used. [User Requirement] 