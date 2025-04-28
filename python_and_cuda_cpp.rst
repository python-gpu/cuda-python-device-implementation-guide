Python and CUDA C++
===================

This section defines interoperability and bindings between Python and
CUDA C++.

Machine Representation
----------------------

Python frameworks execute Python code on devices by translating the
Python code into a *machine representation* that can be executed by CUDA
devices.

Likewise, CUDA C++ evaluates CUDA C++ code on devices by translating the
CUDA C++ code into a machine representation.

``machine_representation() -> str`` Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns an unspecified description of the machine representation format.

Example: ``"itanium"`` on Linux, ``"msvc"`` on Windows.

Interoperable Functions
-----------------------

Python and CUDA C++ interoperate together by calling certain
*interoperable functions* defined by the other language.

Python interoperable functions shall have the same symbol and calling
convention as the machine representation of an
``extern "C" __host__ __device__`` CUDA C++ function of the same name.

The machine representation of a Python interoperable function shall take
the machine representation of each of its parameters as if they were
passed by value in CUDA C++.

This document defines requirements for the machine representation of
types and objects when they are taken as parameters to and returned from
interoperable functions. These requirements only apply when calling
interoperable functions.

