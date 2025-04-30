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