# Usage
Have a look at the [Examples](@ref) and see the [API reference](@ref) for details on the usage of `CellArrays`.

!!! tip "Hint"
    [ParallelStencil.jl] enables straightforward working with `CellArrays`. It automatically allocates [`CellArray`](@ref)s when the keyword arguments `celldims` or `celltype` are given to the architecture-agnostic allocation macros `@zeros`, `@ones`, `@rand`, `@falses`, `@trues` and `@fill` (refer to the documentation of [ParallelStencil.jl] for more details).


## Installation
`CellArrays` can be installed directly with the Julia package manager from the REPL:
```julia-repl
julia>]
  pkg> add CellArrays
```
