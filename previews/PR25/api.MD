```@meta
CurrentModule = CellArrays
```

# API reference

This is the official API reference of `CellArrays`. Note that it can also be queried interactively from the [Julia REPL] using the [help mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Help-mode):
```julia-repl
julia> using CellArrays
julia>?
help?> CellArrays
```


## [`CellArray`](@ref) type and basic constructors
```@docs
CellArray
```


## Convenience type aliases and constructors
#### Index
* [`CellArrays.CPUCellArray`](@ref)
* [`CellArrays.CuCellArray`](@ref)
* [`CellArrays.ROCCellArray`](@ref)

#### Documentation
```@autodocs
Modules = [CellArrays]
Order   = [:type]
Filter = t -> typeof(t) !== CellArray
```


## Functions -- additional to standard `AbstractArray` functionality
#### Index
```@index
Order = [:function]
```
#### Documentation
```@autodocs
Modules = [CellArrays]
Order   = [:function]
```
