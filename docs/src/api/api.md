```@meta
CurrentModule = CellArrays
```

# API reference

## `CellArray` type and basic constructors
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
