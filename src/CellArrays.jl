"""
Module CellArrays

Provides support for an AbstractArray subtype `CellArray`, which represents arrays with cells that can contain logical arrays instead of single values; the data is stored in an optimal fashion for GPU HPC applications.

# General overview and examples
https://github.com/omlins/CellArray.jl
"""
module CellArrays

## Alphabetical include of submodules.
include("Exceptions.jl")
using .Exceptions

## Alphabetical include of function/data type definition files
include("CellArray.jl")

## Exports (need to be after include of submodules if re-exports from them)
export CellArray, CPUCellArray, CuCellArray, ROCCellArray, cellsize, blocklength
end
