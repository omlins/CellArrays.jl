"""
Module CellArrays

Provides support for an AbstractArray subtype `CellArray`, which represents arrays with cells that can contain logical arrays instead of single values; the data is stored in an optimal fashion for GPU HPC applications.

# General overview and examples
https://github.com/omlins/CellArray.jl
"""
module CellArrays

include("CellArray.jl")

export CellArray, CuCellArray, cellsize
end
