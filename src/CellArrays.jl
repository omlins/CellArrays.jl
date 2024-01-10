"""
Module CellArrays

Provides support for an AbstractArray subtype `CellArray`, which represents arrays with cells that can contain logical arrays instead of single values; the data is stored in an optimal fashion for GPU HPC applications.

# General overview and examples
https://github.com/omlins/CellArray.jl

# Constructors
- [`CellArray`](@ref)
- [`CPUCellArray`](@ref)
- [`CuCellArray`](@ref)
- [`ROCCellArray`](@ref)

# Functions (additional to standard AbstractArray functionality)
- [`cellsize`](@ref)
- [`blocklength`](@ref)

To see a description of a constructor or function type `?<constructorname>` or `?<functionname>`, respectively.

!!! note "Performance note"
    If a CellArray's cells contain more than what fits into registers, the performance on Nvidia GPUs will deviate from the optimal, if access is not performed by accessing the cells' values individually to reduce register pressure.
"""
module CellArrays

## Alphabetical include of submodules.
include("Exceptions.jl")
using .Exceptions

## Alphabetical include of function/data type definition files
include("CellArray.jl")

## Exports (need to be after include of submodules if re-exports from them)
export CellArray, CPUCellArray, cellsize, blocklength, field # Exported in extensions: CuCellArray, ROCCellArray # Not working!

## Some function to show that extensions are working in general (relying on multiple dispatch).
some_function(args...) = @ArgumentError("Required extension not loaded. Import CUDA/AMDGPU before using CellArrays with GPU features.")
export some_function

end
