# [CellArrays.jl](https://github.com/omlins/CellArrays.jl) [![Star on GitHub](https://img.shields.io/github/stars/omlins/CellArrays.jl.svg)](https://github.com/omlins/CellArrays.jl/stargazers)
The package `CellArrays` provides support for an `AbstractArray` subtype [`CellArray`](@ref), which represents arrays with cells that can contain logical arrays instead of single values; the data is stored in an optimal fashion for GPU HPC applications.

## Dependencies
`CellArrays` relies on [StaticArrays.jl], [Adapt.jl] and the Julia GPU packages [CUDA.jl] and [AMDGPU.jl].
