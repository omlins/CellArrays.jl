module CellArrays_CUDAExt
    include(joinpath(@__DIR__, "..", "src", "backends", "CUDA.jl"))
    export CuCellArray
end # module