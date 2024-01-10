module CellArrays_AMDGPUExt
    include(joinpath(@__DIR__, "..", "src", "backends", "AMDGPU.jl"))
    export ROCCellArray
end