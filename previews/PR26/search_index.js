var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API reference","title":"API reference","text":"CurrentModule = CellArrays","category":"page"},{"location":"api/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"This is the official API reference of CellArrays. Note that it can also be queried interactively from the Julia REPL using the help mode:","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"julia> using CellArrays\njulia>?\nhelp?> CellArrays","category":"page"},{"location":"api/#[CellArray](@ref)-type-and-basic-constructors","page":"API reference","title":"CellArray type and basic constructors","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"CellArray","category":"page"},{"location":"api/#CellArrays.CellArray","page":"API reference","title":"CellArrays.CellArray","text":"CellArray{T<:Cell,N,B,T_array} <: AbstractArray{T,N} where Cell <: Union{Number, SArray, FieldArray}\n\nN-dimensional array with elements of type T, where T are Cells of type Number, SArray or FieldArray. B defines the blocklength, which refers to the amount of values of a same Cell field that are stored contigously (B=1 means array of struct like storage; B=prod(dims) means array struct of array like storage; B=0 is an alias for B=prod(dims), enabling better peformance thanks to more specialized dispatch). T_array defines the array type used for storage.\n\n\n\nCellArray{T,N,B}(T_arraykind, undef, dims)\nCellArray{T,B}(T_arraykind, undef, dims)\nCellArray{T}(T_arraykind, undef, dims)\n\nConstruct an uninitialized N-dimensional CellArray containing Cells of type T which are stored in an array of kind T_arraykind.\n\nnote: Performance note\nBest performance on GPUs is in general obtained with B=0 as set by default. B=1 migth give better performance in certain cases. Other values of B do with the current implementation not lead to optimal performance on GPU.\n\nSee also: CPUCellArray, CuCellArray, ROCCellArray\n\n\n\n\n\n","category":"type"},{"location":"api/#Convenience-type-aliases-and-constructors","page":"API reference","title":"Convenience type aliases and constructors","text":"","category":"section"},{"location":"api/#Index","page":"API reference","title":"Index","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"CellArrays.CPUCellArray\nCellArrays.CuCellArray\nCellArrays.ROCCellArray","category":"page"},{"location":"api/#Documentation","page":"API reference","title":"Documentation","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [CellArrays]\nOrder   = [:type]\nFilter = t -> typeof(t) !== CellArray","category":"page"},{"location":"api/#Functions-–-additional-to-standard-AbstractArray-functionality","page":"API reference","title":"Functions – additional to standard AbstractArray functionality","text":"","category":"section"},{"location":"api/#Index-2","page":"API reference","title":"Index","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Order = [:function]","category":"page"},{"location":"api/#Documentation-2","page":"API reference","title":"Documentation","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [CellArrays]\nOrder   = [:function]","category":"page"},{"location":"api/#CellArrays.blocklength-Union{Tuple{CellArray{T, N, B, T_array}}, Tuple{T_array}, Tuple{B}, Tuple{N}, Tuple{T}} where {T, N, B, T_array}","page":"API reference","title":"CellArrays.blocklength","text":"blocklength(A)\n\nReturn the blocklength of CellArray A.\n\n\n\n\n\n","category":"method"},{"location":"api/#CellArrays.cellsize-Tuple{AbstractArray}","page":"API reference","title":"CellArrays.cellsize","text":"cellsize(A)\ncellsize(A, dim)\n\nReturn a tuple containing the dimensions of A or return only a specific dimension, specified by dim.\n\n\n\n\n\n","category":"method"},{"location":"api/#CellArrays.field-Union{Tuple{T_array}, Tuple{N}, Tuple{T}, Tuple{CellArray{T, N, 0, T_array}, Int64}} where {T, N, T_array}","page":"API reference","title":"CellArrays.field","text":"field(A, indices)\n\nReturn an array view of the field of CellArray A designated with indices (modifying the view will modify A). The view's dimensionality and size are equal to A's. The operation is not supported if parameter B of A is neither 0 nor 1.\n\nArguments\n\nindices::Int|NTuple{N,Int}: the indices that designate the field in accordance with A's cell type.\n\n\n\n\n\n","category":"method"},{"location":"usage/#Usage","page":"Usage","title":"Usage","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"Have a look at the Examples and see the API reference for details on the usage of CellArrays.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"tip: Hint\nParallelStencil.jl enables straightforward working with CellArrays. It automatically allocates CellArrays when the keyword arguments celldims or celltype are given to the architecture-agnostic allocation macros @zeros, @ones, @rand, @falses, @trues and @fill (refer to the documentation of ParallelStencil.jl for more details).","category":"page"},{"location":"usage/#Installation","page":"Usage","title":"Installation","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"CellArrays can be installed directly with the Julia package manager from the Julia REPL:","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"julia>]\n  pkg> add CellArrays","category":"page"},{"location":"examples/#Examples","page":"...","title":"Examples","text":"","category":"section"},{"location":"examples/","page":"...","title":"...","text":"Pages = [\"examples/memcopyCellArray3D.md\"]","category":"page"},{"location":"examples/","page":"...","title":"...","text":"Pages = [\"examples/memcopyCellArray3D_ParallelStencil.md\"]","category":"page"},{"location":"examples/memcopyCellArray3D_ParallelStencil/#Memory-copy-of-[CellArray](@ref)s-with-4x4-cells-using-[ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)","page":"Memory copy of CellArrays with 4x4 cells using ParallelStencil.jl","title":"Memory copy of CellArrays with 4x4 cells using ParallelStencil.jl","text":"","category":"section"},{"location":"examples/memcopyCellArray3D_ParallelStencil/","page":"Memory copy of CellArrays with 4x4 cells using ParallelStencil.jl","title":"Memory copy of CellArrays with 4x4 cells using ParallelStencil.jl","text":"Main.mdinclude(joinpath(Main.EXAMPLEROOT, \"memcopyCellArray3D_ParallelStencil.jl\"))","category":"page"},{"location":"examples/memcopyCellArray3D_ParallelStencil/","page":"Memory copy of CellArrays with 4x4 cells using ParallelStencil.jl","title":"Memory copy of CellArrays with 4x4 cells using ParallelStencil.jl","text":"The corresponding file can be found here.","category":"page"},{"location":"examples/memcopyCellArray3D/#Memory-copy-of-[CuCellArray](@ref)s-with-4x4-SMatrix-cells","page":"Memory copy of CuCellArrays with 4x4 SMatrix cells","title":"Memory copy of CuCellArrays with 4x4 SMatrix cells","text":"","category":"section"},{"location":"examples/memcopyCellArray3D/","page":"Memory copy of CuCellArrays with 4x4 SMatrix cells","title":"Memory copy of CuCellArrays with 4x4 SMatrix cells","text":"Main.mdinclude(joinpath(Main.EXAMPLEROOT, \"memcopyCellArray3D.jl\"))","category":"page"},{"location":"examples/memcopyCellArray3D/","page":"Memory copy of CuCellArrays with 4x4 SMatrix cells","title":"Memory copy of CuCellArrays with 4x4 SMatrix cells","text":"The corresponding file can be found here.","category":"page"},{"location":"#[CellArrays.jl](https://github.com/omlins/CellArrays.jl)-[![Star-on-GitHub](https://img.shields.io/github/stars/omlins/CellArrays.jl.svg)](https://github.com/omlins/CellArrays.jl/stargazers)","page":"Introduction","title":"CellArrays.jl (Image: Star on GitHub)","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"The package CellArrays provides support for an AbstractArray subtype CellArray, which represents arrays with cells that can contain logical arrays instead of single values; the data is stored in an optimal fashion for GPU HPC applications.","category":"page"},{"location":"#Dependencies","page":"Introduction","title":"Dependencies","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"CellArrays relies on StaticArrays.jl, Adapt.jl and the Julia GPU packages CUDA.jl and AMDGPU.jl.","category":"page"},{"location":"#Contributors","page":"Introduction","title":"Contributors","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"The principal contributors to CellArrays.jl are (ordered by the significance of the relative contributions):","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Dr. Samuel Omlin (@omlins), CSCS - Swiss National Supercomputing Centre, ETH Zurich","category":"page"}]
}
