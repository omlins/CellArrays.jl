import CellArrays
import CellArrays: CellArray, check_T, Cell, _N
import CUDA: CuArray

"""
    CuCellArray{T<:Cell,N,B,T_elem} <: AbstractArray{T,N} where Cell <: Union{Number, SArray, FieldArray}

`N`-dimensional CellArray with cells of type `T`, blocklength `B`, and `T_array` being a `CuArray` of element type `T_elem`: alias for `CellArray{T,N,B,CuArray{T_elem,CellArrays._N}}`.

--------------------------------------------------------------------------------

    CuCellArray{T,B}(undef, dims)
    CuCellArray{T}(undef, dims)

Construct an uninitialized `N`-dimensional `CellArray` containing `Cells` of type `T` which are stored in an array of kind `CuArray`.

See also: [`CellArray`](@ref), [`CPUCellArray`](@ref), [`ROCCellArray`](@ref)
"""
const CuCellArray{T,N,B,T_elem} = CellArray{T,N,B,CuArray{T_elem,_N}}

CuCellArray{T,B}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N,B} = (check_T(T); CuCellArray{T,N,B,eltype(T)}(undef, dims))
CuCellArray{T,B}(::UndefInitializer, dims::Int...) where {T<:Cell,B} = CuCellArray{T,B}(undef, dims)
CuCellArray{T}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N} = CuCellArray{T,0}(undef, dims)
CuCellArray{T}(::UndefInitializer, dims::Int...) where {T<:Cell} = CuCellArray{T}(undef, dims)


## AbstractArray methods

@inline function Base.similar(A::CuCellArray{T0,N0,B,T_elem0}, ::Type{T}, dims::NTuple{N,Int}) where {T0,N0,B,T_elem0,T<:Cell,N}
    CuCellArray{T,N,B,eltype(T)}(undef, dims)
end


## Some function to show that extensions are working in general

CellArrays.some_function(A::CuArray) = A .+ 1