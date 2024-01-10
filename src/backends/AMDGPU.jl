import CellArrays
import CellArrays: CellArray, check_T, Cell, _N
import AMDGPU: ROCArray

"""
    ROCCellArray{T<:Cell,N,B,T_elem} <: AbstractArray{T,N} where Cell <: Union{Number, SArray, FieldArray}

`N`-dimensional CellArray with cells of type `T`, blocklength `B`, and `T_array` being a `ROCArray` of element type `T_elem`: alias for `CellArray{T,N,B,ROCArray{T_elem,CellArrays._N}}`.

--------------------------------------------------------------------------------

    ROCCellArray{T,B}(undef, dims)
    ROCCellArray{T}(undef, dims)

Construct an uninitialized `N`-dimensional `CellArray` containing `Cells` of type `T` which are stored in an array of kind `ROCArray`.

See also: [`CellArray`](@ref), [`CPUCellArray`](@ref), [`CuCellArray`](@ref)
"""
const ROCCellArray{T,N,B,T_elem} = CellArray{T,N,B,ROCArray{T_elem,_N}}

ROCCellArray{T,B}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N,B} = (check_T(T); ROCCellArray{T,N,B,eltype(T)}(undef, dims))
ROCCellArray{T,B}(::UndefInitializer, dims::Int...) where {T<:Cell,B} = ROCCellArray{T,B}(undef, dims)
ROCCellArray{T}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N} = ROCCellArray{T,0}(undef, dims)
ROCCellArray{T}(::UndefInitializer, dims::Int...) where {T<:Cell} = ROCCellArray{T}(undef, dims)


## AbstractArray methods

@inline function Base.similar(A::ROCCellArray{T0,N0,B,T_elem0}, ::Type{T}, dims::NTuple{N,Int}) where {T0,N0,B,T_elem0,T<:Cell,N}
    ROCCellArray{T,N,B,eltype(T)}(undef, dims)
end


## Some function to show that extensions are working in general

CellArrays.some_function(A::ROCArray) = A .+ 1