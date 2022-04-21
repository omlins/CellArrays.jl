using Adapt, CUDA

struct CellArray{T,N,T_array<:AbstractArray} <: AbstractArray{T,N}
    data::T_array
    dims::NTuple{N,Int}

    function CellArray{T,N}(data::T_array, dims::NTuple{N,Int}) where {T,N} where {T_array<:AbstractArray}
        celldims = size(T)  # Note: size must be defined for type T (as it is e.g. for StaticArrays)
        T_elem = eltype(T)  # Note: eltype must be defined for type T (as it is e.g. for StaticArrays)
        if (eltype(data) != T_elem)                     @IncoherentArgumentError("eltype(data) must match eltype(T).") end
        if (ndims(data) != 2)                           @ArgumentError("ndims(data) must be 2.") end
        if (size(data) != (prod(dims), prod(celldims))) @IncoherentArgumentError("size(data) must match (prod(dims), prod(size(T))).") end
        new{T,N,T_array}(data, dims)
    end
    function CellArray{T,N}(dims::NTuple{N,Int}, T_array::Type{<:AbstractArray}) where {T,N}
        celldims = size(T)  # Note: size must be defined for type T (as it is e.g. for StaticArrays)
        T_elem = eltype(T)  # Note: eltype must be defined for type T (as it is e.g. for StaticArrays)
        data = T_array{T_elem,2}(undef, prod(dims), prod(celldims))
        new{T,N,T_array}(data, dims)
    end
    function CellArray{T}(dims::NTuple{N,Int}, T_array::Type{<:AbstractArray}) where {T,N}
        CellArray{T,N}(dims, T_array)
    end
end

CellArray(::Type{T}, dims::NTuple{N,Int}) where {T,N}                   = CellArray{T,N}(dims, Array)
CellArray(::Type{T}, dims::Int...) where {T}                            = CellArray{T,N}(dims)
CuCellArray(::Type{T}, dims::NTuple{N,Int}) where {T,N}                 = CellArray{T,N}(dims, CuArray)
CuCellArray(::Type{T}, dims::Int...) where {T}                          = CuCellArray{T,N}(dims)

@inline Base.IndexStyle(::Type{<:CellArray})                            = IndexLinear()
@inline Base.size(T::Type{<:Number}, args...   )                        = 1
@inline Base.size(A::CellArray)                                         = A.dims
@inline Base.getindex(A::CellArray{T}, i::Int) where {T <: Number}      = T(A.data[i,1])
@inline Base.getindex(A::CellArray{T}, i::Int) where {T}                = T(getindex(A.data, i, j) for j=1:length(T)) # NOTE:The same fails on GPU if convert is used.
@inline Base.setindex!(A::CellArray{T}, x::T, i::Int) where {T<:Number} = (A.data[i] = x; return)
@inline Base.setindex!(A::CellArray{T}, X::T, i::Int) where {T}         = (for j=1:length(T) A.data[i,j] = X[j] end; return)

@inline cellsize(A::AbstractArray)                                      = size(eltype(A))
@inline cellsize(A::AbstractArray, i::Int)                              = cellsize(A)[i]

Adapt.adapt_structure(to, A::CellArray{T,N}) where {T,N}                = CellArray{T,N}(adapt(to, A.data), A.dims)
