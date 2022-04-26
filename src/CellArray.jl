using StaticArrays, Adapt, CUDA, AMDGPU

Cell = Union{Number, SArray, FieldArray}

struct CellArray{T<:Cell,N,T_array<:AbstractArray} <: AbstractArray{T,N}
    data::T_array
    dims::NTuple{N,Int}
    blocklen::Int

    function CellArray{T,N,T_array}(data::T_array, dims::NTuple{N,Int}, blocklen::Int) where {T<:Cell, N, T_array<:AbstractArray}
        check_T(T)
        celldims = size(T)  # Note: size must be defined for type T (as it is e.g. for StaticArrays)
        T_elem = eltype(T)  # Note: eltype must be defined for type T (as it is e.g. for StaticArrays)
        if (eltype(data) != T_elem)                     @IncoherentArgumentError("eltype(data) must match eltype(T).") end
        if (ndims(data) != 3)                           @ArgumentError("ndims(data) must be 3.") end
        if (size(data) != (blocklen, prod(celldims), ceil(Int,prod(dims)/blocklen))) @IncoherentArgumentError("size(data) must match (blocklen, prod(size(T), ceil(prod(dims)/blocklen)).") end
        new{T,N,T_array}(data, dims)
    end
    function CellArray{T,N}(data::T_array, dims::NTuple{N,Int}, blocklen::Int) where {T<:Cell, N, T_array<:AbstractArray}
        CellArray{T,N,T_array}(data, dims)
    end
    function CellArray{T,N}(dims::NTuple{N,Int}, blocklen::Int, T_array::Type{<:AbstractArray}) where {T<:Cell, N}
        check_T(T)
        celldims = size(T)  # Note: size must be defined for type T (as it is e.g. for StaticArrays)
        T_elem = eltype(T)  # Note: eltype must be defined for type T (as it is e.g. for StaticArrays)
        data = T_array{T_elem,3}(undef, blocklen, prod(celldims), ceil(Int,prod(dims)/blocklen))
        CellArray{T,N,T_array}(data, dims, blocklen)
    end
    function CellArray{T}(dims::NTuple{N,Int}, blocklen::Int, T_array::Type{<:AbstractArray}) where {T<:Cell, N}
        CellArray{T,N}(dims, blocklen, T_array)
    end
end

CuCellArray{T,N}  = CellArray{T,N,CuArray}
ROCCellArray{T,N} = CellArray{T,N,ROCArray}

CellArray(::Type{T}, dims::NTuple{N,Int}, blocklen=prod(dims)) where {T<:Cell,N}    = (check_T(T); CellArray{T,N}(dims, blocklen, Array))
CellArray(::Type{T}, dims::Int...) where {T<:Cell}                                  = (check_T(T); CellArray(T, dims))
CuCellArray(::Type{T}, dims::NTuple{N,Int}, blocklen=prod(dims)) where {T<:Cell,N}  = (check_T(T); CellArray{T,N}(dims, blocklen, CuArray))
CuCellArray(::Type{T}, dims::Int...) where {T<:Cell}                                = (check_T(T); CuCellArray(T, dims))
ROCCellArray(::Type{T}, dims::NTuple{N,Int}, blocklen=prod(dims)) where {T<:Cell,N} = (check_T(T); CellArray{T,N}(dims, blocklen, ROCArray))
ROCCellArray(::Type{T}, dims::Int...) where {T<:Cell}                               = (check_T(T); ROCCellArray(T, dims))

@inline function Base.similar(A::CellArray{T0,N0,T_array}, ::Type{T}, dims::NTuple{N,Int}) where {T0,N0,T_array,T<:Cell,N}
    blocklen = (dims==A.dims) ? A.blocklen : prod(dims)  # Note: if the `dims` of the new array are to be the same as those of the old array, then the same blocklength as in the old array are used; else, the default blocklength is used.
    CellArray{T,N}(dims, blocklen, T_array)
end

@inline Base.IndexStyle(::Type{<:CellArray})                                 = IndexLinear()
@inline Base.size(T::Type{<:Number}, args...)                                = 1
@inline Base.size(A::CellArray)                                              = A.dims
@inline Base.getindex(A::CellArray{T}, i::Int) where {T <: Number}           = T(A.data[i%A.blocklen, 1, i÷A.blocklen])
@inline Base.getindex(A::CellArray{T}, i::Int) where {T}                     = T(getindex(A.data, i%A.blocklen, j, i÷A.blocklen) for j=1:length(T)) # NOTE:The same fails on GPU if convert is used.
@inline Base.setindex!(A::CellArray{T}, x::Number, i::Int) where {T<:Number} = (A.data[i%A.blocklen, 1, i÷A.blocklen] = x; return)
@inline Base.setindex!(A::CellArray{T}, X::T, i::Int) where {T}              = (for j=1:length(T) A.data[i%A.blocklen, j, i÷A.blocklen] = X[j] end; return)

@inline cellsize(A::AbstractArray)                                           = size(eltype(A))
@inline cellsize(A::AbstractArray, i::Int)                                   = cellsize(A)[i]
@inline blocklength(A::CellArray)                                            = A.blocklen

Adapt.adapt_structure(to, A::CellArray{T,N}) where {T,N}                     = CellArray{T,N}(adapt(to, A.data), A.dims, A.blocklen)

function check_T(::Type{T}) where {T}
    if !isbitstype(T) @ArgumentError("the celltype, `T`, must be a bitstype.") end # Note: This test is required as FieldArray can be mutable and thus not bitstype (and ismutable() is for values not types...). The following tests would currently not be required as the current definition of the Cell type implies the tests to succeed.
    if !hasmethod(size, Tuple{Type{T}}) @ArgumentError("for the celltype, `T`, the following method must be defined: `@inline Base.size(T::Type{<:T}, args...)`") end
    if !hasmethod(eltype, Tuple{Type{T}}) @ArgumentError("for the celltype, `T`, the following method must be defined: `@inline Base.eltype(T::Type{<:T})`") end
    if !hasmethod(getindex, Tuple{Type{T}}) @ArgumentError("for the celltype, `T`, the following method must be defined: `@inline Base.getindex(X::T, i::Int)`") end
end

check_T(::Type{T}) where {T <: Number} = return
