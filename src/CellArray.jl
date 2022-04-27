using StaticArrays, Adapt, CUDA, AMDGPU

Cell = Union{Number, SArray, FieldArray}

struct CellArray{T<:Cell,N,B,T_array<:AbstractArray} <: AbstractArray{T,N}
    data::T_array
    dims::NTuple{N,Int}

    function CellArray{T,N,B,T_array}(data::T_array, dims::NTuple{N,Int}) where {T<:Cell, N, B, T_array<:AbstractArray}
        check_T(T)
        celldims = size(T)  # Note: size must be defined for type T (as it is e.g. for StaticArrays)
        T_elem = eltype(T)  # Note: eltype must be defined for type T (as it is e.g. for StaticArrays)
        if (eltype(data) != T_elem)                     @IncoherentArgumentError("eltype(data) must match eltype(T).") end
        if (ndims(data) != 3)                           @ArgumentError("ndims(data) must be 3.") end
        if (size(data) != (B, prod(celldims), ceil(Int,prod(dims)/B))) @IncoherentArgumentError("size(data) must match (B, prod(size(T), ceil(prod(dims)/B)).") end
        new{T,N,B,T_array}(data, dims)
    end
    function CellArray{T,N,B}(data::T_array, dims::NTuple{N,Int}) where {T<:Cell, N, B, T_array<:AbstractArray}
        CellArray{T,N,B,T_array}(data, dims)
    end
    function CellArray{T,N,B,T_array}(dims::NTuple{N,Int}) where {T<:Cell, N, B, T_array<:AbstractArray}
        check_T(T)
        celldims = size(T)  # Note: size must be defined for type T (as it is e.g. for StaticArrays)
        T_elem = eltype(T)  # Note: eltype must be defined for type T (as it is e.g. for StaticArrays)
        data = T_array{T_elem,3}(undef, B, prod(celldims), ceil(Int,prod(dims)/B))
        CellArray{T,N,B,T_array}(data, dims)
    end
    function CellArray{T,B,T_array}(dims::NTuple{N,Int}) where {T<:Cell, N, B, T_array<:AbstractArray}
        CellArray{T,N,B,T_array}(dims)
    end
end

CPUCellArray{T,N,B} = CellArray{T,N,B,Array}
 CuCellArray{T,N,B} = CellArray{T,N,B,CuArray}
ROCCellArray{T,N,B} = CellArray{T,N,B,ROCArray}

CPUCellArray{T,B}(dims::NTuple{N,Int}) where {T<:Cell,N,B} = (check_T(T); CPUCellArray{T,N,B}(dims))
 CuCellArray{T,B}(dims::NTuple{N,Int}) where {T<:Cell,N,B} = (check_T(T); CuCellArray{T,N,B}(dims))
ROCCellArray{T,B}(dims::NTuple{N,Int}) where {T<:Cell,N,B} = (check_T(T); ROCCellArray{T,N,B}(dims))

CPUCellArray{T,B}(dims::Int...) where {T<:Cell,B} = (check_T(T); CPUCellArray{T,B}(dims))
 CuCellArray{T,B}(dims::Int...) where {T<:Cell,B} = (check_T(T); CuCellArray{T,B}(dims))
ROCCellArray{T,B}(dims::Int...) where {T<:Cell,B} = (check_T(T); ROCCellArray{T,B}(dims))

CPUCellArray{T}(dims::NTuple{N,Int}) where {T<:Cell,N} = (check_T(T); CPUCellArray{T,N,prod(dims)}(dims))
 CuCellArray{T}(dims::NTuple{N,Int}) where {T<:Cell,N} = (check_T(T); CuCellArray{T,N,prod(dims)}(dims))
ROCCellArray{T}(dims::NTuple{N,Int}) where {T<:Cell,N} = (check_T(T); ROCCellArray{T,N,prod(dims)}(dims))

CPUCellArray{T}(dims::Int...) where {T<:Cell} = (check_T(T); CPUCellArray{T}(dims))
 CuCellArray{T}(dims::Int...) where {T<:Cell} = (check_T(T); CuCellArray{T}(dims))
ROCCellArray{T}(dims::Int...) where {T<:Cell} = (check_T(T); ROCCellArray{T}(dims))

@inline function Base.similar(A::CellArray{T0,N0,B,T_array}, ::Type{T}, dims::NTuple{N,Int}) where {T0,N0,B,T_array,T<:Cell,N}
    CellArray{T,N,B,T_array}(dims)
end

@inline Base.IndexStyle(::Type{<:CellArray})                                         = IndexLinear()
@inline Base.size(T::Type{<:Number}, args...)                                        = 1
@inline Base.size(A::CellArray)                                                      = A.dims
@inline Base.getindex(A::CellArray{T,N,B}, i::Int) where {T <: Number,N,B}           = T(A.data[(i-1)%B+1, 1, (i-1)÷B+1])
@inline Base.getindex(A::CellArray{T,N,B}, i::Int) where {T,N,B}                     = T(getindex(A.data, (i-1)%B+1, j, (i-1)÷B+1) for j=1:length(T)) # NOTE:The same fails on GPU if convert is used.
@inline Base.setindex!(A::CellArray{T,N,B}, x::Number, i::Int) where {T<:Number,N,B} = (A.data[(i-1)%B+1, 1, (i-1)÷B+1] = x; return)
@inline Base.setindex!(A::CellArray{T,N,B}, X::T, i::Int) where {T,N,B}              = (for j=1:length(T) A.data[(i-1)%B+1, j, (i-1)÷B+1] = X[j] end; return)

@inline cellsize(A::AbstractArray)                                                   = size(eltype(A))
@inline cellsize(A::AbstractArray, i::Int)                                           = cellsize(A)[i]
@inline blocklength(A::CellArray{T,N,B}) where {T,N,B}                               = B

Adapt.adapt_structure(to, A::CellArray{T,N,B}) where {T,N,B}                         = CellArray{T,N,B}(adapt(to, A.data), A.dims)

function check_T(::Type{T}) where {T}
    if !isbitstype(T) @ArgumentError("the celltype, `T`, must be a bitstype.") end # Note: This test is required as FieldArray can be mutable and thus not bitstype (and ismutable() is for values not types...). The following tests would currently not be required as the current definition of the Cell type implies the tests to succeed.
    if !hasmethod(size, Tuple{Type{T}}) @ArgumentError("for the celltype, `T`, the following method must be defined: `@inline Base.size(T::Type{<:T}, args...)`") end
    if !hasmethod(eltype, Tuple{Type{T}}) @ArgumentError("for the celltype, `T`, the following method must be defined: `@inline Base.eltype(T::Type{<:T})`") end
    if !hasmethod(getindex, Tuple{Type{T}}) @ArgumentError("for the celltype, `T`, the following method must be defined: `@inline Base.getindex(X::T, i::Int)`") end
end

check_T(::Type{T}) where {T <: Number} = return
