using StaticArrays, Adapt, CUDA, AMDGPU


## Constants

const _N = 3
const Cell = Union{Number, SArray, FieldArray}


## Types and constructors

"""
    CellArray{T<:Cell,N,B,T_array} <: AbstractArray{T,N} where Cell <: Union{Number, SArray, FieldArray}

`N`-dimensional array with elements of type `T`, where `T` are `Cells` of type Number, SArray or FieldArray. `B` defines the blocklength, which refers to the amount of values of a same `Cell` component that are stored contigously (`B=1` means array of struct like storage; `B=prod(dims)` means array struct of array like storage; `B=0` is an alias for `B=prod(dims)`, enabling better peformance thanks to more specialized dispatch). `T_array` defines the array type used for storage.

--------------------------------------------------------------------------------

    CellArray{T,N,B}(T_arraykind, undef, dims)
    CellArray{T,B}(T_arraykind, undef, dims)
    CellArray{T}(T_arraykind, undef, dims)

Construct an uninitialized `N`-dimensional `CellArray` containing `Cells` of type `T` which are stored in an array of kind `T_arraykind`.

--------------------------------------------------------------------------------

    CPUCellArray{T,B}(undef, dims)
    CPUCellArray{T}(undef, dims)

    CuCellArray{T,B}(undef, dims)
    CuCellArray{T}(undef, dims)

    ROCCellArray{T,B}(undef, dims)
    ROCCellArray{T}(undef, dims)

Construct an uninitialized `N`-dimensional `CellArray` containing `Cells` of type `T` which are stored in an array of kind `Array`, `CuArray` or `ROCArray` depending on the constructor chosen (`CPUCellArray` or `CuCellArray` or `ROCCellArray`) .

!!! note "Performance note"
    Best performance on GPUs is in general obtained with `B=0` as set by default. `B=1` migth give better performance in certain cases. Other values of `B` do with the current implementation not lead to optimal performance on GPU.
"""
struct CellArray{T<:Cell,N,B,T_array<:AbstractArray{T_elem,_N} where {T_elem}} <: AbstractArray{T,N}
    data::T_array
    dims::NTuple{N,Int}

    function CellArray{T,N,B,T_array}(data::T_array, dims::NTuple{N,Int}) where {T<:Cell,N,B,T_array<:AbstractArray{T_elem,_N}} where {T_elem}
        check_T(T)
        celldims = size(T)  # Note: size must be defined for type T (as it is e.g. for StaticArrays)
        blocklen = (B == 0) ? prod(dims) : B
        if (eltype(data) != eltype(T))                                               @IncoherentArgumentError("eltype(data) must match eltype(T).") end
        if (ndims(data) != _N)                                                       @ArgumentError("ndims(data) must be $_N.") end
        if (size(data) != (blocklen, prod(celldims), ceil(Int,prod(dims)/blocklen))) @IncoherentArgumentError("size(data) must match (blocklen, prod(size(T), ceil(prod(dims)/blocklen)).") end
        new{T,N,B,T_array}(data, dims)
    end

    function CellArray{T,N,B}(data::T_array, dims::NTuple{N,Int}) where {T<:Cell,N,B,T_array<:AbstractArray{T_elem,_N}} where {T_elem}
        CellArray{T,N,B,T_array}(data, dims)
    end

    function CellArray{T,N,B,T_array}(::Type{T_array}, ::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N,B,T_array<:AbstractArray{T_elem,_N}} where {T_elem} #where {Type{T_array}<:DataType}
        check_T(T)
        if (T_elem != eltype(T)) @IncoherentArgumentError("T_elem must match eltype(T).") end
        celldims = size(T)  # Note: size must be defined for type T (as it is e.g. for StaticArrays)
        blocklen = (B == 0) ? prod(dims) : B
        data = T_array(undef, blocklen, prod(celldims), ceil(Int,prod(dims)/blocklen))
        CellArray{T,N,B,T_array}(data, dims)
    end

    function CellArray{T,N,B,T_array}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N,B,T_array<:AbstractArray{T_elem,_N}} where {T_elem} #where {Type{T_array}<:DataType}
        CellArray{T,N,B,T_array}(T_array, undef, dims)
    end

    function CellArray{T,N,B}(::Type{T_array}, ::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N,B,T_array<:AbstractArray{T_elem,_N}} where {T_elem} #where {Type{T_array}<:DataType}
        CellArray{T,N,B,T_array}(T_array, undef, dims)
    end

    function CellArray{T,B}(::Type{T_array}, ::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N,B,T_array<:AbstractArray{T_elem,_N}} where {T_elem} #where {Type{T_array}<:DataType}
        CellArray{T,N,B}(T_array, undef, dims)
    end

    function CellArray{T,N,B}(::Type{T_arraykind}, ::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N,B,T_arraykind<:AbstractArray} #where {Type{T_arraykind}<:UnionAll}
        CellArray{T,N,B}(T_arraykind{eltype(T),_N}, undef, dims)
    end

    function CellArray{T,B}(::Type{T_arraykind}, ::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N,B,T_arraykind<:AbstractArray} #where {Type{T_arraykind}<:UnionAll}
        CellArray{T,N,B}(T_arraykind, undef, dims)
    end

    function CellArray{T,B}(::Type{T_arraykind}, ::UndefInitializer, dims::Int...) where {T<:Cell,B,T_arraykind<:AbstractArray} #where {Type{T_arraykind}<:UnionAll}
        CellArray{T,B}(T_arraykind, undef, dims)
    end

    function CellArray{T}(::Type{T_arraykind}, ::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N,T_arraykind<:AbstractArray} #where {Type{T_arraykind}<:UnionAll}
        CellArray{T,0}(T_arraykind, undef, dims)
    end

    function CellArray{T}(::Type{T_arraykind}, ::UndefInitializer, dims::Int...) where {T<:Cell,N,T_arraykind<:AbstractArray} #where {Type{T_arraykind}<:UnionAll}
        CellArray{T}(T_arraykind, undef, dims)
    end
end


Adapt.adapt_structure(to, A::CellArray{T,N,B,T_array}) where {T,N,B,T_array} = CellArray{T,N,B}(adapt(to, A.data), A.dims)

"""
    CPUCellArray{T<:Cell,N,B,T_elem} <: AbstractArray{T,N} where Cell <: Union{Number, SArray, FieldArray}

`N`-dimensional CellArray with cells of type `T`, blocklength `B`, and `T_array` being an `Array` of element type `T_elem`: alias for CellArray{T,N,B,Array{T_elem,CellArrays._N}}.
"""
CPUCellArray{T,N,B,T_elem} = CellArray{T,N,B,Array{T_elem,_N}}

"""
    CuCellArray{T<:Cell,N,B,T_elem} <: AbstractArray{T,N} where Cell <: Union{Number, SArray, FieldArray}

`N`-dimensional CellArray with cells of type `T`, blocklength `B`, and `T_array` being a `CuArray` of element type `T_elem`: alias for CellArray{T,N,B,CuArray{T_elem,CellArrays._N}}.
"""
CuCellArray{T,N,B,T_elem}  = CellArray{T,N,B,CuArray{T_elem,_N}}

"""
    ROCCellArray{T<:Cell,N,B,T_elem} <: AbstractArray{T,N} where Cell <: Union{Number, SArray, FieldArray}

`N`-dimensional CellArray with cells of type `T`, blocklength `B`, and `T_array` being a `ROCArray` of element type `T_elem`: alias for CellArray{T,N,B,ROCArray{T_elem,CellArrays._N}}.
"""
ROCCellArray{T,N,B,T_elem} = CellArray{T,N,B,ROCArray{T_elem,_N}}


CPUCellArray{T,B}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N,B} = (check_T(T); CPUCellArray{T,N,B,eltype(T)}(undef, dims))
 CuCellArray{T,B}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N,B} = (check_T(T); CuCellArray{T,N,B,eltype(T)}(undef, dims))
ROCCellArray{T,B}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N,B} = (check_T(T); ROCCellArray{T,N,B,eltype(T)}(undef, dims))

CPUCellArray{T,B}(::UndefInitializer, dims::Int...) where {T<:Cell,B} = CPUCellArray{T,B}(undef, dims)
 CuCellArray{T,B}(::UndefInitializer, dims::Int...) where {T<:Cell,B} = CuCellArray{T,B}(undef, dims)
ROCCellArray{T,B}(::UndefInitializer, dims::Int...) where {T<:Cell,B} = ROCCellArray{T,B}(undef, dims)

CPUCellArray{T}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N} = CPUCellArray{T,0}(undef, dims)
 CuCellArray{T}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N} = CuCellArray{T,0}(undef, dims)
ROCCellArray{T}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N} = ROCCellArray{T,0}(undef, dims)

CPUCellArray{T}(::UndefInitializer, dims::Int...) where {T<:Cell} = CPUCellArray{T}(undef, dims)
 CuCellArray{T}(::UndefInitializer, dims::Int...) where {T<:Cell} = CuCellArray{T}(undef, dims)
ROCCellArray{T}(::UndefInitializer, dims::Int...) where {T<:Cell} = ROCCellArray{T}(undef, dims)


## CellArray functions

@inline Base.IndexStyle(::Type{<:CellArray})                                 = IndexLinear()
@inline Base.size(T::Type{<:Number}, args...)                                = (1,)
@inline Base.size(A::CellArray)                                              = A.dims
@inline Base.length(T::Type{<:Number}, args...)                              = 1


@inline function Base.similar(A::CPUCellArray{T0,N0,B,T_elem0}, ::Type{T}, dims::NTuple{N,Int}) where {T0,N0,B,T_elem0,T<:Cell,N}
    CPUCellArray{T,N,B,eltype(T)}(undef, dims)
end

@inline function Base.similar(A::CuCellArray{T0,N0,B,T_elem0}, ::Type{T}, dims::NTuple{N,Int}) where {T0,N0,B,T_elem0,T<:Cell,N}
    CuCellArray{T,N,B,eltype(T)}(undef, dims)
end

@inline function Base.similar(A::ROCCellArray{T0,N0,B,T_elem0}, ::Type{T}, dims::NTuple{N,Int}) where {T0,N0,B,T_elem0,T<:Cell,N}
    ROCCellArray{T,N,B,eltype(T)}(undef, dims)
end

@inline function Base.similar(A::CellArray{T0,N0,B,T_array0}, ::Type{T}, dims::NTuple{N,Int}) where {T0,N0,B,T_array0,T<:Cell,N}
    check_T(T)
    T_arraykind = Base.typename(T_array0).wrapper  # Note: an alternative would be: T_array = typeof(similar(A.data, eltype(T), dims.*0)); CellArray{T,N,B}(T_array, dims)
    CellArray{T,N,B}(T_arraykind{eltype(T),_N}, undef, dims)
end


@inline function Base.fill!(A::CellArray{T,N,B,T_array}, x) where {T<:Number,N,B,T_array}
    cell = convert(T, x)
    A.data[:, 1, :] .= cell
    return A
end

@inline function Base.fill!(A::CellArray{T,N,B,T_array}, X) where {T<:Union{SArray,FieldArray},N,B,T_array}
    cell = convert(T, X)
    for j=1:length(T)
        A.data[:, j, :] .= cell[j]
    end
    return A
end


@inline function Base.getindex(A::CellArray{T,N,B,T_array}, i::Int) where {T<:Number,N,B,T_array<:AbstractArray{T,_N}}
    T(A.data[Base._to_linear_index(A.data::T_array, (i-1)%B+1, 1, (i-1)÷B+1)])
end

@inline function Base.setindex!(A::CellArray{T,N,B,T_array}, x::Number, i::Int) where {T<:Number,N,B,T_array}
    A.data[Base._to_linear_index(A.data::T_array, (i-1)%B+1, 1, (i-1)÷B+1)] = x
    return
end

@inline function Base.getindex(A::CellArray{T,N,B,T_array}, i::Int) where {T<:Union{SArray,FieldArray},N,B,T_array}
    T(getindex(A.data, Base._to_linear_index(A.data::T_array, (i-1)%B+1, j, (i-1)÷B+1)) for j=1:length(T)) # NOTE:The same fails on GPU if convert is used.
end

@inline function Base.setindex!(A::CellArray{T,N,B,T_array}, X::T, i::Int) where {T<:Union{SArray,FieldArray},N,B,T_array}
    for j=1:length(T)
        A.data[Base._to_linear_index(A.data::T_array, (i-1)%B+1, j, (i-1)÷B+1)] = X[j]
    end
    return
end


@inline Base.getindex(A::CellArray{T,N,0,T_array}, i::Int) where {T<:Number,N,T_array<:AbstractArray{T,_N}} = T(A.data[i])
@inline Base.setindex!(A::CellArray{T,N,0,T_array}, x::Number, i::Int) where {T<:Number,N,T_array}          = (A.data[i] = x; return)

@inline function Base.getindex(A::CellArray{T,N,0,T_array}, i::Int) where {T<:Union{SArray,FieldArray},N,T_array}
    T(getindex(A.data, Base._to_linear_index(A.data::T_array, i, j, 1)) for j=1:length(T)) # NOTE:The same fails on GPU if convert is used.
end

@inline function Base.setindex!(A::CellArray{T,N,0,T_array}, X::T, i::Int) where {T<:Union{SArray,FieldArray},N,T_array}
    for j=1:length(T)
        A.data[Base._to_linear_index(A.data::T_array, i, j, 1)] = X[j]
    end
    return
end


@inline Base.getindex(A::CellArray{T,N,1,T_array}, i::Int) where {T<:Number,N,T_array<:AbstractArray{T,_N}} = T(A.data[i])
@inline Base.setindex!(A::CellArray{T,N,1,T_array}, x::Number, i::Int) where {T<:Number,N,T_array}          = (A.data[i] = x; return)

@inline function Base.getindex(A::CellArray{T,N,1,T_array}, i::Int) where {T<:Union{SArray,FieldArray},N,T_array}
    T(getindex(A.data, Base._to_linear_index(A.data::T_array, 1, j, i)) for j=1:length(T)) # NOTE:The same fails on GPU if convert is used.
end

@inline function Base.setindex!(A::CellArray{T,N,1,T_array}, X::T, i::Int) where {T<:Union{SArray,FieldArray},N,T_array}
    for j=1:length(T)
        A.data[Base._to_linear_index(A.data::T_array, 1, j, i)] = X[j]
    end
    return
end


## API functions

"""
    cellsize(A)
    cellsize(A, dim)

Return a tuple containing the dimensions of `A` or return only a specific dimension, specified by `dim`.

"""
@inline cellsize(A::AbstractArray)           = size(eltype(A))
@inline cellsize(A::AbstractArray, dim::Int) = cellsize(A)[dim]


"""
    blocklength(A)

Return the blocklength of CellArray `A`.

"""
@inline blocklength(A::CellArray{T,N,B,T_array}) where {T,N,B,T_array} = (B == 0) ? prod(A.dims) : B


## Helper functions

function check_T(::Type{T}) where {T}
    if !isbitstype(T) @ArgumentError("the celltype, `T`, must be a bitstype.") end # Note: This test is required as FieldArray can be mutable and thus not bitstype (and ismutable() is for values not types...). The following tests would currently not be required as the current definition of the Cell type implies the tests to succeed.
    if !hasmethod(size, Tuple{Type{T}}) @ArgumentError("for the celltype, `T`, the following method must be defined: `@inline Base.size(T::Type{<:T}, args...)`") end
    if !hasmethod(eltype, Tuple{Type{T}}) @ArgumentError("for the celltype, `T`, the following method must be defined: `@inline Base.eltype(T::Type{<:T})`") end
    if !hasmethod(getindex, Tuple{Type{T}}) @ArgumentError("for the celltype, `T`, the following method must be defined: `@inline Base.getindex(X::T, i::Int)`") end
end

check_T(::Type{T}) where {T<:Number} = return
