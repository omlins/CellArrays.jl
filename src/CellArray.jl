using StaticArrays, Adapt


## Constants

const _N = 3
const Cell = Union{Number, SArray, FieldArray}


## Types and constructors

"""
    CellArray{T<:Cell,N,B,T_array} <: AbstractArray{T,N} where Cell <: Union{Number, SArray, FieldArray}

`N`-dimensional array with elements of type `T`, where `T` are `Cells` of type Number, SArray or FieldArray. `B` defines the blocklength, which refers to the amount of values of a same `Cell` field that are stored contigously (`B=1` means array of struct like storage; `B=prod(dims)` means array struct of array like storage; `B=0` is an alias for `B=prod(dims)`, enabling better peformance thanks to more specialized dispatch). `T_array` defines the array type used for storage.

--------------------------------------------------------------------------------

    CellArray{T,N,B}(T_arraykind, undef, dims)
    CellArray{T,B}(T_arraykind, undef, dims)
    CellArray{T}(T_arraykind, undef, dims)

Construct an uninitialized `N`-dimensional `CellArray` containing `Cells` of type `T` which are stored in an array of kind `T_arraykind`.

!!! note "Performance note"
    Best performance on GPUs is in general obtained with `B=0` as set by default. `B=1` migth give better performance in certain cases. Other values of `B` do with the current implementation not lead to optimal performance on GPU.

See also: [`CPUCellArray`](@ref), [`CuCellArray`](@ref), [`ROCCellArray`](@ref)
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

    function CellArray{T}(::Type{T_arraykind}, ::UndefInitializer, dims::Int...) where {T<:Cell,T_arraykind<:AbstractArray} #where {Type{T_arraykind}<:UnionAll}
        CellArray{T}(T_arraykind, undef, dims)
    end
end

Adapt.adapt_structure(to, A::CellArray{T,N,B,T_array}) where {T,N,B,T_array} = CellArray{T,N,B}(adapt(to, A.data), A.dims)


"""
    CPUCellArray{T<:Cell,N,B,T_elem} <: AbstractArray{T,N} where Cell <: Union{Number, SArray, FieldArray}

`N`-dimensional CellArray with cells of type `T`, blocklength `B`, and `T_array` being an `Array` of element type `T_elem`: alias for `CellArray{T,N,B,Array{T_elem,CellArrays._N}}`.

--------------------------------------------------------------------------------

    CPUCellArray{T,B}(undef, dims)
    CPUCellArray{T}(undef, dims)

Construct an uninitialized `N`-dimensional `CellArray` containing `Cells` of type `T` which are stored in an array of kind `Array`.

See also: [`CellArray`](@ref), [`CuCellArray`](@ref), [`ROCCellArray`](@ref)
"""
const CPUCellArray{T,N,B,T_elem} = CellArray{T,N,B,Array{T_elem,_N}}

CPUCellArray{T,B}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N,B} = (check_T(T); CPUCellArray{T,N,B,eltype(T)}(undef, dims))
CPUCellArray{T,B}(::UndefInitializer, dims::Int...) where {T<:Cell,B} = CPUCellArray{T,B}(undef, dims)
CPUCellArray{T}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:Cell,N} = CPUCellArray{T,0}(undef, dims)
CPUCellArray{T}(::UndefInitializer, dims::Int...) where {T<:Cell} = CPUCellArray{T}(undef, dims)

CPUCellArray(A::CellArray{T,N,B,T_array}) where {T,N,B,T_array} = CellArray{T,N,B}(Array(A.data), A.dims)


"""
    @define_CuCellArray

Define the following type alias and constructors in the caller module:

********************************************************************************
    CuCellArray{T<:Cell,N,B,T_elem} <: AbstractArray{T,N} where Cell <: Union{Number, SArray, FieldArray}

`N`-dimensional CellArray with cells of type `T`, blocklength `B`, and `T_array` being a `CuArray` of element type `T_elem`: alias for `CellArray{T,N,B,CuArray{T_elem,CellArrays._N,CUDA.DeviceMemory}}`.

--------------------------------------------------------------------------------

    CuCellArray{T,B}(undef, dims)
    CuCellArray{T}(undef, dims)

Construct an uninitialized `N`-dimensional `CellArray` containing `Cells` of type `T` which are stored in an array of kind `CuArray`.

See also: [`CellArray`](@ref), [`CPUCellArray`](@ref), [`ROCCellArray`](@ref)
********************************************************************************

!!! note "Avoiding unneeded dependencies"
    The type aliases and constructors for GPU `CellArray`s are provided via macros to avoid unneeded dependencies on the GPU packages in CellArrays.

See also: [`@define_ROCCellArray`](@ref)
"""
macro define_CuCellArray() esc(define_CuCellArray()) end

function define_CuCellArray()
    quote
        const CuCellArray{T,N,B,T_elem} = CellArrays.CellArray{T,N,B,CUDA.CuArray{T_elem,CellArrays._N,CUDA.DeviceMemory}}

        CuCellArray{T,B}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:CellArrays.Cell,N,B} = ( CellArrays.check_T(T); A = CuCellArray{T,N,B,CellArrays.eltype(T)}(undef, dims); f(A)=(CellArrays.plain(A); return); if (B in (0,1)) @cuda f(A) end; A )
        CuCellArray{T,B}(::UndefInitializer, dims::Int...) where {T<:CellArrays.Cell,B} = CuCellArray{T,B}(undef, dims)
        CuCellArray{T}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:CellArrays.Cell,N} = CuCellArray{T,0}(undef, dims)
        CuCellArray{T}(::UndefInitializer, dims::Int...) where {T<:CellArrays.Cell} = CuCellArray{T}(undef, dims)

        CuCellArray(A::CellArrays.CellArray{T,N,B,T_array}) where {T,N,B,T_array} = (A = CellArrays.CellArray{T,N,B}(CUDA.CuArray(A.data), A.dims); f(A)=(CellArrays.plain(A); return); if (B in (0,1)) @cuda f(A) end; A)

        Base.show(io::IO, A::CuCellArray)                                          = Base.show(io, CellArrays.CPUCellArray(A))
        Base.show(io::IO, ::MIME"text/plain", A::CuCellArray{T,N,B}) where {T,N,B} = ( println(io, "$(length(A))-element CuCellArray{$T, $N, $B, $(CellArrays.eltype(T))}:");  Base.print_array(io, CellArrays.CPUCellArray(A)) )
    end
end

"""
    @define_ROCCellArray

Define the following type alias and constructors in the caller module:

********************************************************************************
    ROCCellArray{T<:Cell,N,B,T_elem} <: AbstractArray{T,N} where Cell <: Union{Number, SArray, FieldArray}

`N`-dimensional CellArray with cells of type `T`, blocklength `B`, and `T_array` being a `ROCArray` of element type `T_elem`: alias for `CellArray{T,N,B,ROCArray{T_elem,CellArrays._N}}`.

--------------------------------------------------------------------------------

    ROCCellArray{T,B}(undef, dims)
    ROCCellArray{T}(undef, dims)

Construct an uninitialized `N`-dimensional `CellArray` containing `Cells` of type `T` which are stored in an array of kind `ROCArray`.

See also: [`CellArray`](@ref), [`CPUCellArray`](@ref), [`CuCellArray`](@ref)
********************************************************************************

!!! note "Avoiding unneeded dependencies"
    The type aliases and constructors for GPU `CellArray`s are provided via macros to avoid unneeded dependencies on the GPU packages in CellArrays.

See also: [`@define_CuCellArray`](@ref)
"""
macro define_ROCCellArray() esc(define_ROCCellArray()) end

function define_ROCCellArray()
    quote
        const ROCCellArray{T,N,B,T_elem} = CellArrays.CellArray{T,N,B,AMDGPU.ROCArray{T_elem,CellArrays._N}}
        
        ROCCellArray{T,B}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:CellArrays.Cell,N,B} = ( CellArrays.check_T(T); A = ROCCellArray{T,N,B,CellArrays.eltype(T)}(undef, dims); A ) # TODO: Once reshape is implemented in AMDGPU, the workaround can be applied as well: f(A)=(CellArrays.plain(A); return); if (B in (0,1)) @roc f(A) end; A )
        ROCCellArray{T,B}(::UndefInitializer, dims::Int...) where {T<:CellArrays.Cell,B} = ROCCellArray{T,B}(undef, dims)
        ROCCellArray{T}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:CellArrays.Cell,N} = ROCCellArray{T,0}(undef, dims)
        ROCCellArray{T}(::UndefInitializer, dims::Int...) where {T<:CellArrays.Cell} = ROCCellArray{T}(undef, dims)

        ROCCellArray(A::CellArrays.CellArray{T,N,B,T_array}) where {T,N,B,T_array} = ( A = CellArrays.CellArray{T,N,B}(AMDGPU.ROCArray(A.data), A.dims); A ) # TODO: Once reshape is implemented in AMDGPU, the workaround can be applied as well: f(A)=(CellArrays.plain(A); return); if (B in (0,1)) @roc f(A) end; A )

        Base.show(io::IO, A::ROCCellArray)                                          = Base.show(io, CellArrays.CPUCellArray(A))
        Base.show(io::IO, ::MIME"text/plain", A::ROCCellArray{T,N,B}) where {T,N,B} = ( println(io, "$(length(A))-element ROCCellArray{$T, $N, $B, $(CellArrays.eltype(T))}:");  Base.print_array(io, CellArrays.CPUCellArray(A)) )
    end
end

"""
    @define_MtlCellArray

Define the following type alias and constructors in the caller module:

********************************************************************************
    MtlCellArray{T<:Cell,N,B,T_elem} <: AbstractArray{T,N} where Cell <: Union{Number, SArray, FieldArray}

`N`-dimensional CellArray with cells of type `T`, blocklength `B`, and `T_array` being a `MtlArray` of element type `T_elem`: alias for `CellArray{T,N,B,MtlArray{T_elem,CellArrays._N}}`.

--------------------------------------------------------------------------------

    MtlCellArray{T,B}(undef, dims)
    MtlCellArray{T}(undef, dims)

Construct an uninitialized `N`-dimensional `CellArray` containing `Cells` of type `T` which are stored in an array of kind `MtlArray`.

See also: [`CellArray`](@ref), [`CPUCellArray`](@ref), [`CuCellArray`](@ref), [`ROCCellArray`](@ref)
********************************************************************************

!!! note "Avoiding unneeded dependencies"
    The type aliases and constructors for GPU `CellArray`s are provided via macros to avoid unneeded dependencies on the GPU packages in CellArrays.

See also: [`@define_CuCellArray`](@ref), [`@define_ROCCellArray`](@ref)
"""
macro define_MtlCellArray() esc(define_MtlCellArray()) end

function define_MtlCellArray()
    quote
        const MtlCellArray{T,N,B,T_elem} = CellArrays.CellArray{T,N,B,Metal.MtlArray{T_elem,CellArrays._N}}

        MtlCellArray{T,B}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:CellArrays.Cell,N,B} = ( CellArrays.check_T(T); A = MtlCellArray{T,N,B,CellArrays.eltype(T)}(undef, dims); f(A)=(CellArrays.plain(A); return); if (B in (0,1)) @metal f(A) end; A )
        MtlCellArray{T,B}(::UndefInitializer, dims::Int...) where {T<:CellArrays.Cell,B} = MtlCellArray{T,B}(undef, dims)
        MtlCellArray{T}(::UndefInitializer, dims::NTuple{N,Int}) where {T<:CellArrays.Cell,N} = MtlCellArray{T,0}(undef, dims)
        MtlCellArray{T}(::UndefInitializer, dims::Int...) where {T<:CellArrays.Cell} = MtlCellArray{T}(undef, dims)

        MtlCellArray(A::CellArrays.CellArray{T,N,B,T_array}) where {T,N,B,T_array} = ( A = CellArrays.CellArray{T,N,B}(Metal.MtlArray(A.data), A.dims); f(A)=(CellArrays.plain(A); return); if (B in (0,1)) @metal f(A) end; A )

        Base.show(io::IO, A::MtlCellArray)                                          = Base.show(io, CellArrays.CPUCellArray(A))
        Base.show(io::IO, ::MIME"text/plain", A::MtlCellArray{T,N,B}) where {T,N,B} = ( println(io, "$(length(A))-element MtlCellArray{$T, $N, $B, $(CellArrays.eltype(T))}:");  Base.print_array(io, CellArrays.CPUCellArray(A)) )
    end
end


## AbstractArray methods

@inline Base.IndexStyle(::Type{<:CellArray})    = IndexLinear()
@inline Base.size(T::Type{<:Number}, args...)   = (1,)
@inline Base.size(A::CellArray)                 = A.dims
@inline Base.length(T::Type{<:Number}, args...) = 1


@inline function Base.similar(A::CellArray{T0,N0,B,T_array0}, ::Type{T}, dims::NTuple{N,Int}) where {T0,N0,B,T_array0,T<:Cell,N}
    check_T(T)
    T_array = typeof(similar(A.data, eltype(T), ntuple(i -> 0, _N))) # Note: an alternative would have been in the past (this misses however the CUDA.DeviceMemory argument if T_arraykind is CuArray): T_arraykind = Base.typename(T_array0).wrapper; CellArray{T,N,B}(T_arraykind{eltype(T),_N}, undef, dims)
    CellArray{T,N,B}(T_array, undef, dims)
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

@inline function Base.getindex(A::CPUCellArray{T,N,1,T_elem}, i::Int) where {T<:Union{SArray,FieldArray},N,T_elem}
    getindex(reinterpret(reshape, T, view(A.data::Array{T_elem,_N},1,:,:)), i)  # NOTE: reinterpret is not implemented for CUDA device arrays, i.e. for usage in kernels
end

@inline function Base.setindex!(A::CPUCellArray{T,N,1,T_elem}, X::T, i::Int) where {T<:Union{SArray,FieldArray},N,T_elem}
    setindex!(reinterpret(reshape, T, view(A.data::Array{T_elem,_N},1,:,:)), X ,i)   # NOTE: reinterpret is not implemented for CUDA device arrays, i.e. for usage in kernels
    return
end


## CellArray properties

@inline Base.getproperty(A::CellArray{T,N,B,T_array}, fieldname::Symbol) where {T<:FieldArray,N,B,T_array} = getproperty(A, Val(fieldname))

@inline Base.getproperty(A::CellArray{T,N,B,T_array}, ::Val{:data}) where {T<:FieldArray{N2,T2,D},N,B,T_array} where {N2,T2,D} = getfield(A, :data)
@inline Base.getproperty(A::CellArray{T,N,B,T_array}, ::Val{:dims}) where {T<:FieldArray{N2,T2,D},N,B,T_array} where {N2,T2,D} = getfield(A, :dims)

@inline @generated function Base.getproperty(A::CellArray{T,N,B,T_array}, ::Val{fieldname}) where {T<:FieldArray{N2,T2,D},N,B,T_array,fieldname} where {N2,T2,D}
    names   = SArray{N2}(fieldnames(T))
    indices = Tuple(findfirst(x->x===fieldname, names))
    return :(field(A, $(indices)))
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


"""
    field(A, indices)
    field(A, fieldname)

Return an array view of the field of CellArray `A` designated with `indices` or `fieldname` (modifying the view will modify `A`). The view's dimensionality and size are equal to `A`'s. The operation is not supported if parameter `B` of `A` is neither `0` nor `1`.

## Arguments
- `indices::Int|NTuple{N,Int}`: the `indices` that designate the field in accordance with `A`'s cell type.
- `fieldname::Symbol`: the `fieldname` that designates the field in accordance with `A`'s cell type.
"""
@inline field(A::CellArray{T,N,0,T_array}, index::Int)                        where {T,N,T_array}                                     = view(plain(A), Base.OneTo.(size(A))..., index)
@inline field(A::CellArray{T,N,0,T_array}, indices::NTuple{M,Int})            where {T_elem,M,T<:AbstractArray{T_elem,M},N,  T_array} = view(plain(A), Base.OneTo.(size(A))..., indices...)
@inline field(A::CellArray{T,N,1,T_array}, index::Int)                        where {T,N,T_array}                                     = view(plain(A), index,      Base.OneTo.(size(A))...)
@inline field(A::CellArray{T,N,1,T_array}, indices::NTuple{M,Int})            where {T_elem,M,T<:AbstractArray{T_elem,M},N,  T_array} = view(plain(A), indices..., Base.OneTo.(size(A))...)
@inline field(A::CellArray{T,N,B,T_array}, indices::Union{Int,NTuple{M,Int}}) where {T_elem,M,T<:AbstractArray{T_elem,M},N,B,T_array} = @ArgumentError("the operation is not supported if parameter `B` of `A` is neither `0` nor `1`.")
@inline field(A::CellArray, indices::Int...)                                                                                          = field(A, indices)
@inline field(A::CellArray{T,N,B,T_array}, fieldname::Symbol)                 where {T<:FieldArray,N,B,T_array}                       = getproperty(A, fieldname)


## Helper functions

# """
#     plain(A)
#
# Return a plain `N`-dimensional array view of CellArray `A` (modifying the view will modify `A`), where `N` is the sum of the dimensionalities of `A` and the cell type of `A`. The view's dimensions are `(size(A)..., cellsize(A)...)` if parameter `B` of `A` is `0`, and `(cellsize(A)..., size(A)...)` if parameter `B` of `A` is `1`. The operation is not supported if parameter `B` of `A` is neither `0` nor `1`.
#
# """
@inline plain(A::CellArray{T,N,0,T_array}) where {T,N,  T_array} = reshape(A.data, (size(A)..., cellsize(A)...))
@inline plain(A::CellArray{T,N,1,T_array}) where {T,N,  T_array} = reshape(A.data, (cellsize(A)..., size(A)...))
@inline plain(A::CellArray{T,N,B,T_array}) where {T,N,B,T_array} = @ArgumentError("The operation is not supported if parameter `B` of `A` is neither `0` nor `1`.")


function check_T(::Type{T}) where {T}
    if !isbitstype(T) @ArgumentError("the celltype, `T`, must be a bitstype.") end # Note: This test is required as FieldArray can be mutable and thus not bitstype (and ismutable() is for values not types...). The following tests would currently not be required as the current definition of the Cell type implies the tests to succeed.
    if !hasmethod(size, Tuple{Type{T}}) @ArgumentError("for the celltype, `T`, the following method must be defined: `@inline Base.size(T::Type{<:T}, args...)`") end
    if !hasmethod(eltype, Tuple{Type{T}}) @ArgumentError("for the celltype, `T`, the following method must be defined: `@inline Base.eltype(T::Type{<:T})`") end
    if !hasmethod(getindex, Tuple{Type{T}}) @ArgumentError("for the celltype, `T`, the following method must be defined: `@inline Base.getindex(X::T, i::Int)`") end
end

check_T(::Type{T}) where {T<:Number} = return
