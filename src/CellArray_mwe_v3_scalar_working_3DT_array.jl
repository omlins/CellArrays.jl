using StaticArrays, Adapt, CUDA

# struct CellArray{T<:Cell,N,T_array<:AbstractArray} <: AbstractArray{T,N}
#     data::T_array
#     dims::NTuple{N,Int}
#
#     function CellArray{T,N}(data::T_array, dims::NTuple{N,Int}) where {T,N} where {T_array<:AbstractArray}
#         celldims = size(T)  # Note: size must be defined for type T (as it is e.g. for StaticArrays)
#         T_elem = eltype(T)  # Note: eltype must be defined for type T (as it is e.g. for StaticArrays)
#         if (eltype(data) != T_elem)                     error("eltype(data) must match eltype(T).") end
#         if (ndims(data) != 2)                           error("ndims(data) must be 2.") end
#         if (size(data) != (prod(dims), prod(celldims))) error("size(data) must match (prod(dims), prod(size(T))).") end
#         new{T,N,T_array}(data, dims)
#     end
#     function CellArray{T,N}(dims::NTuple{N,Int}, T_array::Type{<:AbstractArray}) where {T,N}
#         celldims = size(T)  # Note: size must be defined for type T (as it is e.g. for StaticArrays)
#         T_elem = eltype(T)  # Note: eltype must be defined for type T (as it is e.g. for StaticArrays)
#         data = T_array{T_elem,2}(undef, prod(dims), prod(celldims))
#         new{T,N,T_array}(data, dims)
#     end
#     function CellArray{T}(dims::NTuple{N,Int}, T_array::Type{<:AbstractArray}) where {T,N}
#         CellArray{T,N}(dims, T_array)
#     end
# end

Cell = Union{Number, SArray, FieldArray}

# struct CellArray{T<:Cell,N,B,T_array<:AbstractArray} <: AbstractArray{T,N}
#     data::T_array
#     dims::NTuple{N,Int}
#
#     function CellArray{T,N,B,T_array}(data::T_array, dims::NTuple{N,Int}) where {T<:Cell, N, B, T_array<:AbstractArray}
#         check_T(T)
#         celldims = size(T)  # Note: size must be defined for type T (as it is e.g. for StaticArrays)
#         T_elem = eltype(T)  # Note: eltype must be defined for type T (as it is e.g. for StaticArrays)
#         if (eltype(data) != T_elem)                     error("eltype(data) must match eltype(T).") end
#         #if (ndims(data) != 3)                           error("ndims(data) must be 3.") end
#         # if (size(data) != (B, prod(celldims), ceil(Int,prod(dims)/B))) error("size(data) must match (B, prod(size(T), ceil(prod(dims)/B)).") end
#         if (ndims(data) != 3)                           error("ndims(data) must be 3.") end
#         if (size(data) != (prod(dims), prod(celldims), 1)) error("size(data) must match (prod(dims), prod(size(T))).") end
#         new{T,N,B,T_array}(data, dims)
#     end
#     function CellArray{T,N,B}(data::T_array, dims::NTuple{N,Int}) where {T<:Cell, N, B, T_array<:AbstractArray}
#         CellArray{T,N,B,T_array}(data, dims)
#     end
#     function CellArray{T,N,B,T_array}(dims::NTuple{N,Int}) where {T<:Cell, N, B, T_array<:AbstractArray}
#         check_T(T)
#         celldims = size(T)  # Note: size must be defined for type T (as it is e.g. for StaticArrays)
#         T_elem = eltype(T)  # Note: eltype must be defined for type T (as it is e.g. for StaticArrays)
#         #data = T_array{T_elem,3}(undef, B, prod(celldims), ceil(Int,prod(dims)/B))
#         data = T_array{T_elem,3}(undef, prod(dims), prod(celldims), 1)
#         CellArray{T,N,B,T_array}(data, dims)
#     end
#     function CellArray{T,B,T_array}(dims::NTuple{N,Int}) where {T<:Cell, N, B, T_array<:AbstractArray}
#         CellArray{T,N,B,T_array}(dims)
#     end
# end

struct CellArray{T<:Cell,N,B,T_array<:AbstractArray{T_elem,3} where {T_elem}} <: AbstractArray{T,N}
    data::T_array
    dims::NTuple{N,Int}

    function CellArray{T,N,B,T_array}(data::T_array, dims::NTuple{N,Int}) where {T<:Cell, N, B, T_array<:AbstractArray{T_elem,3}} where {T_elem}
        check_T(T)
        celldims = size(T)  # Note: size must be defined for type T (as it is e.g. for StaticArrays)
        #T_elem = eltype(T)  # Note: eltype must be defined for type T (as it is e.g. for StaticArrays)
        if (eltype(data) != eltype(T))                     error("eltype(data) must match eltype(T).") end
        #if (ndims(data) != 3)                           error("ndims(data) must be 3.") end
        # if (size(data) != (B, prod(celldims), ceil(Int,prod(dims)/B))) error("size(data) must match (B, prod(size(T), ceil(prod(dims)/B)).") end
        if (ndims(data) != 3)                           error("ndims(data) must be 3.") end
        if (size(data) != (prod(dims), prod(celldims), 1)) error("size(data) must match (prod(dims), prod(size(T))).") end
        new{T,N,B,T_array}(data, dims)
    end
    function CellArray{T,N,B}(data::T_array, dims::NTuple{N,Int}) where {T<:Cell, N, B, T_array<:AbstractArray{T_elem,3}} where {T_elem}
        CellArray{T,N,B,T_array}(data, dims)
    end
    function CellArray{T,N,B}(::Type{T_arraykind}, dims::NTuple{N,Int}) where {T<:Cell, N, B, T_arraykind<:AbstractArray}
        check_T(T)
        celldims = size(T)  # Note: size must be defined for type T (as it is e.g. for StaticArrays)
        T_elem = eltype(T)  # Note: eltype must be defined for type T (as it is e.g. for StaticArrays)
        #data = T_array{T_elem,3}(undef, B, prod(celldims), ceil(Int,prod(dims)/B))
        T_array = T_arraykind{T_elem,3}
        data = T_array(undef, prod(dims), prod(celldims), 1)
        CellArray{T,N,B,T_array}(data, dims)
    end
    function CellArray{T,B}(::Type{T_arraykind}, dims::NTuple{N,Int}) where {T<:Cell, N, B, T_arraykind<:AbstractArray}
        CellArray{T,N,B,T_arraykind}(dims)
    end
    # function CellArray{T,N,B,T_arraykind}(dims::NTuple{N,Int}) where {T<:Cell, N, B, T_arraykind<:AbstractArray}
    #     check_T(T)
    #     CellArray{T,N,B,T_arraykind{eltype(T),3}}(dims)
    # end
    # function CellArray{T,B,T_arraykind}(dims::NTuple{N,Int}) where {T<:Cell, N, B, T_arraykind<:AbstractArray}
    #     CellArray{T,N,B,T_arraykind}(dims)
    # end
end

CellArray(::Type{T}, dims::NTuple{N,Int}) where {T,N}                   = CellArray{T,N,prod(dims)}(Array, dims)
CellArray(::Type{T}, dims::Int...) where {T}                            = CellArray{T,N}(dims)
CuCellArray(::Type{T}, dims::NTuple{N,Int}) where {T,N}                 = CellArray{T,N,prod(dims)}(CuArray, dims)
CuCellArray(::Type{T}, dims::Int...) where {T}                          = CuCellArray{T,N}(dims)

@inline Base.IndexStyle(::Type{<:CellArray})                            = IndexLinear()
@inline Base.size(T::Type{<:Number}, args...   )                        = 1
@inline Base.size(A::CellArray)                                         = A.dims
@inline Base.getindex(A::CellArray{T}, i::Int) where {T <: Number}      = T(A.data[i])
@inline Base.getindex(A::CellArray{T}, i::Int) where {T}                = T(getindex(A.data, i, j, 1) for j=1:length(T)) # NOTE:The same fails on GPU if convert is used.
@inline Base.setindex!(A::CellArray{T}, x::T, i::Int) where {T<:Number} = (A.data[i] = x; return)
@inline Base.setindex!(A::CellArray{T}, X::T, i::Int) where {T}         = (for j=1:length(T) A.data[i,j,1] = X[j] end; return)

@inline cellsize(A::AbstractArray)                                      = size(eltype(A))
@inline cellsize(A::AbstractArray, i::Int)                              = cellsize(A)[i]

# Adapt.adapt_structure(to, A::CellArray{T,N}) where {T,N}                = CellArray{T,N}(adapt(to, A.data), A.dims)
Adapt.adapt_structure(to, A::CellArray{T,N,B}) where {T,N,B}            = CellArray{T,N,B}(adapt(to, A.data), A.dims)

function check_T(::Type{T}) where {T}
    if !isbitstype(T) error("the celltype, `T`, must be a bitstype.") end # Note: This test is required as FieldArray can be mutable and thus not bitstype (and ismutable() is for values not types...). The following tests would currently not be required as the current definition of the Cell type implies the tests to succeed.
    if !hasmethod(size, Tuple{Type{T}}) error("for the celltype, `T`, the following method must be defined: `@inline Base.size(T::Type{<:T}, args...)`") end
    if !hasmethod(eltype, Tuple{Type{T}}) error("for the celltype, `T`, the following method must be defined: `@inline Base.eltype(T::Type{<:T})`") end
    if !hasmethod(getindex, Tuple{Type{T}}) error("for the celltype, `T`, the following method must be defined: `@inline Base.getindex(X::T, i::Int)`") end
end

check_T(::Type{T}) where {T <: Number} = return


## Test
using Test

dims      = (2,3)
celldims  = (4,4)
T_Float64 = SMatrix{celldims..., Float64, prod(celldims)}
A = CuCellArray(Float64,dims)
C = CuCellArray(T_Float64,dims)
A.data.=0; C.data.=0;

A[2,2:3] .= 9
C[2,2:3] .= (T_Float64(1:length(T_Float64)), T_Float64(1:length(T_Float64)))
@test all(A[2,2:3] .== 9.0)
@test all(C[2,2:3] .== (T_Float64(1:length(T_Float64)), T_Float64(1:length(T_Float64))))

function add2D!(A, B)
    ix = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x
    iy = (CUDA.blockIdx().y-1) * CUDA.blockDim().y + CUDA.threadIdx().y
    A[ix,iy] = A[ix,iy] + B[ix,iy];
    #A[ix] = A[ix] + B[ix];
    return
end
function matsquare2D!(A)
    ix = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x
    iy = (CUDA.blockIdx().y-1) * CUDA.blockDim().y + CUDA.threadIdx().y
    A[ix,iy] = A[ix,iy] * A[ix,iy];
    #A[ix,iy] = A[ix,iy] + A[ix,iy];
    #A.data[ix+(iy-1)*2] = A.data[ix+(iy-1)*2] + A.data[ix+(iy-1)*2];
    return
end
A.data.=3; @cuda blocks=size(A) matsquare2D!(A); synchronize(); @test all(A.data .== 9)
A.data.=3; @cuda blocks=size(A) add2D!(A, A); synchronize(); @test all(A.data .== 6)
C.data.=3; @cuda blocks=size(C) matsquare2D!(C); synchronize(); @test all(C.data .== 36)
C.data.=3; @cuda blocks=size(C) add2D!(C, C); synchronize(); @test all(C.data .== 6)
