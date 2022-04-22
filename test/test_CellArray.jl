using Test
using CellArrays
using CUDA, AMDGPU, StaticArrays
import CellArrays: IncoherentArgumentError, ArgumentError

test_cuda = CUDA.functional()
test_amdgpu = false

array_types           = ["CPU"]
ArrayConstructors     = [Array]
CellArrayConstructors = [CellArray]
if test_cuda
	cuzeros = CUDA.zeros
	push!(array_types, "CUDA")
	push!(ArrayConstructors, CuArray)
	push!(CellArrayConstructors, CuCellArray)
end
if test_amdgpu
	roczeros = AMDGPU.zeros
	push!(array_types, "AMDGPU")
	push!(ArrayConstructors, ROCArray)
	push!(CellArrayConstructors, ROCCellArray)
end

@testset "$(basename(@__FILE__))" begin
    @testset "1. CellArray allocation ($array_type arrays)" for (array_type, Array, CellArray) in zip(array_types, ArrayConstructors, CellArrayConstructors)
        @testset "Number cells" begin
			dims = (2,3)
			A = CellArray(Float64, dims)
			B = CellArray(Int32, dims...)
			@test typeof(A.data) <: Array
			@test typeof(B.data) <: Array
			@test eltype(A.data) == Float64
			@test eltype(B.data) == Int32
			@test eltype(A)      == Float64
			@test eltype(B)      == Int32
			@test typeof(A)      == CellArrays.CellArray{Float64, length(dims), Array}
			@test typeof(B)      == CellArrays.CellArray{Int32, length(dims), Array}
			@test length(A.data) == prod(dims)
			@test length(B.data) == prod(dims)
			@test A.dims         == dims
			@test B.dims         == dims
        end;
		@testset "SArray cells" begin
			dims      = (2,3)
			celldims  = (3,4)
			T_Float64 = SMatrix{celldims..., Float64, prod(celldims)}
			T_Int32   = SMatrix{celldims...,   Int32, prod(celldims)}
			A = CellArray(T_Float64, dims)
			B = CellArray(T_Int32, dims...)
			@test typeof(A.data) <: Array
			@test typeof(B.data) <: Array
			@test eltype(A.data) == Float64
			@test eltype(B.data) == Int32
			@test eltype(A)      == T_Float64
			@test eltype(B)      == T_Int32
			@test typeof(A)      == CellArrays.CellArray{T_Float64, length(dims), Array}
			@test typeof(B)      == CellArrays.CellArray{T_Int32, length(dims), Array}
			@test length(A.data) == prod(dims)*prod(celldims)
			@test length(B.data) == prod(dims)*prod(celldims)
			@test A.dims         == dims
			@test B.dims         == dims
        end;
    end;
	@testset "2. functions ($array_type arrays)" for (array_type, Array, CellArray) in zip(array_types, ArrayConstructors, CellArrayConstructors)
		dims      = (2,3)
		celldims  = (3,4)
		T_Float64 = SMatrix{celldims..., Float64, prod(celldims)}
		T_Int32   = SMatrix{celldims...,   Int32, prod(celldims)}
		A = CellArray(Float64, dims)
		B = CellArray(Int32, dims)
		C = CellArray(T_Float64, dims)
		D = CellArray(T_Int32, dims)
        @testset "size" begin
			@test size(A) == dims
			@test size(B) == dims
			@test size(C) == dims
			@test size(D) == dims
			@test cellsize(A) == 1
			@test cellsize(B) == 1
			@test cellsize(C) == celldims
			@test cellsize(D) == celldims
        end;
    end;
	@testset "3. Exceptions ($array_type arrays)" for (array_type, Array, CellArray) in zip(array_types, ArrayConstructors, CellArrayConstructors)
		dims      = (2,3)
		celldims  = (3,4)
		T_Float64 = SMatrix{celldims..., Float64, prod(celldims)}
		T_Int32   = SMatrix{celldims...,   Int32, prod(celldims)}
		A = CellArray(Float64, dims)
		B = CellArray(Int32, dims)
		C = CellArray(T_Float64, dims)
		D = CellArray(T_Int32, dims)
		@test_throws IncoherentArgumentError CellArrays.CellArray{Float64,2}(similar(A.data, Float32), A.dims)                # Error: eltype(data) must match eltype(T).
		@test_throws IncoherentArgumentError CellArrays.CellArray{Int32,2}(similar(B.data, Float64), B.dims)                  # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T_Float64,2}(similar(C.data, Float32), C.dims)              # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T_Int32,2}(similar(D.data, T_Int32), D.dims)                # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{Float64,2}(similar(A.data, Float64, (1,2)), A.dims)         # Error: size(data) must match (prod(dims), prod(size(T))).
		@test_throws IncoherentArgumentError CellArrays.CellArray{T_Float64,2}(similar(C.data, Float64, (1,2)), C.dims)       # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{SMatrix{(4,5)..., Float64, prod((4,5))},2}(C.data, C.dims)  # ...
		@test_throws ArgumentError CellArrays.CellArray{Float64,2}(A.data[:], A.dims)                                         # Error: ndims(data) must be 2.
		@test_throws ArgumentError CellArrays.CellArray{T_Float64,2}(C.data[:], C.dims)                                       # Error: ...
	end;
end;
