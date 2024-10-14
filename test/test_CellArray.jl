using Test
using CUDA, AMDGPU, Metal, StaticArrays
import CellArrays
import CellArrays: CPUCellArray, @define_CuCellArray, @define_ROCCellArray, @define_MtlCellArray, cellsize, blocklength, _N
import CellArrays: IncoherentArgumentError, ArgumentError

@define_CuCellArray
@define_ROCCellArray
@define_MtlCellArray

test_cuda = CUDA.functional()
test_amdgpu = AMDGPU.functional()
test_metal = Metal.functional()

array_types           = ["CPU"]
ArrayConstructors     = [Array]
CellArrayConstructors = [CPUCellArray]
precision_types       = [Float64]
allowscalar_functions = Function[(x->x)]
if test_cuda
	cuzeros = CUDA.zeros
	push!(array_types, "CUDA")
	push!(ArrayConstructors, CuArray)
	push!(CellArrayConstructors, CuCellArray)
	push!(allowscalar_functions, CUDA.allowscalar)
	push!(precision_types, Float64)
end
if test_amdgpu
	roczeros = AMDGPU.zeros
	push!(array_types, "AMDGPU")
	push!(ArrayConstructors, ROCArray)
	push!(CellArrayConstructors, ROCCellArray)
	push!(allowscalar_functions, AMDGPU.allowscalar)
  push!(precision_types, Float64)
end
if test_metal
	metalzeros = Metal.zeros
	push!(array_types, "Metal")
	push!(ArrayConstructors, MtlArray)
	push!(CellArrayConstructors, MtlCellArray)
	push!(allowscalar_functions, Metal.allowscalar)
	push!(precision_types, Float32)
end

struct MyFieldArray{T} <: FieldArray{Tuple{2,2,2,2}, T, 4}
    xxxx::T
    yxxx::T
    xyxx::T
    yyxx::T
    xxyx::T
    yxyx::T
    xyyx::T
    yyyx::T
    xxxy::T
    yxxy::T
    xyxy::T
    yyxy::T
    xxyy::T
    yxyy::T
    xyyy::T
    yyyy::T
end

mutable struct MyMutableFieldArray{T} <: FieldArray{Tuple{2}, T, 1}
    xxxx::T
    yxxx::T
end

@testset "$(basename(@__FILE__))" begin
    @testset "1. CellArray allocation ($array_type arrays) (precision: $(nameof(Float))" for (array_type, Array, CellArray, allowscalar, Float) in zip(array_types, ArrayConstructors, CellArrayConstructors, allowscalar_functions, precision_types) 
        @testset "Number cells" begin
			dims = (2,3)
			A = CellArray{Float}(undef, dims)
			B = CellArrays.CellArray{Int32,prod(dims)}(Array, undef, dims...)
			C = CellArray{Float,1}(undef, dims)
			D = CellArray{Int32,4}(undef, dims)
			@test typeof(A.data) <: Array
			@test typeof(B.data) <: Array
			@test typeof(C.data) <: Array
			@test typeof(D.data) <: Array
			@test eltype(A.data) == Float
			@test eltype(B.data) == Int32
			@test eltype(C.data) == Float
			@test eltype(D.data) == Int32
			@test eltype(A)      == Float
			@test eltype(B)      == Int32
			@test eltype(C)      == Float
			@test eltype(D)      == Int32
			@test typeof(A)      == CellArrays.CellArray{Float, length(dims), 0, Array{eltype(A.data),_N}}
			@test typeof(B)      == CellArrays.CellArray{Int32, length(dims), prod(dims), Array{eltype(B.data),_N}}
			@test typeof(C)      == CellArrays.CellArray{Float, length(dims), 1, Array{eltype(C.data),_N}}
			@test typeof(D)      == CellArrays.CellArray{Int32, length(dims), 4, Array{eltype(D.data),_N}}
			@test length(A.data) == prod(dims)
			@test length(B.data) == prod(dims)
			@test length(C.data) == prod(dims)
			@test (length(D.data) > prod(dims) && length(D.data)%prod(dims) <= 4)
			@test A.dims         == dims
			@test B.dims         == dims
			@test C.dims         == dims
			@test D.dims         == dims
        end;
		@testset "SArray cells" begin
			dims      = (2,3)
			celldims  = (3,4)
			T_Float = SMatrix{celldims..., Float, prod(celldims)}
			T_Int32   = SMatrix{celldims...,   Int32, prod(celldims)}
			A = CellArray{T_Float}(undef, dims)
			B = CellArrays.CellArray{T_Int32,prod(dims)}(Array, undef, dims...)
			C = CellArray{T_Float,1}(undef, dims)
			D = CellArray{T_Int32,4}(undef, dims)
			@test typeof(A.data) <: Array
			@test typeof(B.data) <: Array
			@test typeof(C.data) <: Array
			@test typeof(D.data) <: Array
			@test eltype(A.data) == Float
			@test eltype(B.data) == Int32
			@test eltype(C.data) == Float
			@test eltype(D.data) == Int32
			@test eltype(A)      == T_Float
			@test eltype(B)      == T_Int32
			@test eltype(C)      == T_Float
			@test eltype(D)      == T_Int32
			@test typeof(A)      == CellArrays.CellArray{T_Float, length(dims), 0, Array{eltype(A.data),_N}}
			@test typeof(B)      == CellArrays.CellArray{T_Int32, length(dims), prod(dims), Array{eltype(B.data),_N}}
			@test typeof(C)      == CellArrays.CellArray{T_Float, length(dims), 1, Array{eltype(C.data),_N}}
			@test typeof(D)      == CellArrays.CellArray{T_Int32, length(dims), 4, Array{eltype(D.data),_N}}
			@test length(A.data) == prod(dims)*prod(celldims)
			@test length(B.data) == prod(dims)*prod(celldims)
			@test length(C.data) == prod(dims)*prod(celldims)
			@test ((length(D.data) > prod(dims)*prod(celldims)) && (length(D.data)%prod(dims) <= 4))
			@test A.dims         == dims
			@test B.dims         == dims
			@test C.dims         == dims
			@test D.dims         == dims
        end;
		@testset "FieldArray cells" begin
			dims      = (2,3)
			celldims  = size(MyFieldArray)
			T_Float = MyFieldArray{Float}
			T_Int32   = MyFieldArray{Int32}
			A = CellArray{T_Float}(undef, dims)
			B = CellArrays.CellArray{T_Int32,prod(dims)}(Array, undef, dims...)
			C = CellArray{T_Float,1}(undef, dims)
			D = CellArray{T_Int32,4}(undef, dims)
			@test typeof(A.data) <: Array
			@test typeof(B.data) <: Array
			@test typeof(C.data) <: Array
			@test typeof(D.data) <: Array
			@test eltype(A.data) == Float
			@test eltype(B.data) == Int32
			@test eltype(C.data) == Float
			@test eltype(D.data) == Int32
			@test eltype(A)      == T_Float
			@test eltype(B)      == T_Int32
			@test eltype(C)      == T_Float
			@test eltype(D)      == T_Int32
			@test typeof(A)      == CellArrays.CellArray{T_Float, length(dims), 0, Array{eltype(A.data),_N}}
			@test typeof(B)      == CellArrays.CellArray{T_Int32, length(dims), prod(dims), Array{eltype(B.data),_N}}
			@test typeof(C)      == CellArrays.CellArray{T_Float, length(dims), 1, Array{eltype(C.data),_N}}
			@test typeof(D)      == CellArrays.CellArray{T_Int32, length(dims), 4, Array{eltype(D.data),_N}}
			@test length(A.data) == prod(dims)*prod(celldims)
			@test length(B.data) == prod(dims)*prod(celldims)
			@test length(C.data) == prod(dims)*prod(celldims)
			@test ((length(D.data) > prod(dims)*prod(celldims)) && (length(D.data)%prod(dims) <= 4))
			@test A.dims         == dims
			@test B.dims         == dims
			@test C.dims         == dims
			@test D.dims         == dims
        end;
    end;
	@testset "2. functions ($array_type arrays) (precision: $(nameof(Float))" for (array_type, Array, CellArray, allowscalar, Float) in zip(array_types, ArrayConstructors, CellArrayConstructors, allowscalar_functions, precision_types) 
		dims      = (2,3)
		celldims  = (3,4) # Needs to be compatible for matrix multiplication!
		T_Float = SMatrix{celldims..., Float, prod(celldims)}
		T_Int32   = SMatrix{celldims...,   Int32, prod(celldims)}
		T2_Float = MyFieldArray{Float}
		T2_Int32   = MyFieldArray{Int32}
		A = CellArray{Float}(undef, dims)
		B = CellArrays.CellArray{Int32,prod(dims)}(Array, undef, dims)
		C = CellArray{T_Float}(undef, dims)
		D = CellArray{T_Int32,prod(dims)}(undef, dims)
		E = CellArray{T2_Float}(undef, dims)
		F = CellArray{T2_Int32,prod(dims)}(undef, dims)
		G = CellArray{T_Float,1}(undef, dims)
		H = CellArray{T_Int32,4}(undef, dims)
        @testset "size" begin
			@test size(A) == dims
			@test size(B) == dims
			@test size(C) == dims
			@test size(D) == dims
			@test size(E) == dims
			@test size(F) == dims
			@test size(G) == dims
			@test size(H) == dims
        end;
		@testset "similar" begin
			@test typeof(similar(A, T_Int32)) == CellArrays.CellArray{T_Int32, length(dims),              0, Array{eltype(T_Int32),_N}}
			@test typeof(similar(B, T_Int32)) == CellArrays.CellArray{T_Int32, length(dims), blocklength(B), Array{eltype(T_Int32),_N}}
			@test typeof(similar(C, T_Int32)) == CellArrays.CellArray{T_Int32, length(dims),              0, Array{eltype(T_Int32),_N}}
			@test typeof(similar(D, T_Int32)) == CellArrays.CellArray{T_Int32, length(dims), blocklength(D), Array{eltype(T_Int32),_N}}
			@test typeof(similar(E, T_Int32)) == CellArrays.CellArray{T_Int32, length(dims),              0, Array{eltype(T_Int32),_N}}
			@test typeof(similar(F, T_Int32)) == CellArrays.CellArray{T_Int32, length(dims), blocklength(F), Array{eltype(T_Int32),_N}}
			@test typeof(similar(G, T_Int32)) == CellArrays.CellArray{T_Int32, length(dims), blocklength(G), Array{eltype(T_Int32),_N}}
			@test typeof(similar(H, T_Int32)) == CellArrays.CellArray{T_Int32, length(dims), blocklength(H), Array{eltype(T_Int32),_N}}
			@test typeof(similar(A, T_Int32, (1,2))) == CellArrays.CellArray{T_Int32, 2,              0, Array{eltype(T_Int32),_N}}
			@test typeof(similar(B, T_Int32, (1,2))) == CellArrays.CellArray{T_Int32, 2, blocklength(B), Array{eltype(T_Int32),_N}}
			@test typeof(similar(C, T_Int32, (1,2))) == CellArrays.CellArray{T_Int32, 2,              0, Array{eltype(T_Int32),_N}}
			@test typeof(similar(D, T_Int32, (1,2))) == CellArrays.CellArray{T_Int32, 2, blocklength(D), Array{eltype(T_Int32),_N}}
			@test typeof(similar(E, T_Int32, (1,2))) == CellArrays.CellArray{T_Int32, 2,              0, Array{eltype(T_Int32),_N}}
			@test typeof(similar(F, T_Int32, (1,2))) == CellArrays.CellArray{T_Int32, 2, blocklength(F), Array{eltype(T_Int32),_N}}
			@test typeof(similar(G, T_Int32, (1,2))) == CellArrays.CellArray{T_Int32, 2, blocklength(G), Array{eltype(T_Int32),_N}}
			@test typeof(similar(H, T_Int32, (1,2))) == CellArrays.CellArray{T_Int32, 2, blocklength(H), Array{eltype(T_Int32),_N}}
        end;
		@testset "fill!" begin
			allowscalar() do
				fill!(A, 9);   @test all(Base.Array(A.data) .== 9.0)
				fill!(B, 9.0); @test all(Base.Array(B.data) .== 9)
				fill!(C, (1:length(eltype(C)))); @test all(C .== (T_Float(1:length(eltype(C)))  for i=1:dims[1], j=1:dims[2]))
				fill!(D, (1:length(eltype(D)))); @test all(D .== (T_Int32(1:length(eltype(D)))    for i=1:dims[1], j=1:dims[2]))
				fill!(E, (1:length(eltype(E)))); @test all(E .== (T2_Float(1:length(eltype(E))) for i=1:dims[1], j=1:dims[2]))
				fill!(F, (1:length(eltype(F)))); @test all(F .== (T2_Int32(1:length(eltype(F)))   for i=1:dims[1], j=1:dims[2]))
				fill!(G, (1:length(eltype(G)))); @test all(G .== (T_Float(1:length(eltype(G)))  for i=1:dims[1], j=1:dims[2]))
				fill!(H, (1:length(eltype(H)))); @test all(H .== (T_Int32(1:length(eltype(H)))    for i=1:dims[1], j=1:dims[2]))
			end
		end
		@testset "getindex / setindex! (array programming)" begin
			allowscalar() do
				A.data.=0; B.data.=0; C.data.=0; D.data.=0; E.data.=0; F.data.=0; G.data.=0; H.data.=0;
				A[2,2:3] .= 9
				B[2,2:3] .= 9.0
				C[2,2:3] .= (T_Float(1:length(T_Float)), T_Float(1:length(T_Float)))
				D[2,2:3] .= (T_Int32(1:length(T_Int32)), T_Int32(1:length(T_Int32)))
				E[2,2:3] .= (T2_Float(1:length(T2_Float)), T2_Float(1:length(T2_Float)))
				F[2,2:3] .= (T2_Int32(1:length(T2_Int32)), T2_Int32(1:length(T2_Int32)))
				G[2,2:3] .= (T_Float(1:length(T_Float)), T_Float(1:length(T_Float)))
				H[2,2:3] .= (T_Int32(1:length(T_Int32)), T_Int32(1:length(T_Int32)))
				@test all(A[2,2:3] .== 9.0)
				@test all(B[2,2:3] .== 9)
				@test all(C[2,2:3] .== (T_Float(1:length(T_Float)), T_Float(1:length(T_Float))))
				@test all(D[2,2:3] .== (T_Int32(1:length(T_Int32)), T_Int32(1:length(T_Int32))))
				@test all(E[2,2:3] .== (T2_Float(1:length(T2_Float)), T2_Float(1:length(T2_Float))))
				@test all(F[2,2:3] .== (T2_Int32(1:length(T2_Int32)), T2_Int32(1:length(T2_Int32))))
				@test all(G[2,2:3] .== (T_Float(1:length(T_Float)), T_Float(1:length(T_Float))))
				@test all(H[2,2:3] .== (T_Int32(1:length(T_Int32)), T_Int32(1:length(T_Int32))))
			end
        end;
		@testset "getindex / setindex! (GPU kernel programming)" begin
			celldims2 = (4,4) # Needs to be compatible for matrix multiplication!
			T_Int32   = SMatrix{celldims2...,   Int32, prod(celldims2)}
			J         = CellArray{T_Int32,4}(undef, dims)
			J_ref     = CellArray{T_Int32,4}(undef, dims)
			if array_type == "CUDA"
				function add2D_CUDA!(A, B)
				    ix = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x
				    iy = (CUDA.blockIdx().y-1) * CUDA.blockDim().y + CUDA.threadIdx().y
				    A[ix,iy] = A[ix,iy] + B[ix,iy];
				    return
				end
				function matsquare2D_CUDA!(A)
				    ix = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x
				    iy = (CUDA.blockIdx().y-1) * CUDA.blockDim().y + CUDA.threadIdx().y
					A[ix,iy] = A[ix,iy] * A[ix,iy];
				    return
				end
				A.data.=3;                 @cuda blocks=size(A) matsquare2D_CUDA!(A); CUDA.synchronize(); @test all(Base.Array(A.data) .== 9)
				J.data.=3; J_ref.data.=36; @cuda blocks=size(J) matsquare2D_CUDA!(J); CUDA.synchronize(); @test CUDA.@allowscalar all(J .== J_ref)
				C.data.=2; G.data.=3;      @cuda blocks=size(C) add2D_CUDA!(C, G);    CUDA.synchronize(); @test all(Base.Array(C.data) .== 5)
			end
			if array_type == "AMDGPU"
				function add2D_AMDGPU!(A, B)
				    ix = (AMDGPU.blockIdx().x-1) * AMDGPU.blockDim().x + AMDGPU.threadIdx().x
				    iy = (AMDGPU.blockIdx().y-1) * AMDGPU.blockDim().y + AMDGPU.threadIdx().y
				    A[ix,iy] = A[ix,iy] + B[ix,iy];
				    return
				end
				function matsquare2D_AMDGPU!(A)
				    ix = (AMDGPU.blockIdx().x-1) * AMDGPU.blockDim().x + AMDGPU.threadIdx().x
				    iy = (AMDGPU.blockIdx().y-1) * AMDGPU.blockDim().y + AMDGPU.threadIdx().y
					A[ix,iy] = A[ix,iy] * A[ix,iy];
				    return
				end
				A.data.=3;                 @roc gridsize=size(A) matsquare2D_AMDGPU!(A); AMDGPU.synchronize(); @test all(Base.Array(A.data) .== 9)
				J.data.=3; J_ref.data.=36; @roc gridsize=size(J) matsquare2D_AMDGPU!(J); AMDGPU.synchronize(); @test AMDGPU.@allowscalar all(J .== J_ref)
				C.data.=2; G.data.=3;      @roc gridsize=size(C) add2D_AMDGPU!(C, G);    AMDGPU.synchronize(); @test all(Base.Array(C.data) .== 5)
			end
			if array_type == "Metal"
				function add2D_Metal!(A, B)
				    ix = (Metal.threadgroup_position_in_grid_3d().x-1) * Metal.threads_per_threadgroup_3d().x + Metal.thread_position_in_threadgroup_3d().x
				    iy = (Metal.threadgroup_position_in_grid_3d().y-1) * Metal.threads_per_threadgroup_3d().y + Metal.thread_position_in_threadgroup_3d().y
				    A[ix,iy] = A[ix,iy] + B[ix,iy];
				    return
				end
				function matsquare2D_Metal!(A)
				    ix = (Metal.threadgroup_position_in_grid_3d().x-1) * Metal.threads_per_threadgroup_3d().x + Metal.thread_position_in_threadgroup_3d().x
				    iy = (Metal.threadgroup_position_in_grid_3d().y-1) * Metal.threads_per_threadgroup_3d().y + Metal.thread_position_in_threadgroup_3d().y
					A[ix,iy] = A[ix,iy] * A[ix,iy];
				    return
				end
				A.data.=3;                 @metal groups=size(A) matsquare2D_Metal!(A); Metal.synchronize(); @test all(Base.Array(A.data) .== 9)
				J.data.=3; J_ref.data.=36; @metal groups=size(J) matsquare2D_Metal!(J); Metal.synchronize(); @test Metal.@allowscalar all(J .== J_ref)
				C.data.=2; G.data.=3;      @metal groups=size(C) add2D_Metal!(C, G);    Metal.synchronize(); @test all(Base.Array(C.data) .== 5)
			end
        end;
		@testset "cellsize" begin
			@test cellsize(A) == (1,)
			@test cellsize(B) == (1,)
			@test cellsize(C) == celldims
			@test cellsize(D) == celldims
			@test cellsize(E) == size(MyFieldArray)
			@test cellsize(F) == size(MyFieldArray)
			@test cellsize(G) == celldims
			@test cellsize(H) == celldims
		end;
		@testset "blocklength" begin
			@test blocklength(A) == prod(dims)
			@test blocklength(B) == prod(dims)
			@test blocklength(C) == prod(dims)
			@test blocklength(D) == prod(dims)
			@test blocklength(E) == prod(dims)
			@test blocklength(F) == prod(dims)
			@test blocklength(G) == 1
			@test blocklength(H) == 4
		end;
    end;
	@testset "3. Exceptions ($array_type arrays) (precision: $(nameof(Float))" for (array_type, Array, CellArray, allowscalar, Float) in zip(array_types, ArrayConstructors, CellArrayConstructors, allowscalar_functions, precision_types) 
		dims       = (2,3)
		celldims   = (3,4)
		T_Float  = SMatrix{celldims..., Float, prod(celldims)}
		T_Int32    = SMatrix{celldims...,   Int32, prod(celldims)}
		T2_Float = MyFieldArray{Float}
		T2_Int32   = MyFieldArray{Int32}
		A = CellArray{Float}(undef, dims)
		B = CellArrays.CellArray{Int32,prod(dims)}(Array, undef, dims)
		C = CellArray{T_Float}(undef, dims)
		D = CellArray{T_Int32,prod(dims)}(undef, dims)
		E = CellArray{T2_Float}(undef, dims)
		F = CellArray{T2_Int32,prod(dims)}(undef, dims)
		G = CellArray{T_Float,1}(undef, dims)
		H = CellArray{T_Int32,4}(undef, dims)
		@test_throws TypeError CellArray{Array}(undef, dims)                                                                           # Error: TypeError (celltype T is restricted to `Cell` in the package)
		@test_throws TypeError CellArray{MMatrix{celldims..., Float, prod(celldims)}}(undef, dims)                                   # ...
		@test_throws ArgumentError CellArray{MyMutableFieldArray}(undef, dims)                                                         # Error: the celltype, `T`, must be a bitstype.
		@test_throws IncoherentArgumentError CellArrays.CellArray{Float,2,0}(similar(A.data, Int64), A.dims)                # Error: eltype(data) must match eltype(T).
		@test_throws IncoherentArgumentError CellArrays.CellArray{Int32,2,prod(dims)}(similar(B.data, Float), B.dims)         # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T_Float,2,0}(similar(C.data, Int64), C.dims)              # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T_Int32,2,prod(dims)}(similar(D.data, T_Int32), D.dims)       # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T2_Float,2,0}(similar(E.data, Int64), E.dims)             # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T2_Int32,2,prod(dims)}(similar(F.data, T2_Int32), F.dims)     # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T_Float,2,1}(similar(G.data, Int64), G.dims)              # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T_Int32,2,4}(similar(H.data, T_Int32), H.dims)                # ...
		@test_throws MethodError CellArrays.CellArray{Float,2,0}(A.data[:], A.dims)                                           # Error: ndims(data) must be 3.
		@test_throws MethodError CellArrays.CellArray{T_Float,2,0}(C.data[:], C.dims)                                         # Error: ...
		@test_throws MethodError CellArrays.CellArray{T2_Float,2,0}(E.data[:], E.dims)                                        # Error: ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{Float,2,0}(similar(A.data, Float, (1,2,1)), A.dims)       # Error: size(data) must match (blocklen, prod(size(T), ceil(prod(dims)/blocklen)).
		@test_throws IncoherentArgumentError CellArrays.CellArray{T_Float,2,0}(similar(C.data, Float, (1,2,1)), C.dims)     # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T2_Float,2,0}(similar(E.data, Float, (1,2,1)), E.dims)    # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{SMatrix{(4,5)..., Float, prod((4,5))},2,0}(C.data, C.dims)  # ...
	end;
end;