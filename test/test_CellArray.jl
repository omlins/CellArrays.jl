using Test
using CUDA, AMDGPU, StaticArrays
import CellArrays
import CellArrays: CPUCellArray, CuCellArray, ROCCellArray, cellsize, blocklength, _N
import CellArrays: IncoherentArgumentError, ArgumentError


test_cuda = CUDA.functional()
test_amdgpu = AMDGPU.functional()

array_types           = ["CPU"]
ArrayConstructors     = [Array]
CellArrayConstructors = [CPUCellArray]
allowscalar_functions = Function[(x->x)]
if test_cuda
	cuzeros = CUDA.zeros
	push!(array_types, "CUDA")
	push!(ArrayConstructors, CuArray)
	push!(CellArrayConstructors, CuCellArray)
	push!(allowscalar_functions, CUDA.allowscalar)
end
if test_amdgpu
	roczeros = AMDGPU.zeros
	push!(array_types, "AMDGPU")
	push!(ArrayConstructors, ROCArray)
	push!(CellArrayConstructors, ROCCellArray)
	push!(allowscalar_functions, AMDGPU.allowscalar) #TODO: to be implemented
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
    @testset "1. CellArray allocation ($array_type arrays)" for (array_type, Array, CellArray) in zip(array_types, ArrayConstructors, CellArrayConstructors)
        @testset "Number cells" begin
			dims = (2,3)
			A = CellArray{Float64}(undef, dims)
			B = CellArrays.CellArray{Int32,prod(dims)}(Array, undef, dims...)
			C = CellArray{Float64,1}(undef, dims)
			D = CellArray{Int32,4}(undef, dims)
			@test typeof(A.data) <: Array
			@test typeof(B.data) <: Array
			@test typeof(C.data) <: Array
			@test typeof(D.data) <: Array
			@test eltype(A.data) == Float64
			@test eltype(B.data) == Int32
			@test eltype(C.data) == Float64
			@test eltype(D.data) == Int32
			@test eltype(A)      == Float64
			@test eltype(B)      == Int32
			@test eltype(C)      == Float64
			@test eltype(D)      == Int32
			@test typeof(A)      == CellArrays.CellArray{Float64, length(dims), 0, Array{eltype(A.data),_N}}
			@test typeof(B)      == CellArrays.CellArray{Int32, length(dims), prod(dims), Array{eltype(B.data),_N}}
			@test typeof(C)      == CellArrays.CellArray{Float64, length(dims), 1, Array{eltype(C.data),_N}}
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
			T_Float64 = SMatrix{celldims..., Float64, prod(celldims)}
			T_Int32   = SMatrix{celldims...,   Int32, prod(celldims)}
			A = CellArray{T_Float64}(undef, dims)
			B = CellArrays.CellArray{T_Int32,prod(dims)}(Array, undef, dims...)
			C = CellArray{T_Float64,1}(undef, dims)
			D = CellArray{T_Int32,4}(undef, dims)
			@test typeof(A.data) <: Array
			@test typeof(B.data) <: Array
			@test typeof(C.data) <: Array
			@test typeof(D.data) <: Array
			@test eltype(A.data) == Float64
			@test eltype(B.data) == Int32
			@test eltype(C.data) == Float64
			@test eltype(D.data) == Int32
			@test eltype(A)      == T_Float64
			@test eltype(B)      == T_Int32
			@test eltype(C)      == T_Float64
			@test eltype(D)      == T_Int32
			@test typeof(A)      == CellArrays.CellArray{T_Float64, length(dims), 0, Array{eltype(A.data),_N}}
			@test typeof(B)      == CellArrays.CellArray{T_Int32, length(dims), prod(dims), Array{eltype(B.data),_N}}
			@test typeof(C)      == CellArrays.CellArray{T_Float64, length(dims), 1, Array{eltype(C.data),_N}}
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
			T_Float64 = MyFieldArray{Float64}
			T_Int32   = MyFieldArray{Int32}
			A = CellArray{T_Float64}(undef, dims)
			B = CellArrays.CellArray{T_Int32,prod(dims)}(Array, undef, dims...)
			C = CellArray{T_Float64,1}(undef, dims)
			D = CellArray{T_Int32,4}(undef, dims)
			@test typeof(A.data) <: Array
			@test typeof(B.data) <: Array
			@test typeof(C.data) <: Array
			@test typeof(D.data) <: Array
			@test eltype(A.data) == Float64
			@test eltype(B.data) == Int32
			@test eltype(C.data) == Float64
			@test eltype(D.data) == Int32
			@test eltype(A)      == T_Float64
			@test eltype(B)      == T_Int32
			@test eltype(C)      == T_Float64
			@test eltype(D)      == T_Int32
			@test typeof(A)      == CellArrays.CellArray{T_Float64, length(dims), 0, Array{eltype(A.data),_N}}
			@test typeof(B)      == CellArrays.CellArray{T_Int32, length(dims), prod(dims), Array{eltype(B.data),_N}}
			@test typeof(C)      == CellArrays.CellArray{T_Float64, length(dims), 1, Array{eltype(C.data),_N}}
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
	@testset "2. functions ($array_type arrays)" for (array_type, Array, CellArray, allowscalar) in zip(array_types, ArrayConstructors, CellArrayConstructors, allowscalar_functions)
		dims      = (2,3)
		celldims  = (3,4) # Needs to be compatible for matrix multiplication!
		T_Float64 = SMatrix{celldims..., Float64, prod(celldims)}
		T_Int32   = SMatrix{celldims...,   Int32, prod(celldims)}
		T2_Float64 = MyFieldArray{Float64}
		T2_Int32   = MyFieldArray{Int32}
		A = CellArray{Float64}(undef, dims)
		B = CellArrays.CellArray{Int32,prod(dims)}(Array, undef, dims)
		C = CellArray{T_Float64}(undef, dims)
		D = CellArray{T_Int32,prod(dims)}(undef, dims)
		E = CellArray{T2_Float64}(undef, dims)
		F = CellArray{T2_Int32,prod(dims)}(undef, dims)
		G = CellArray{T_Float64,1}(undef, dims)
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
			allowscalar(true)
			fill!(A, 9);   @test all(Base.Array(A.data) .== 9.0)
			fill!(B, 9.0); @test all(Base.Array(B.data) .== 9)
			fill!(C, (1:length(eltype(C)))); @test all(C .== (T_Float64(1:length(eltype(C)))  for i=1:dims[1], j=1:dims[2]))
			fill!(D, (1:length(eltype(D)))); @test all(D .== (T_Int32(1:length(eltype(D)))    for i=1:dims[1], j=1:dims[2]))
			fill!(E, (1:length(eltype(E)))); @test all(E .== (T2_Float64(1:length(eltype(E))) for i=1:dims[1], j=1:dims[2]))
			fill!(F, (1:length(eltype(F)))); @test all(F .== (T2_Int32(1:length(eltype(F)))   for i=1:dims[1], j=1:dims[2]))
			fill!(G, (1:length(eltype(G)))); @test all(G .== (T_Float64(1:length(eltype(G)))  for i=1:dims[1], j=1:dims[2]))
			fill!(H, (1:length(eltype(H)))); @test all(H .== (T_Int32(1:length(eltype(H)))    for i=1:dims[1], j=1:dims[2]))
			allowscalar(false)
		end
		@testset "getindex / setindex! (array programming)" begin
			allowscalar(true)
			A.data.=0; B.data.=0; C.data.=0; D.data.=0; E.data.=0; F.data.=0; G.data.=0; H.data.=0;
			A[2,2:3] .= 9
			B[2,2:3] .= 9.0
			C[2,2:3] .= (T_Float64(1:length(T_Float64)), T_Float64(1:length(T_Float64)))
			D[2,2:3] .= (T_Int32(1:length(T_Int32)), T_Int32(1:length(T_Int32)))
			E[2,2:3] .= (T2_Float64(1:length(T2_Float64)), T2_Float64(1:length(T2_Float64)))
			F[2,2:3] .= (T2_Int32(1:length(T2_Int32)), T2_Int32(1:length(T2_Int32)))
			G[2,2:3] .= (T_Float64(1:length(T_Float64)), T_Float64(1:length(T_Float64)))
			H[2,2:3] .= (T_Int32(1:length(T_Int32)), T_Int32(1:length(T_Int32)))
			@test all(A[2,2:3] .== 9.0)
			@test all(B[2,2:3] .== 9)
			@test all(C[2,2:3] .== (T_Float64(1:length(T_Float64)), T_Float64(1:length(T_Float64))))
			@test all(D[2,2:3] .== (T_Int32(1:length(T_Int32)), T_Int32(1:length(T_Int32))))
			@test all(E[2,2:3] .== (T2_Float64(1:length(T2_Float64)), T2_Float64(1:length(T2_Float64))))
			@test all(F[2,2:3] .== (T2_Int32(1:length(T2_Int32)), T2_Int32(1:length(T2_Int32))))
			@test all(G[2,2:3] .== (T_Float64(1:length(T_Float64)), T_Float64(1:length(T_Float64))))
			@test all(H[2,2:3] .== (T_Int32(1:length(T_Int32)), T_Int32(1:length(T_Int32))))
			allowscalar(false)
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
	@testset "3. Exceptions ($array_type arrays)" for (array_type, Array, CellArray) in zip(array_types, ArrayConstructors, CellArrayConstructors)
		dims       = (2,3)
		celldims   = (3,4)
		T_Float64  = SMatrix{celldims..., Float64, prod(celldims)}
		T_Int32    = SMatrix{celldims...,   Int32, prod(celldims)}
		T2_Float64 = MyFieldArray{Float64}
		T2_Int32   = MyFieldArray{Int32}
		A = CellArray{Float64}(undef, dims)
		B = CellArrays.CellArray{Int32,prod(dims)}(Array, undef, dims)
		C = CellArray{T_Float64}(undef, dims)
		D = CellArray{T_Int32,prod(dims)}(undef, dims)
		E = CellArray{T2_Float64}(undef, dims)
		F = CellArray{T2_Int32,prod(dims)}(undef, dims)
		G = CellArray{T_Float64,1}(undef, dims)
		H = CellArray{T_Int32,4}(undef, dims)
		@test_throws TypeError CellArray{Array}(undef, dims)                                                                           # Error: TypeError (celltype T is restricted to `Cell` in the package)
		@test_throws TypeError CellArray{MMatrix{celldims..., Float64, prod(celldims)}}(undef, dims)                                   # ...
		@test_throws ArgumentError CellArray{MyMutableFieldArray}(undef, dims)                                                         # Error: the celltype, `T`, must be a bitstype.
		@test_throws IncoherentArgumentError CellArrays.CellArray{Float64,2,0}(similar(A.data, Float32), A.dims)                # Error: eltype(data) must match eltype(T).
		@test_throws IncoherentArgumentError CellArrays.CellArray{Int32,2,prod(dims)}(similar(B.data, Float64), B.dims)         # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T_Float64,2,0}(similar(C.data, Float32), C.dims)              # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T_Int32,2,prod(dims)}(similar(D.data, T_Int32), D.dims)       # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T2_Float64,2,0}(similar(E.data, Float32), E.dims)             # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T2_Int32,2,prod(dims)}(similar(F.data, T2_Int32), F.dims)     # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T_Float64,2,1}(similar(G.data, Float32), G.dims)              # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T_Int32,2,4}(similar(H.data, T_Int32), H.dims)                # ...
		@test_throws MethodError CellArrays.CellArray{Float64,2,0}(A.data[:], A.dims)                                           # Error: ndims(data) must be 3.
		@test_throws MethodError CellArrays.CellArray{T_Float64,2,0}(C.data[:], C.dims)                                         # Error: ...
		@test_throws MethodError CellArrays.CellArray{T2_Float64,2,0}(E.data[:], E.dims)                                        # Error: ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{Float64,2,0}(similar(A.data, Float64, (1,2,1)), A.dims)       # Error: size(data) must match (blocklen, prod(size(T), ceil(prod(dims)/blocklen)).
		@test_throws IncoherentArgumentError CellArrays.CellArray{T_Float64,2,0}(similar(C.data, Float64, (1,2,1)), C.dims)     # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{T2_Float64,2,0}(similar(E.data, Float64, (1,2,1)), E.dims)    # ...
		@test_throws IncoherentArgumentError CellArrays.CellArray{SMatrix{(4,5)..., Float64, prod((4,5))},2,0}(C.data, C.dims)  # ...
	end;
end;
