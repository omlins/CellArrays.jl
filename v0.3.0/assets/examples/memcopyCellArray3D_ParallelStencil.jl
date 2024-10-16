const USE_GPU = true
using CellArrays, StaticArrays
import CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

@parallel function copy3D!(T2::CellArray, T::CellArray, Ci::CellArray)
    @all(T2) = @all(T) + @all(Ci);
    return
end

@parallel_indices (ix,iy,iz) function copy3D_explicit!(T2::CellArray, T::CellArray, Ci::CellArray)
    T2[ix,iy,iz] = T[ix,iy,iz] .+ Ci[ix,iy,iz];
    return
end


function memcopy3D()
# Numerics
nx, ny, nz = 128, 128, 1024
celldims   = (4, 4)
nt         = 100;

# Array initializations
T  = @zeros(nx, ny, nz, celldims=celldims);
T2 = @zeros(nx, ny, nz, celldims=celldims);
Ci = @zeros(nx, ny, nz, celldims=celldims);

# Initial conditions
@fill!(Ci, 0.5);
@fill!(T, 1.7);
copy!(T2.data, T.data);

# Time loop
for it = 1:nt
    if (it == 11) global t0=time(); end
    @parallel copy3D!(T2, T, Ci);  # or: @parallel copy3D_explicit!(T2, T, Ci);
    T, T2 = T2, T;
end
time_s=time()-t0

# Performance
A_eff = (2*1+1)*1/1e9*nx*ny*nz*prod(celldims)*sizeof(Data.Number);
t_it  = time_s/(nt-10);
T_eff = A_eff/t_it;
println("time_s=$time_s T_eff=$T_eff");
end

memcopy3D()
