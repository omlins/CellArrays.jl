using CellArrays, StaticArrays, CUDA

@define_CuCellArray

function copy3D!(T2::CellArray, T::CellArray, Ci::CellArray)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    iz = (blockIdx().z-1) * blockDim().z + threadIdx().z
    T2[ix,iy,iz] = T[ix,iy,iz] .+ Ci[ix,iy,iz];
    return
end


function memcopy3D()
# Numerics
nx, ny, nz = 128, 128, 1024
celldims   = (4, 4)
nt         = 100;

# Array initializations
Cell = SMatrix{celldims..., Float64, prod(celldims)}
T  = CuCellArray{Cell}(undef, nx, ny, nz);
T2 = CuCellArray{Cell}(undef, nx, ny, nz);
Ci = CuCellArray{Cell}(undef, nx, ny, nz);

# Initial conditions
Ci.data .= 0.5;
T.data  .= 1.7;
copy!(T2.data, T.data);

# GPU launch parameters
threads = (32, 8, 1)
blocks  = (nx, ny, nz) .÷ threads

# Time loop
for it = 1:nt
    if (it == 11) global t0=time(); end
    @cuda blocks=blocks threads=threads copy3D!(T2, T, Ci);
    synchronize()
    T, T2 = T2, T;
end
time_s=time()-t0

# Performance
A_eff = (2*1+1)*1/1e9*nx*ny*nz*prod(celldims)*sizeof(Float64);
t_it  = time_s/(nt-10);
T_eff = A_eff/t_it;
println("time_s=$time_s T_eff=$T_eff");
end

memcopy3D()
