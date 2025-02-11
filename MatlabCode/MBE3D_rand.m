clc;
clear;
close all;
fclose('all');

disp('START MBE3D')

%% Parameter Initialization
FigDraw = 0;

% Spatial Parameters
Nx = 32; 
Ny = 32;
Nz = 32;
Lx = 2 * pi;
Ly = 2 * pi;
Lz = 2 * pi;
hx = Lx / Nx;
hy = Ly / Ny;
hz = Lz / Nz;

x = linspace(-0.5 * Lx + hx, 0.5 * Lx, Nx);
y = linspace(-0.5 * Ly + hy, 0.5 * Ly, Ny);
z = linspace(-0.5 * Lz + hz, 0.5 * Lz, Nz);
[xx, yy, zz] = ndgrid(x, y, z);

% Constant
epsilon = 0.1;

% Discrete Fourier Transform
kx = 1i * 2 * pi / Lx * [0:Nx/2 -Nx/2+1:-1];
ky = 1i * 2 * pi / Ly * [0:Ny/2 -Ny/2+1:-1];
kz = 1i * 2 * pi / Lz * [0:Nz/2 -Nz/2+1:-1];
[kxx, kyy, kzz] = ndgrid(kx, ky, kz);
k2x = (2 * pi / Lx * [0:Nx/2 -Nx/2+1:-1]).^2;
k2y = (2 * pi / Ly * [0:Ny/2 -Ny/2+1:-1]).^2;
k2z = (2 * pi / Lz * [0:Nz/2 -Nz/2+1:-1]).^2;
[kxx2, kyy2, kzz2] = ndgrid(k2x, k2y, k2z);

% Time Discretization
dt = 0.002;
Nt = 200; %2500; %25000;
num_saved_steps = 101;
ns = Nt / (num_saved_steps - 1);

Nts = linspace(0, 1, num_saved_steps); 
Nts = round(1 + (Nt - 1) * Nts.^2);
Nts = unique(Nts);
size(Nts)

% Dataset
data_size = 2000;
binary_filename = "MBE3D_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + "_Nx_" + num2str(Nx) + ".bin";
mat_filename = "MBE3D_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + "_Nx_" + num2str(Nx) + ".mat";

%% Prepare Binary File
fileID = fopen(binary_filename, 'wb');
if fileID == -1
    error("Cannot open binary file for writing.");
end

%% Initial Condition
tau = 150;
alpha = 100.0;

for data_num = 1:data_size
    %tic;
    disp("data number = " + num2str(data_num))
    
    u = 10 * GRF3D(alpha, tau, Nx);
    all_iterations = zeros(num_saved_steps, Nx, Ny, Nz, 'single');
    
    save_idx = 1;
    iter = 1;
    while iter <= Nt
        if iter == 1 || ismember(iter, Nts) || iter == Nt
            all_iterations(save_idx, :, :, :) = u;
            save_idx = save_idx + 1;
        end

        u = real(u);
        tu = fftn(u);
        fx = real(ifftn(kxx .* tu));
        fy = real(ifftn(kyy .* tu));
        fz = real(ifftn(kzz .* tu));
        
        f1 = (fx.^2 + fy.^2 + fz.^2) .* fx;
        f2 = (fx.^2 + fy.^2 + fz.^2) .* fy;
        f3 = (fx.^2 + fy.^2 + fz.^2) .* fz;
        
        s_hat = fftn(u / dt) + (kxx .* fftn(f1) + kyy .* fftn(f2) + kzz .* fftn(f3));
        v_hat = s_hat ./ (1 / dt - (kxx2 + kyy2 + kzz2) + epsilon * (kxx2 + kyy2 + kzz2).^2);
        u = ifftn(v_hat);

        if isnan(sum(u(:)))
            disp('The iteration is repeated')
            u = 10 * GRF3D(alpha, tau, Nx);
            all_iterations = zeros(num_saved_steps, Nx, Ny, Nz, 'single');
            save_idx = 1;
            iter = 1;
        end
        
        iter = iter + 1;
    end
    
    %disp('Writing')
    fwrite(fileID, all_iterations, 'single');
    %toc;
end

%% Convert Binary Data to MAT File
fclose(fileID);
fileID = fopen(binary_filename, 'rb');
if fileID == -1
    error("Cannot open binary file for reading.");
end

phi_mat = matfile(mat_filename, 'Writable', true);
phi_mat.phi = zeros([data_size, num_saved_steps, Nx, Ny, Nz], 'single');

for data_num = 1:data_size
    disp("Saving dataset " +  num2str(data_num));
    data_chunk = fread(fileID, num_saved_steps * Nx * Ny * Nz, 'single');
    phi_mat.phi(data_num, :, :, :, :) = reshape(data_chunk, [1, num_saved_steps, Nx, Ny, Nz]);
end

fclose(fileID);

disp('MBE3D Simulation Complete!')
