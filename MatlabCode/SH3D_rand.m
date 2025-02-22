clc;
clear;
close all;
fclose('all');

%% Parameter Initialization
FigDraw = 0 ; % Enable/Disable visualization (1 = On, 0 = Off)

% Spatial Parameters
Nx = 32; %80 % 64; % Grid size in x direction
Ny = Nx; %80; % 64; % Grid size in y direction
Nz = Nx; %80; % 64; % Grid size in z direction
Lx = 3; %90; % 64 % Domain size in x direction
Ly = 3; %90; % 64; % Domain size in y direction
Lz = 3; %90; % 64; % Domain size in z direction
hx = Lx / Nx;
hy = Ly / Ny;
hz = Lz / Nz;

x = linspace(-0.5 * Lx + hx, 0.5 * Lx, Nx);
y = linspace(-0.5 * Ly + hy, 0.5 * Ly, Ny);
z = linspace(-0.5 * Lz + hz, 0.5 * Lz, Nz);
[xx, yy, zz] = ndgrid(x, y, z);

% Constant
epsilon = 0.15;

% Discrete Fourier Transform
kx = 2 * pi / Lx * [0:Nx/2, -Nx/2+1:-1];
ky = 2 * pi / Ly * [0:Ny/2, -Ny/2+1:-1];
kz = 2 * pi / Lz * [0:Nz/2, -Nz/2+1:-1];
k2x = kx.^2;
k2y = ky.^2;
k2z = kz.^2;
[kxx, kyy, kzz] = ndgrid(k2x, k2y, k2z);

% Time Discretization
dt = 0.0000002; % Time step
Nt = 100; %1000; % Total number of time steps
num_saved_steps = 101; %101; % Number of saved time steps
ns = Nt / (num_saved_steps - 1); % Save interval

% Dataset
data_size = 1500; % Number of random datasets
binary_filename = "SH3D_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
                  "_Nx_" + num2str(Nx) + ".bin";
mat_filename = "SH3D_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
               "_Nx_" + num2str(Nx) + ".mat";

%% Prepare Binary File
fileID = fopen(binary_filename, 'wb');
if fileID == -1
    error("Cannot open binary file for writing.");
end

%% Random Initial Condition Parameters
tau = 315; % 10; % Controls length scale of fluctuations
alpha = 115; % Controls smoothness of fluctuations

%% Simulation Loop
if FigDraw
    figure;
end

for data_num = 1:data_size
    disp("Data number = " + num2str(data_num));

    % Generate random initial condition using GRF3D
    norm_a = GRF3D(alpha, tau, Nx);
    %u = zeros(Nx, Ny, Nz);
    %u(norm_a >= 0) = 1;
    %u(norm_a < 0) = -1;
    norm_a = norm_a - 0.85* std(norm_a(:));   %
    u = ones(Nx,Ny,Nz);
    u(norm_a < 0) = -1;
    % Initialize storage for saving time steps
    all_iterations = zeros(num_saved_steps, Nx, Ny, Nz, 'single');

    %% Update Loop
    save_idx = 1;
    for iter = 1:Nt
        if iter == 1 || mod(iter, ns) == 0 || iter == Nt
            all_iterations(save_idx, :, :, :) = u;
            save_idx = save_idx + 1;
        end

        %% Visualization (Plot Section)
        if FigDraw && mod(iter, ns) == 0
            clf;
            p1 = patch(isosurface(xx, yy, zz, real(u), 0));
            set(p1, 'FaceColor', 'r', 'EdgeColor', 'none'); % Blue surface
            daspect([1 1 1]);
            camlight;
            lighting phong;
            box on;
            axis image;
            view(45, 45); % Consistent viewing angle
            pause(0.01); % Adjust for rendering speed
        end

        % Time evolution of the SH3D equation
        u = real(u);
        s_hat = fftn(u / dt) - fftn(u.^3) + 2 * (kxx + kyy + kzz) .* fftn(u);
        v_hat = s_hat ./ (1.0 / dt + (1 - epsilon) + (kxx + kyy + kzz).^2);
        u = ifftn(v_hat);
    end

    % Write this dataset to binary file
    fwrite(fileID, all_iterations, 'single');
end

fclose(fileID);

%% Convert Binary Data to MAT File
fileID = fopen(binary_filename, 'rb');
if fileID == -1
    error("Cannot open binary file for reading.");
end

phi_mat = matfile(mat_filename, 'Writable', true);
phi_mat.phi = zeros([data_size, num_saved_steps, Nx, Ny, Nz], 'single');

for data_num = 1:data_size
    disp("Saving dataset " + num2str(data_num));
    data_chunk = fread(fileID, num_saved_steps * Nx * Ny * Nz, 'single');
    phi_mat.phi(data_num, :, :, :, :) = reshape(data_chunk, [1, num_saved_steps, Nx, Ny, Nz]);
end

fclose(fileID);

disp('Simulation complete.');
