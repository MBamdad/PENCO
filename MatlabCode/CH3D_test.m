clc;
clear;
close all
fclose('all');

%% Parameter Initialization

FigDraw = 1; % Set to 1 to enable visualization

% Spatial Parameters
Nx = 80; Ny = 80; Nz = 80; % Based on the photo
Lx = 1.2; Ly = 1.2; Lz = 1.2; % Domain size
hx = Lx / Nx;
hy = Ly / Ny;
hz = Lz / Nz;
x = linspace(-0.5 * Lx + hx, 0.5 * Lx, Nx);
y = linspace(-0.5 * Ly + hy, 0.5 * Ly, Ny);
z = linspace(-0.5 * Lz + hz, 0.5 * Lz, Nz);
[xx, yy, zz] = ndgrid(x, y, z);

% Interfacial energy constant
epsilon = 2.5 * hx; % Based on photo (2.5h)
Cahn = epsilon^2;

% Discrete Fourier Transform
kx = 2 * pi / Lx * [0:Nx/2 -Nx/2+1:-1];
ky = 2 * pi / Ly * [0:Ny/2 -Ny/2+1:-1];
kz = 2 * pi / Lz * [0:Nz/2 -Nz/2+1:-1];
k2x = kx.^2; 
k2y = ky.^2; 
k2z = kz.^2;
[kxx, kyy, kzz] = ndgrid(k2x, k2y, k2z);

% Time Discretization
dt = 0.0001; % Time step
Nt = 10000; % Number of time steps
T = Nt * dt; % Total simulation time
num_saved_steps = 101;
ns = Nt / (num_saved_steps - 1);

% Dataset
data_size = 1; % Number of datasets to generate
binary_filename = "CH3D_Figure_Parameters.bin";
mat_filename = "CH3D_Figure_Parameters.mat";

%% Prepare Binary File
fileID = fopen(binary_filename, 'wb');
if fileID == -1
    error("Cannot open binary file for writing.");
end

%% Initial Condition
tau = 3.5; % Correlation length for random initial condition
alpha = 2; % Amplitude for random initial condition
u_m = -0.3;
if FigDraw
    figure;
end

for data_num = 1:data_size
    disp("Data number = " + num2str(data_num))
    
    % Generate initial random field with Gaussian random noise
    %norm_a = GRF3D(alpha, tau, Nx); % Replace with your GRF3D function
    %norm_a = norm_a + 0.2 * std(norm_a(:));   
    %u = ones(Nx, Ny, Nz);
    %u(norm_a < 0) = -1;
    u = u_m + 0.15*rand(Nx,Ny,Nz);
    % Set boundary conditions
    %u(:,1,:) = 1; u(:,end,:) = 1;
    %u(:,:,1) = 1; u(:,:,end,:) = 1;

    all_iterations = zeros(num_saved_steps, Nx, Ny, Nz, 'single');
    
    %% Initial Preview
    if FigDraw
        clf;
        p1 = patch(isosurface(xx, yy, zz, real(u), 0));
        set(p1, 'FaceColor', 'r', 'EdgeColor', 'none'); % Red color for the surface
        daspect([1 1 1]);
        camlight;
        lighting phong;
        box on;
        axis image;
        view(45, 45); % Set the viewing angle
        pause(2);
    end

    %% Update
    save_idx = 1;
    for iter = 1:Nt
        if iter == 1 || mod(iter, ns) == 0 || iter == Nt
            all_iterations(save_idx, :, :, :) = u;
            save_idx = save_idx + 1;
        end

        % Perform Cahn-Hilliard update
        u = real(u);
        s_hat = fftn(u) - dt * (kxx + kyy + kzz) .* fftn(u.^3 - 3 * u);
        v_hat = s_hat ./ (1.0 + dt * (2.0 * (kxx + kyy + kzz) + Cahn * (kxx + kyy + kzz).^2));
        u = ifftn(v_hat);

        % Update Visualization During Simulation
        if FigDraw && mod(iter, ns) == 0
            figure(1);
            clf;
            p1 = patch(isosurface(xx, yy, zz, real(u), 0));
            set(p1, 'FaceColor', 'r', 'EdgeColor', 'none'); % Red color for the surface
            daspect([1 1 1]);
            camlight;
            lighting phong;
            box on;
            axis image;
            view(45, 45); % Maintain consistent view
            pause(0.01);
        end
    end

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
    disp("Saving dataset " +  num2str(data_num));
    data_chunk = fread(fileID, num_saved_steps * Nx * Ny * Nz, 'single');
    phi_mat.phi(data_num, :, :, :, :) = reshape(data_chunk, [1, num_saved_steps, Nx, Ny, Nz]);
end

fclose(fileID);
