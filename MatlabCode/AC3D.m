
clc;
clear;

%% Parameter Initialization

FigDraw = 1; % Set to 1 to enable visualization, 0 to disable


% Spatial Parameters
Nx = 32;
Ny = Nx;
Nz = Nx;
Lx = 5;
Ly = Lx;
Lz = Lx;
hx = Lx / Nx;
hy = Ly / Ny;
hz = Lz / Nz;

x = linspace(-0.5 * Lx + hx, 0.5 * Lx, Nx);
y = linspace(-0.5 * Ly + hy, 0.5 * Ly, Ny);
z = linspace(-0.5 * Lz + hz, 0.5 * Lz, Nz);

[xx, yy, zz] = ndgrid(x, y, z);

% Interfacial energy constant
epsilon = 0.1;
Cahn = epsilon^2;

% Discrete Fourier Transform
kx = 2 * pi / Lx * [0:Nx / 2 -Nx / 2 + 1:-1];
ky = 2 * pi / Ly * [0:Ny / 2 -Ny / 2 + 1:-1];
kz = 2 * pi / Lz * [0:Nz / 2 -Nz / 2 + 1:-1];
k2x = kx.^2;
k2y = ky.^2;
k2z = kz.^2;
[kxx, kyy, kzz] = ndgrid(k2x, k2y, k2z);

% Time Discretization
%dt = 0.001;
%T = 100 * 0.001;
%Nt = round(T / dt);
%ns = 1;


% Time Discretization
dt = 0.0005; % Time step
Nt = 100; % Number of time steps
T = Nt * dt; % Total simulation time
num_saved_steps = 101;
ns = Nt / (num_saved_steps - 1);

% Dataset
data_size = 1500;
num_saved_steps = Nt + 1; % Save all steps
binary_filename = "AC3D_" + num2str(Nx) + "_" + num2str(data_size) + ".bin";
mat_filename = "AC3D_" + num2str(Nx) + "_" + num2str(data_size) + ".mat";

%% Prepare Binary File
fileID = fopen(binary_filename, 'wb');
if fileID == -1
    error("Cannot open binary file for writing.");
end

%% Visualization (Plot Section)


if FigDraw
    figure;
end

%% Initial Condition
tau = 315;
alpha = 115;

for data_num = 1:data_size
    disp("Data number = " + num2str(data_num));

    %%%%%
    % ==================== MODIFIED IC SECTION (AS REQUESTED) ====================
    % Strategy: Randomize GRF parameters within the user-specified ranges
    % to create a rich and diverse dataset for robust NN training.

    % 1. Define the min/max for the random ranges
    tau_min = 290;
    tau_max = 320;
    alpha_min = 90;
    alpha_max = 120;

    % 2. Generate a random value for tau and alpha in their respective ranges.
    % The formula is: min_val + (max_val - min_val) * rand()
    current_tau = tau_min + (tau_max - tau_min) * rand();
    current_alpha = alpha_min + (alpha_max - alpha_min) * rand();

    % Generate the continuous random field with these new parameters
    norm_a = GRF3D(current_alpha, current_tau, Nx);

    % 3. Randomize the thresholding shift to vary the volume fraction.
    shift_factor = 0.7 - 0.4 * rand(); % Range: [0.7, 1.2]
    norm_a = norm_a - shift_factor * std(norm_a(:));

    % Threshold the field to create the binary initial condition
    u = ones(Nx,Ny,Nz);
    u(norm_a < 0) = -1;
    % ===================== END OF MODIFIED IC SECTION =====================
    %%%%


    %norm_a = GRF3D(alpha, tau, Nx);
    %norm_a = norm_a - 0.85 * std(norm_a(:));
    %u = ones(Nx, Nx, Nz);
    %u(norm_a < 0) = -1;

    all_iterations = zeros(num_saved_steps, Nx, Ny, Nz, 'single');

    %% Initial Preview
    if FigDraw
        clf;
        p1 = patch(isosurface(xx, yy, zz, real(u), 0));
        set(p1, 'FaceColor', 'r', 'EdgeColor', 'none'); % Blue color for the surface
        daspect([1 1 1]);
        camlight;
        lighting phong;
        box on;
        axis image;
        view(45, 45); % Set the viewing angle
        pause(2);
    end

    %% Update
    for iter = 1:Nt
        if FigDraw && mod(iter, ns) == 0
            figure(1);
            clf;
            p1 = patch(isosurface(xx, yy, zz, real(u), 0));
            set(p1, 'FaceColor', 'r', 'EdgeColor', 'none'); % Blue color for the surface
            daspect([1 1 1]);
            camlight;
            lighting phong;
            box on;
            axis image;
            view(45, 45); % Maintain consistent view
            pause(0.01);
        end

        all_iterations(iter, :, :, :) = u;
        %u = real(u);

        %s_hat = fftn(Cahn * u - dt * (u.^3 - 3 * u));
        %v_hat = s_hat ./ (Cahn + dt * (2 + Cahn * (kxx + kyy + kzz)));
        %u = ifftn(v_hat);
        nonlinear_term_hat = fftn(u.^3 - u);
        u_hat = fftn(u);
        v_hat = (u_hat - (dt/Cahn) * nonlinear_term_hat) ./ (1 + dt * (kxx + kyy + kzz));
        u = ifftn(v_hat);

    end

    all_iterations(end, :, :, :) = u; % Save final step

    % Write data to binary file
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

disp("Saving dataset ");