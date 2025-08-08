clc;
clear;
close all;
fclose('all');
disp('START PFC3D')

%% Parameter Initialization
FigDraw = 0; % Enable/Disable visualization (1 = On, 0 = Off)

% Spatial Parameters
Nx = 32; %42; %32; %64;
Ny = Nx;
Nz = Nx;
Lx = 10*pi; %  20; %12; %3; %64;
Ly = Lx;
Lz = Lx;
hx = Lx/Nx;
hy = Ly/Ny;
hz = Lz/Nz;

x = linspace(-0.5*Lx+hx, 0.5*Lx, Nx);
y = linspace(-0.5*Ly+hy, 0.5*Ly, Ny);
z = linspace(-0.5*Lz+hz, 0.5*Lz, Nz);
[xx, yy, zz] = ndgrid(x, y, z); % For plotting

% Constant
epsilon = 0.5;

% Discrete Fourier Transform
p = 2*pi/Lx*[0:Nx/2 -Nx/2+1:-1];
q = 2*pi/Ly*[0:Ny/2 -Ny/2+1:-1];
r = 2*pi/Lz*[0:Nz/2 -Nz/2+1:-1];
p2 = p.^2;
q2 = q.^2;
r2 = r.^2;
[pp2, qq2, rr2] = ndgrid(p2, q2, r2);

% Time Discretization
dt = 0.005; %0.1;
Nt = 100; %200; %10000;
num_saved_steps = 101;
ns = Nt / (num_saved_steps - 1);

% Dataset
data_size = 1500; %4440;
binary_filename = "PFC3D_Augmented_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
                  "_Nx_" + num2str(Nx) + ".bin";
mat_filename = "PFC3D_Augmented_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
               "_Nx_" + num2str(Nx) + ".mat";

%% Prepare Binary File
fileID = fopen(binary_filename, 'wb');
if fileID == -1
    error("Cannot open binary file for writing.");
end


%% Initial Condition

u_mean = 0.01;
tau = 315; %115; %3.5;
alpha = 115; %45; %2.0;

if FigDraw
    figure;
end

for data_num = 1:data_size
    %tic;
    disp("data number = " + num2str(data_num))

    % ==================== MODIFIED IC SECTION ====================
    % Strategy: 80% random GRF, 20% star-shaped initial condition
    if rand() <= 0.2  % 20% chance to generate a star
        %disp("   -> Generating a random STAR");
        
        % Define random parameters for the star
        interface_width = sqrt(2.0) * epsilon;
        
        % Randomize the number of star points (between 4 and 8)
        num_points = randi([4, 8]);
        
        % Randomize the amplitude of star points (between 0.5 and 2.0)
        amplitude = 0.5 + 1.5 * rand();
        
        % Randomize the star size (major radius between 3 and 6)
        R_base = 3 + 3 * rand();
        
        % Create star shape
        theta = atan2(zz, xx);
        R_theta = R_base + amplitude * cos(num_points * theta); 
        dist = sqrt(xx.^2 + 2*yy.^2 + zz.^2);  % Slightly elliptical in y-direction
        
        % Randomize the interface width slightly
        effective_interface = interface_width * (0.8 + 0.4 * rand());
        
        u = tanh((R_theta - dist) / effective_interface);
        
        % Randomly flip some stars inside-out
        if rand() > 0.5
            u = -u;
        end
        
    else % 80% chance to generate a GRF
        %disp("   -> Generating a random GRF");
        
        % Define the min/max for the random ranges
        tau_min = 290;
        tau_max = 315;
        alpha_min = 95;
        alpha_max = 120;
        
        % Generate random parameters
        current_tau = tau_min + (tau_max - tau_min) * rand();
        current_alpha = alpha_min + (alpha_max - alpha_min) * rand();

        % Generate the continuous random field
        norm_a = GRF3D(current_alpha, current_tau, Nx);
        
        % Randomize the thresholding shift
        shift_factor = 0.7 - 0.4 * rand(); % Range: [0.7, 1.2]
        norm_a = norm_a - shift_factor * std(norm_a(:));
        
        % Threshold the field
        u = ones(Nx,Ny,Nz);
        u(norm_a < 0) = -1;
    end
    % ===================== END OF MODIFIED IC SECTION =====================


    %%
    %u = u_mean - u_mean * GRF3D(alpha, tau, Nx);
    %norm_a = GRF3D(alpha, tau, Nx);
    %norm_a = norm_a - 0.85 * std(norm_a(:));
    %u = ones(Nx, Nx, Nz);
    %u(norm_a < 0) = -1;


    all_iterations = zeros(num_saved_steps, Nx, Ny, Nz, 'single');

    %% Update
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
            set(p1, 'FaceColor', 'g', 'EdgeColor', 'none'); % Blue surface
            daspect([1 1 1]);
            camlight;
            lighting phong;
            box on;
            axis image;
            view(45, 45); % Consistent viewing angle
            pause(0.01); % Adjust for rendering speed
        end
        
        %% Update u
        u = real(u);
        s_hat = fftn(u/dt) - (pp2 + qq2 + rr2) .* fftn(u.^3) + 2 * (pp2 + qq2 + rr2).^2 .* fftn(u);
        v_hat = s_hat ./ (1.0/dt + (1 - epsilon) * (pp2 + qq2 + rr2) + (pp2 + qq2 + rr2).^3);
        u = ifftn(v_hat);
    end
    
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
    disp("Saving dataset " + num2str(data_num));
    data_chunk = fread(fileID, num_saved_steps * Nx * Ny * Nz, 'single');
    phi_mat.phi(data_num, :, :, :, :) = reshape(data_chunk, [1, num_saved_steps, Nx, Ny, Nz]);
end

fclose(fileID);
