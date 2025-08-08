clc;
clear;
close all;
fclose('all');
disp('START')

%% Parameter Initialization

FigDraw = 0; % Set to 1 to enable visualization

% Spatial Parameters
Nx = 32;
Ny = 32;
Nz = 32;
Lx = 2*pi;
Ly = Lx;
Lz = Lx;
hx = Lx/Nx;
hy = Ly/Ny;
hz = Lz/Nz;

x = linspace(-0.5*Lx+hx, 0.5*Lx, Nx);
y = linspace(-0.5*Ly+hy, 0.5*Ly, Ny);
z = linspace(-0.5*Lz+hz, 0.5*Lz, Nz);
[xx,yy,zz] = ndgrid(x,y,z);

% Constant
epsilon = 0.1;

% Discrete Fourier Transform
p = 1i*2*pi/Lx*[0:Nx/2-1 0 -Nx/2+1:-1];
q = 1i*2*pi/Ly*[0:Ny/2-1 0 -Ny/2+1:-1];
r = 1i*2*pi/Lz*[0:Nz/2-1 0 -Nz/2+1:-1];
[pp,qq,rr] = ndgrid(p,q,r);
p2 = (2*pi/Lx*[0:Nx/2 -Nx/2+1:-1]).^2;
q2 = (2*pi/Ly*[0:Ny/2 -Ny/2+1:-1]).^2;
r2 = (2*pi/Lz*[0:Nz/2 -Nz/2+1:-1]).^2;
[pp2,qq2,rr2] = ndgrid(p2,q2,r2);

% Time Discretization
dt = 0.001;
Nt = 100;
num_saved_steps = 101;
ns = Nt/(num_saved_steps-1);

% Dataset
data_size = 2000;
binary_filename = "MBE3D_Augmented_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
                  "_Nx_" + num2str(Nx) + ".bin";
mat_filename = "MBE3D_Augmented_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
               "_Nx_" + num2str(Nx) + ".mat";

%% Prepare Binary File
fileID = fopen(binary_filename, 'wb');
if fileID == -1
    error("Cannot open binary file for writing.");
end

%% Initial Condition (Unused, kept for reference)
tau = 150;
alpha = 100;

if FigDraw
    figure('Color', 'white');
    set(gcf, 'Position', [100, 100, 800, 600]);
end

%% Main Data Generation Loop
for data_num = 1:data_size
    tic;
    disp("Data number = " + num2str(data_num));

    % =======================================================================
    % --- MODIFICATION: Mixed Initial Conditions ---
    % =======================================================================

    if rand() <= 0.3  % 30% chance to generate a torus
        %disp("   -> Generating a random TORUS");

        % Define random parameters for the torus
        R_major_min = 1.2;
        R_major_max = 3.0; % Slightly larger range for variety
        r_minor_min = 0.3;
        r_minor_max = 0.8;

        % Generate random radii and ensure r < R
        R = R_major_min + (R_major_max - R_major_min) * rand();
        r = r_minor_min + (r_minor_max - r_minor_min) * rand();
        r = min(r, R - 0.2); % Prevents self-intersection and keeps it thick enough

        % Create the torus initial condition
        interface_width = sqrt(2) * epsilon;
        torus_dist = sqrt((sqrt(xx.^2 + yy.^2) - R).^2 + zz.^2);
        u = tanh((r - torus_dist) / interface_width);

    else % 70% chance to generate a GRF
        %disp("   -> Generating a random GRF");

        % Define random parameters for GRF (as in your original code)
        tau_min = 150;
        tau_max = 170;
        alpha_min = 60;
        alpha_max = 110;

        current_tau = tau_min + (tau_max - tau_min) * rand();
        current_alpha = alpha_min + (alpha_max - alpha_min) * rand();
        u = 10*GRF3D(current_alpha, current_tau, Nx);
    end

    % =======================================================================
    % --- END OF MODIFICATION ---
    % =======================================================================

    all_iterations = zeros(num_saved_steps, Nx, Ny, Nz, 'single');

    %% Initial Preview with Modified Visualization
    if FigDraw
        clf;
        p = patch(isosurface(xx, yy, zz, real(u), 0));
        set(p, 'FaceColor', [0.5 0.5 1], 'EdgeColor', 'none');
        daspect([1 1 1]);
        axis tight;
        grid on;
        box on;
        camlight('headlight');
        lighting gouraud;
        material dull;
        view(3);
        xlabel('x'); ylabel('y'); zlabel('z');
        title('Initial Condition');
        xlim([-pi pi]); ylim([-pi pi]); zlim([-pi pi]);
        drawnow;
        pause(0.5); % Reduced pause time
    end

    %% Update
    save_idx = 1;
    for iter = 1:Nt
        if iter == 1 || mod(iter, ns) == 0 || iter == Nt
            all_iterations(save_idx, :, :, :) = u;
            save_idx = save_idx + 1;
        end

        u = real(u);
        tu = fftn(u);
        fx = real(ifftn(pp.*tu));
        fy = real(ifftn(qq.*tu));
        fz = real(ifftn(rr.*tu));
        f1 = (fx.^2 + fy.^2 + fz.^2).*fx;
        f2 = (fx.^2 + fy.^2 + fz.^2).*fy;
        f3 = (fx.^2 + fy.^2 + fz.^2).*fz;
        s_hat = fftn(u/dt) + (pp.*fftn(f1) + qq.*fftn(f2) + rr.*fftn(f3));
        v_hat = s_hat./(1/dt - (pp2 + qq2 + rr2) + epsilon*(pp2 + qq2 + rr2).^2);
        u = ifftn(v_hat);

        if isnan(sum(u(:)))
            warning('NaN detected in simulation, skipping this sample.');
            % To avoid getting stuck, we will just skip this problematic sample
            % and continue to the next data_num. The loop will then restart.
            % We need to rewind the file pointer to overwrite this corrupted entry.
            fseek(fileID, (data_num-1)*num_saved_steps*Nx*Ny*Nz*4, 'bof'); % 4 bytes for single
            data_num = data_num - 1; % Redo this data number
            break; % Exit the inner for-loop
        end

        %% Modified Visualization During Simulation
        if FigDraw && mod(iter, ns) == 0
            clf;
            p = patch(isosurface(xx, yy, zz, real(u), 0));
            set(p, 'FaceColor', [0.5 0.5 1], 'EdgeColor', 'none');
            daspect([1 1 1]);
            axis tight;
            grid on;
            box on;
            camlight('headlight');
            lighting gouraud;
            material dull;
            view(3);
            xlabel('x'); ylabel('y'); zlabel('z');
            title(['Iteration: ' num2str(iter)]);
            xlim([-pi pi]); ylim([-pi pi]); zlim([-pi pi]);
            drawnow;
        end
    end

    % Only write to file if the simulation completed without NaN
    if ~isnan(sum(u(:)))
        fwrite(fileID, all_iterations, 'single');
    end
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
    data_chunk = fread(fileID, num_saved_steps*Nx*Ny*Nz, 'single');
    if isempty(data_chunk)
        warning(['Dataset ended prematurely at sample ', num2str(data_num), '. The final .mat file may be smaller than data_size.']);
        phi_mat.phi(data_num:end,:,:,:,:) = []; % Truncate the pre-allocated matrix
        break;
    end
    phi_mat.phi(data_num, :, :, :, :) = reshape(data_chunk, [1, num_saved_steps, Nx, Ny, Nz]);
end

fclose(fileID);
disp("Simulation and data generation complete!");