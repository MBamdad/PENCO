clc;
clear;
close all;
fclose('all');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART 0: USER CONFIGURATION - SET THESE PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choose the case study and initial condition
case_study = 'AC3D';  % Options: 'SH3D', 'AC3D', 'CH3D', 'MBE3D', 'PFC3D'
initial_condition_type = 'sphere'; % Options vary by case study (see below) (SH3D, AC3D --> sphere) , (CH3D, PFC3D --> star), MBE--> torus

% Choose which model results to load (MHNO or PI-MHNO)
use_PIMHNO = true % true % false; % Set to true to use PI-MHNO results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART 1: SETUP PARAMETERS BASED ON CASE STUDY AND INITIAL CONDITION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize common variables
python_data_file = '';
Nx = 0; Ny = 0; Nz = 0; Lx = 0; Ly = 0; Lz = 0;
epsilon = 0; dt = 0; Nt = 100; selected_frames = [0, 50, 90];
u = []; % Will hold initial condition
x_grid = []; y_grid = []; z_grid = []; xx = []; yy = []; zz = [];

% Set parameters based on case study and initial condition
switch case_study
    case 'SH3D'
        % SH3D parameters (only sphere IC available)
        initial_condition_type = 'sphere'; % SH3D only has sphere
        Nx = 32; Ny = Nx; Nz = Nx;
        Lx = 15; Ly = Lx; Lz = Lx;
        epsilon = 0.15;
        dt = 0.05;
         selected_frames = [0, 70, 90];  % Modified this line for SH3D

        % Set file path based on MHNO/PI-MHNO selection
        if use_PIMHNO
            %python_data_file = '/scratch/noqu8762/phase_field_equations_4d/SH3D_python_predictions_sphere_PIMHNO.mat';
            python_data_file = '/scratch/noqu8762/phase_field_equations_4d/In_distribution/SH3D_python_predictions_sphere_PI-MHNO.mat'
        else
            %python_data_file = '/scratch/noqu8762/phase_field_equations_4d/SH3D_python_predictions_sphere.mat';
            python_data_file = '/scratch/noqu8762/phase_field_equations_4d/In_distribution/SH3D_python_predictions_sphere_MHNO.mat'
        end

        % Create spherical initial condition
        x_grid = linspace(-Lx/2, Lx/2, Nx);
        y_grid = linspace(-Ly/2, Ly/2, Ny);
        z_grid = linspace(-Lz/2, Lz/2, Nz);
        [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
        radius = 2;
        transition_width = sqrt(2)*epsilon;
        r = sqrt(xx.^2 + yy.^2 + zz.^2);
        u = tanh((radius - r) / transition_width);

    case 'AC3D'
        % AC3D parameters (multiple ICs available)
        valid_ICs = {'sphere', 'dumbbell', 'star', 'heart'};
        if ~ismember(initial_condition_type, valid_ICs)
            error('AC3D: Invalid initial condition. Choose from: %s', strjoin(valid_ICs, ', '));
        end

        % Set file path based on initial condition and MHNO/PI-MHNO
        %base_path = '/scratch/noqu8762/phase_field_equations_4d/AC3D_python_predictions_';
        base_path = '/scratch/noqu8762/phase_field_equations_4d/OOD_MatFiles/Old_version_MatFiles/AC3D_python_predictions_'
        if use_PIMHNO
            python_data_file = [base_path initial_condition_type '_PIMHNO.mat'];
        else
            python_data_file = [base_path initial_condition_type '.mat'];
        end

        % Set parameters based on initial condition
        switch initial_condition_type
            case 'sphere'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 5; Ly = 5; Lz = 5;
                epsilon = 0.1;
                x_grid = linspace(-Lx/2, Lx/2, Nx);
                y_grid = linspace(-Ly/2, Ly/2, Ny);
                z_grid = linspace(-Lz/2, Lz/2, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                radius = 0.5;
                interface_width = sqrt(2) * epsilon;
                u = tanh((radius - sqrt(xx.^2 + yy.^2 + zz.^2)) / interface_width);

            case 'dumbbell'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 2.0; Ly = 1.0; Lz = 1.0;
                epsilon = 0.05;
                x_grid = linspace(0, Lx, Nx);
                y_grid = linspace(0, Ly, Ny);
                z_grid = linspace(0, Lz, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                R0 = 0.25;
                interface_width = sqrt(2) * epsilon;
                r1 = sqrt((xx - 0.3).^2 + (yy - 0.5).^2 + (zz - 0.5).^2);
                r2 = sqrt((xx - 1.7).^2 + (yy - 0.5).^2 + (zz - 0.5).^2);
                u_spheres = tanh((R0 - r1) / interface_width) + tanh((R0 - r2) / interface_width) + 1;
                bar_mask = (xx > 0.4 & xx < 1.6 & yy > 0.4 & yy < 0.6 & zz > 0.4 & zz < 0.6);
                u = u_spheres;
                u(bar_mask) = 1.0;
                u = max(-1.0, min(1.0, u));

            case 'star'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 5.0; Ly = 5.0; Lz = 5.0;
                epsilon = 0.05;
                x_grid = linspace(-Lx/2, Lx/2, Nx);
                y_grid = linspace(-Ly/2, Ly/2, Ny);
                z_grid = linspace(-Lz/2, Lz/2, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                interface_width = sqrt(2.0) * epsilon;
                theta = atan2(zz, xx);
                R_theta = 0.7 + 0.2 * cos(6 * theta);
                dist = sqrt(xx.^2 + 2*yy.^2 + zz.^2);
                u = tanh((R_theta - dist) / interface_width);

            case 'heart'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 3.0; Ly = 3.0; Lz = 3.0;
                epsilon = 0.05;
                x_grid = linspace(-Lx/2, Lx/2, Nx);
                y_grid = linspace(-Ly/2, Ly/2, Ny);
                z_grid = linspace(-Lz/2, Lz/2, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                interface_term = sqrt(2*epsilon);
                numerator = (xx.^2 + (9/4)*yy.^2 + zz.^2 - 1).^3 - xx.^2.*zz.^3 - (9/80)*yy.^2.*zz.^3;
                u = tanh(numerator ./ interface_term);
        end
        dt = 0.0005;

    case 'CH3D'
        % CH3D parameters (multiple ICs available)
        valid_ICs = {'sphere', 'dumbbell', 'star', 'heart'};
        if ~ismember(initial_condition_type, valid_ICs)
            error('CH3D: Invalid initial condition. Choose from: %s', strjoin(valid_ICs, ', '));
        end

        % Set file path based on initial condition and MHNO/PI-MHNO
        %base_path = '/scratch/noqu8762/phase_field_equations_4d/CH3D_python_predictions_';
        base_path = '/scratch/noqu8762/phase_field_equations_4d/OOD_MatFiles/Old_version_MatFiles/CH3D_python_predictions_'
        if use_PIMHNO
            python_data_file = [base_path initial_condition_type '_PIMHNO.mat'];
        else
            python_data_file = [base_path initial_condition_type '.mat'];
        end

        % Set parameters based on initial condition
        switch initial_condition_type
            case 'sphere'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 2; Ly = 2; Lz = 2;
                epsilon = 0.05;
                x_grid = linspace(-Lx/2, Lx/2, Nx);
                y_grid = linspace(-Ly/2, Ly/2, Ny);
                z_grid = linspace(-Lz/2, Lz/2, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                radius = 0.5;
                interface_width = sqrt(2) * epsilon;
                u = tanh((radius - sqrt(xx.^2 + yy.^2 + zz.^2)) / interface_width);

            case 'dumbbell'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 2.5; Ly = 1.0; Lz = 1.0;
                epsilon = 0.05;
                x_grid = linspace(0, Lx, Nx);
                y_grid = linspace(0, Ly, Ny);
                z_grid = linspace(0, Lz, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                R0 = 0.25;
                interface_width = sqrt(2) * epsilon;
                r1 = sqrt((xx - 0.3).^2 + (yy - 0.5).^2 + (zz - 0.5).^2);
                r2 = sqrt((xx - 1.7).^2 + (yy - 0.5).^2 + (zz - 0.5).^2);
                u_spheres = tanh((R0 - r1) / interface_width) + tanh((R0 - r2) / interface_width) + 1;
                bar_mask = (xx > 0.4 & xx < 1.6 & yy > 0.4 & yy < 0.6 & zz > 0.4 & zz < 0.6);
                u = u_spheres;
                u(bar_mask) = 1.0;
                u = max(-1.0, min(1.0, u));

            case 'star'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 2.0; Ly = Lx; Lz = Lx;
                epsilon = 0.05;
                x_grid = linspace(-Lx/2, Lx/2, Nx);
                y_grid = linspace(-Ly/2, Ly/2, Ny);
                z_grid = linspace(-Lz/2, Lz/2, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                interface_width = sqrt(2.0) * epsilon;
                theta = atan2(zz, xx);
                R_theta = 0.7 + 0.2 * cos(6 * theta);
                dist = sqrt(xx.^2 + 2*yy.^2 + zz.^2);
                u = tanh((R_theta - dist) / interface_width);

            case 'heart'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 5.0; Ly = Lx; Lz = Lx;
                epsilon = 0.15;
                x_grid = linspace(-Lx/2, Lx/2, Nx);
                y_grid = linspace(-Ly/2, Ly/2, Ny);
                z_grid = linspace(-Lz/2, Lz/2, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                interface_term = sqrt(2*epsilon);
                numerator = (xx.^2 + (9/4)*yy.^2 + zz.^2 - 1).^3 - xx.^2.*zz.^3 - (9/80)*yy.^2.*zz.^3;
                u = tanh(numerator ./ interface_term);
        end
        dt = 0.0005;

    case 'MBE3D'
        % MBE3D parameters (multiple ICs available)
        valid_ICs = {'sphere', 'dumbbell', 'star', 'torus'};
        if ~ismember(initial_condition_type, valid_ICs)
            error('MBE3D: Invalid initial condition. Choose from: %s', strjoin(valid_ICs, ', '));
        end

        % Set file path based on initial condition and MHNO/PI-MHNO
        %base_path = '/scratch/noqu8762/phase_field_equations_4d/MBE3D_python_predictions_';
        base_path = '/scratch/noqu8762/phase_field_equations_4d/OOD_MatFiles/Old_version_MatFiles/MBE3D_python_predictions_';
        if use_PIMHNO
            python_data_file = [base_path initial_condition_type '_PIMHNO.mat'];
        else
            python_data_file = [base_path initial_condition_type '.mat'];
        end

        % Set parameters based on initial condition
        switch initial_condition_type
            case 'sphere'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 2*pi; Ly = Lx; Lz = Lx;
                epsilon = 0.5;
                dt = 0.0001;
                x_grid = linspace(-Lx/2, Lx/2, Nx);
                y_grid = linspace(-Ly/2, Ly/2, Ny);
                z_grid = linspace(-Lz/2, Lz/2, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                radius = 1.5;
                interface_width = sqrt(2) * epsilon;
                u = tanh((radius - sqrt(xx.^2 + yy.^2 + zz.^2)) / interface_width);

            case 'torus'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 2*pi; Ly = Lx; Lz = Lx;
                epsilon = 0.1;
                dt = 0.001;
                x_grid = linspace(-Lx/2, Lx/2, Nx);
                y_grid = linspace(-Ly/2, Ly/2, Ny);
                z_grid = linspace(-Lz/2, Lz/2, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                R = 2.1; % Major radius
                r = 0.7; % Minor radius
                interface_width = sqrt(2) * epsilon;
                torus_dist = sqrt((sqrt(xx.^2 + yy.^2) - R).^2 + zz.^2);
                u = tanh((r - torus_dist) / interface_width);

            case 'dumbbell'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 40; Ly = 20; Lz = 20;
                epsilon = 0.05;
                dt = 0.01;
                x_grid = linspace(0, Lx, Nx);
                y_grid = linspace(0, Ly, Ny);
                z_grid = linspace(0, Lz, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                R0 = 0.25;
                interface_width = sqrt(2) * epsilon;
                r1 = sqrt((xx - 0.3).^2 + (yy - 0.5).^2 + (zz - 0.5).^2);
                r2 = sqrt((xx - 1.7).^2 + (yy - 0.5).^2 + (zz - 0.5).^2);
                u_spheres = tanh((R0 - r1) / interface_width) + tanh((R0 - r2) / interface_width) + 1;
                bar_mask = (xx > 0.4 & xx < 1.6 & yy > 0.4 & yy < 0.6 & zz > 0.4 & zz < 0.6);
                u = u_spheres;
                u(bar_mask) = 1.0;
                u = max(-1.0, min(1.0, u));

            case 'star'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 10*pi; Ly = Lx; Lz = Lx;
                epsilon = 0.5;
                dt = 0.005;
                x_grid = linspace(-Lx/2, Lx/2, Nx);
                y_grid = linspace(-Ly/2, Ly/2, Ny);
                z_grid = linspace(-Lz/2, Lz/2, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                interface_width = sqrt(2.0) * epsilon;
                theta = atan2(zz, xx);
                R_theta = 5.0 + 1.0 * cos(6 * theta);
                dist = sqrt(xx.^2 + 2*yy.^2 + zz.^2);
                u = tanh((R_theta - dist) / interface_width);
        end

    case 'PFC3D'
        % PFC3D parameters (multiple ICs available)
        valid_ICs = {'sphere', 'dumbbell', 'star', 'torus', 'separation'};
        if ~ismember(initial_condition_type, valid_ICs)
            error('PFC3D: Invalid initial condition. Choose from: %s', strjoin(valid_ICs, ', '));
        end

        % Set file path based on initial condition and MHNO/PI-MHNO
        %base_path = '/scratch/noqu8762/phase_field_equations_4d/PFC3D_python_predictions_';
        base_path = '/scratch/noqu8762/phase_field_equations_4d/OOD_MatFiles/Old_version_MatFiles/PFC3D_python_predictions_';
        if use_PIMHNO
            python_data_file = [base_path initial_condition_type '_PIMHNO.mat'];
        else
            python_data_file = [base_path initial_condition_type '.mat'];
        end

        % Set parameters based on initial condition
        switch initial_condition_type
            case 'sphere'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 10*pi; Ly = Lx; Lz = Lx;
                epsilon = 0.15;
                dt = 0.05;
                x_grid = linspace(-Lx/2, Lx/2, Nx);
                y_grid = linspace(-Ly/2, Ly/2, Ny);
                z_grid = linspace(-Lz/2, Lz/2, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                radius = 6.0;
                interface_width = sqrt(2) * epsilon;
                u = tanh((radius - sqrt(xx.^2 + yy.^2 + zz.^2)) / interface_width);

            case 'dumbbell'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 40; Ly = 20; Lz = 20;
                epsilon = 0.05;
                dt = 0.01;
                x_grid = linspace(0, Lx, Nx);
                y_grid = linspace(0, Ly, Ny);
                z_grid = linspace(0, Lz, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                R0 = 0.25;
                interface_width = sqrt(2) * epsilon;
                r1 = sqrt((xx - 0.3).^2 + (yy - 0.5).^2 + (zz - 0.5).^2);
                r2 = sqrt((xx - 1.7).^2 + (yy - 0.5).^2 + (zz - 0.5).^2);
                u_spheres = tanh((R0 - r1) / interface_width) + tanh((R0 - r2) / interface_width) + 1;
                bar_mask = (xx > 0.4 & xx < 1.6 & yy > 0.4 & yy < 0.6 & zz > 0.4 & zz < 0.6);
                u = u_spheres;
                u(bar_mask) = 1.0;
                u = max(-1.0, min(1.0, u));

            case 'star'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 10*pi; Ly = Lx; Lz = Lx;
                epsilon = 0.5;
                dt = 0.005;
                x_grid = linspace(-Lx/2, Lx/2, Nx);
                y_grid = linspace(-Ly/2, Ly/2, Ny);
                z_grid = linspace(-Lz/2, Lz/2, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                interface_width = sqrt(2.0) * epsilon;
                theta = atan2(zz, xx);
                R_theta = 5 + 1.0 * cos(6 * theta);
                dist = sqrt(xx.^2 + 2*yy.^2 + zz.^2);
                u = tanh((R_theta - dist) / interface_width);

            case 'torus'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 6*pi; Ly = Lx; Lz = Lx;
                epsilon = 0.5;
                dt = 0.05;
                x_grid = linspace(-Lx/2, Lx/2, Nx);
                y_grid = linspace(-Ly/2, Ly/2, Ny);
                z_grid = linspace(-Lz/2, Lz/2, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                R = 5.5; % Major radius
                r = 3.5; % Minor radius
                interface_width = sqrt(2) * epsilon;
                torus_dist = sqrt((sqrt(xx.^2 + yy.^2) - R).^2 + zz.^2);
                u = tanh((r - torus_dist) / interface_width);

            case 'separation'
                Nx = 32; Ny = 32; Nz = 32;
                Lx = 2*pi; Ly = Lx; Lz = Lx;
                epsilon = 0.5;
                dt = 0.0005;
                x_grid = linspace(-Lx/2, Lx/2, Nx);
                y_grid = linspace(-Ly/2, Ly/2, Ny);
                z_grid = linspace(-Lz/2, Lz/2, Nz);
                [xx, yy, zz] = ndgrid(x_grid, y_grid, z_grid);
                interface_width = sqrt(2) * epsilon;
                r1 = sqrt((xx + 1).^2 + yy.^2 + zz.^2);
                r2 = sqrt((xx - 1).^2 + yy.^2 + zz.^2);
                u = tanh((1 - r1) / interface_width) + tanh((1 - r2) / interface_width);
        end

    otherwise
        error('Unknown case study. Choose from: SH3D, AC3D, CH3D, MBE3D, PFC3D');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART 2: LOAD PYTHON MODEL RESULTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(['Loading Python predictions from: ', python_data_file]);
try
    python_data = load(python_data_file);
    python_pred = squeeze(python_data.python_pred);

    % Check dimensions match
    [py_Nx, py_Ny, py_Nz, ~] = size(python_pred);
    if py_Nx ~= Nx || py_Ny ~= Ny || py_Nz ~= Nz
        error('Dimension mismatch! Python data: [%d x %d x %d], MATLAB grid: [%d x %d x %d]', ...
              py_Nx, py_Ny, py_Nz, Nx, Ny, Nz);
    end

    num_selected = length(selected_frames);

    % Get inference time or use default
    if isfield(python_data, 'inference_time')
        python_inference_time = python_data.inference_time;
    else
        python_inference_time = 0.045; % Default value
    end

catch ME
    error('Could not load Python prediction file: %s. Error: %s', python_data_file, ME.message);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART 3: RUN MATLAB (DNS) EXACT SOLUTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(['Starting MATLAB (DNS) simulation for ', case_study, '...']);
tic;

% Initialize storage
all_iterations_dns = zeros(Nt + 1, Nx, Ny, Nz, 'single');
all_iterations_dns(1, :, :, :) = u;

% Run appropriate simulation based on case study
switch case_study
    case 'SH3D'
        % SH3D simulation
        kx = 2*pi/Lx * [0:Nx/2, -Nx/2+1:-1];
        ky = 2*pi/Ly * [0:Ny/2, -Ny/2+1:-1];
        kz = 2*pi/Lz * [0:Nz/2, -Nz/2+1:-1];
        [kxx, kyy, kzz] = ndgrid(kx.^2, ky.^2, kz.^2);

        for iter = 1:Nt
            u = real(u);
            s_hat = fftn(u/dt) - fftn(u.^3) + 2*(kxx + kyy + kzz).*fftn(u);
            v_hat = s_hat ./ (1.0/dt + (1-epsilon) + (kxx + kyy + kzz).^2);
            u = ifftn(v_hat);
            all_iterations_dns(iter + 1, :, :, :) = u;
        end

    case 'AC3D'
        % AC3D simulation
        epsilon1 = 0.1;
        Cahn = epsilon1^2;

        kx = 2*pi/Lx * [0:Nx/2, -Nx/2+1:-1];
        ky = 2*pi/Ly * [0:Ny/2, -Ny/2+1:-1];
        kz = 2*pi/Lz * [0:Nz/2, -Nz/2+1:-1];
        [kxx, kyy, kzz] = ndgrid(kx.^2, ky.^2, kz.^2);
        K2_laplace = kxx + kyy + kzz;

        for iter = 1:Nt
            u = real(u);
            nonlinear_term_hat = fftn(u.^3 - u);
            u_hat = fftn(u);
            v_hat = (u_hat - (dt/Cahn) * nonlinear_term_hat) ./ (1 + dt * K2_laplace);
            u = real(ifftn(v_hat));
            all_iterations_dns(iter + 1, :, :, :) = u;
        end

    case 'CH3D'
        % CH3D simulation
        Cahn = epsilon^2;

        kx = 2*pi/Lx * [0:Nx/2, -Nx/2+1:-1];
        ky = 2*pi/Ly * [0:Ny/2, -Ny/2+1:-1];
        kz = 2*pi/Lz * [0:Nz/2, -Nz/2+1:-1];
        [kxx, kyy, kzz] = ndgrid(kx.^2, ky.^2, kz.^2);
        K2_laplace = kxx + kyy + kzz;

        for iter = 1:Nt
            u = real(u);
            nonlinear_term_hat = fftn(u.^3 - 3*u);
            s_hat = fftn(u) - dt * K2_laplace .* nonlinear_term_hat;
            v_hat = s_hat ./ (1.0 + dt * (2.0 * K2_laplace + Cahn * K2_laplace.^2));
            u = real(ifftn(v_hat));
            all_iterations_dns(iter + 1, :, :, :) = u;
        end

    case 'MBE3D'
        % MBE3D simulation
        kx_fft = 1i * 2 * pi / Lx * [0:Nx/2 -Nx/2+1:-1];
        ky_fft = 1i * 2 * pi / Ly * [0:Ny/2 -Ny/2+1:-1];
        kz_fft = 1i * 2 * pi / Lz * [0:Nz/2 -Nz/2+1:-1];
        [kxx_fft, kyy_fft, kzz_fft] = ndgrid(kx_fft, ky_fft, kz_fft);

        k2x_fft = (2 * pi / Lx * [0:Nx/2 -Nx/2+1:-1]).^2;
        k2y_fft = (2 * pi / Ly * [0:Ny/2 -Ny/2+1:-1]).^2;
        k2z_fft = (2 * pi / Lz * [0:Nz/2 -Nz/2+1:-1]).^2;
        [kxx2_fft, kyy2_fft, kzz2_fft] = ndgrid(k2x_fft, k2y_fft, k2z_fft);

        Lap_f = (kxx2_fft + kyy2_fft + kzz2_fft);

        for iter = 1:Nt
            u = real(u);
            tu = fftn(u);

            % Calculate gradients
            fx = real(ifftn(kxx_fft .* tu));
            fy = real(ifftn(kyy_fft .* tu));
            fz = real(ifftn(kzz_fft .* tu));

            % Calculate the isotropic non-linear term
            grad_sq = (fx.^2 + fy.^2 + fz.^2);
            f1 = grad_sq .* fx;
            f2 = grad_sq .* fy;
            f3 = grad_sq .* fz;

            % Divergence of the non-linear term
            s_hat_nonlinear_part = (kxx_fft .* fftn(f1) + kyy_fft .* fftn(f2) + kzz_fft .* fftn(f3));

            % Full update equation
            s_hat = fftn(u / dt) + s_hat_nonlinear_part;
            v_hat = s_hat ./ (1 / dt - Lap_f + epsilon * Lap_f.^2);
            u = ifftn(v_hat);
            all_iterations_dns(iter + 1, :, :, :) = u;
        end

    case 'PFC3D'
        % PFC3D simulation
        p = 2*pi/Lx*[0:Nx/2 -Nx/2+1:-1];
        q = 2*pi/Ly*[0:Ny/2 -Ny/2+1:-1];
        r = 2*pi/Lz*[0:Nz/2 -Nz/2+1:-1];
        p2 = p.^2;
        q2 = q.^2;
        r2 = r.^2;
        [pp2, qq2, rr2] = ndgrid(p2, q2, r2);

        for iter = 1:Nt
            u = real(u);
            s_hat = fftn(u/dt) - (pp2 + qq2 + rr2) .* fftn(u.^3) + 2 * (pp2 + qq2 + rr2).^2 .* fftn(u);
            v_hat = s_hat ./ (1.0/dt + (1 - epsilon) * (pp2 + qq2 + rr2) + (pp2 + qq2 + rr2).^3);
            u = ifftn(v_hat);

            if ismember(iter, selected_frames)
                all_iterations_dns(iter + 1, :, :, :) = u;
            end

            if isnan(sum(u(:)))
                error('Simulation diverged (NaN values found). Check parameters.');
            end
        end
end

matlab_simulation_time = toc;
disp(['MATLAB (DNS) simulation complete. Elapsed time: ', num2str(matlab_simulation_time), ' s']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART 4: COMBINED VISUALIZATION (UNIFIED PLOTTING)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Generating comparison plots...');

% Create OOD directory if it doesn't exist
if ~exist('OOD', 'dir')
    mkdir('OOD');
end

% Set figure size based on number of selected frames
fig_width = 450 * num_selected;
fig_height = 1000;
fig = figure('Position', [100 100 fig_width fig_height], 'Color', 'w');

% Create dynamic title based on case study, method, and initial condition
method_str = 'PI-MHNO';
if ~use_PIMHNO
    method_str = 'MHNO';
end
sgtitle(sprintf('%s %s (%s): Pred. Time: %.2f s vs. Ex. Time: %.2f s', ...
                case_study, method_str, upper(initial_condition_type), ...
                python_inference_time, matlab_simulation_time), ...
                'FontSize', 24, 'FontWeight', 'bold', 'Interpreter', 'none');

% Layout parameters
left_margin = 0.08;
right_margin = 0.02;
top_margin = 0.15;
bottom_margin = 0.46;
horz_spacing = 0.0001;
vert_spacing = 0.12;

% Calculate plot dimensions
total_plot_width = 1 - left_margin - right_margin;
total_plot_height = 1 - top_margin - bottom_margin;
plot_width = (total_plot_width - (num_selected - 1) * horz_spacing) / num_selected;
plot_height = (total_plot_height - vert_spacing) / 2;

for i = 1:num_selected
    frame_to_plot = selected_frames(i);
    time_label = sprintf('t = %dΔt', frame_to_plot);

    % Calculate positions for this column
    current_left = left_margin + (i-1) * (plot_width + horz_spacing);
    bottom_pred_row = bottom_margin + plot_height + vert_spacing;
    bottom_exact_row = bottom_margin;

    % --- Plot Python results (TOP ROW) ---
    ax1 = axes('Position', [current_left, bottom_pred_row, plot_width, plot_height]);
    frame_idx_py = frame_to_plot + 1;
    u_python = python_pred(:,:,:,frame_idx_py);
    [f, v] = isosurface(xx, yy, zz, u_python, 0);
    if ~isempty(v)
        patch(ax1, 'Faces', f, 'Vertices', v, 'FaceColor', 'interp', 'EdgeColor', 'none', ...
              'FaceVertexCData', v(:,3), 'FaceAlpha', 0.8);
        lighting(ax1, 'gouraud');
        light(ax1, 'Position', [1 1 1], 'Style', 'infinite');
        material(ax1, 'dull');
        colormap(ax1, jet);
        view(45, 30);
        axis(ax1, 'tight', 'equal');
        title(ax1, sprintf('\n\n%s', time_label), 'FontSize', 22, 'FontWeight', 'bold');
    else
        title(ax1, sprintf('\n\n%s (Vanished)', time_label), 'FontSize', 22, 'FontWeight', 'bold');
        view(45, 30);
        axis(ax1, 'tight', 'equal');
    end
    set(ax1, 'FontSize', 18, 'FontWeight', 'bold');
    xlabel(ax1, 'X', 'FontSize', 18, 'FontWeight', 'bold');
    ylabel(ax1, 'Y', 'FontSize', 18, 'FontWeight', 'bold');
    if i == 1
        zlabel(ax1, 'Prediction', 'FontSize', 24, 'FontWeight', 'bold');
    else
        zlabel(ax1, 'Z', 'FontSize', 18, 'FontWeight', 'bold');
    end
    camzoom(ax1, 1.3);

    % --- Plot MATLAB results (BOTTOM ROW) ---
    ax2 = axes('Position', [current_left, bottom_exact_row, plot_width, plot_height]);
    frame_idx_dns = min(max(round(frame_to_plot / (Nt/(length(all_iterations_dns)-1)) + 1), 1), size(all_iterations_dns,1));
    u_matlab = squeeze(all_iterations_dns(frame_idx_dns,:,:,:));
    [f, v] = isosurface(xx, yy, zz, u_matlab, 0);
    if ~isempty(v)
        patch(ax2, 'Faces', f, 'Vertices', v, 'FaceColor', 'interp', 'EdgeColor', 'none', ...
              'FaceVertexCData', v(:,3), 'FaceAlpha', 0.8);
        lighting(ax2, 'gouraud');
        light(ax2, 'Position', [1 1 1], 'Style', 'infinite');
        material(ax2, 'dull');
        colormap(ax2, jet);
        view(45, 30);
        axis(ax2, 'tight', 'equal');
        title(ax2, sprintf('\n\n%s', time_label), 'FontSize', 22, 'FontWeight', 'bold');
    else
        title(ax2, sprintf('\n\n%s (Vanished)', time_label), 'FontSize', 22, 'FontWeight', 'bold');
        view(45, 30);
        axis(ax2, 'tight', 'equal');
    end
    set(ax2, 'FontSize', 18, 'FontWeight', 'bold');
    xlabel(ax2, 'X', 'FontSize', 18, 'FontWeight', 'bold');
    ylabel(ax2, 'Y', 'FontSize', 18, 'FontWeight', 'bold');
    if i == 1
        zlabel(ax2, 'Exact', 'FontSize', 24, 'FontWeight', 'bold');
    else
        zlabel(ax2, 'Z', 'FontSize', 18, 'FontWeight', 'bold');
    end
    camzoom(ax2, 1.3);
end

% === 1D PROFILE PLOT ===
ax_profile_pos = [0.1, 0.08, 0.85, 0.25];
ax_profile = axes('Position', ax_profile_pos);

hold(ax_profile, 'on');
grid(ax_profile, 'on');
box(ax_profile, 'on');

line_styles = {'-', '--'};
line_width = 3.0;
colors = get(ax_profile, 'ColorOrder');

% Get centerline indices
center_y_idx = round(Ny/2);
center_z_idx = round(Nz/2);

for i = 1:num_selected
    frame_to_plot = selected_frames(i);
    current_color = colors(mod(i-1, size(colors, 1)) + 1, :);

    % Plot DNS profile
    frame_idx_dns = min(max(round(frame_to_plot / (Nt/(length(all_iterations_dns)-1)) + 1), 1), size(all_iterations_dns,1));
    profile_dns = squeeze(all_iterations_dns(frame_idx_dns, :, center_y_idx, center_z_idx));
    plot(ax_profile, x_grid, profile_dns, 'LineStyle', line_styles{1}, 'Color', current_color, 'LineWidth', line_width);

    % Plot Python profile
    frame_idx_py = frame_to_plot + 1;
    profile_py = squeeze(python_pred(:, center_y_idx, center_z_idx, frame_idx_py));
    plot(ax_profile, x_grid, profile_py, 'LineStyle', line_styles{2}, 'Color', current_color, 'LineWidth', line_width);
end

% Create legend proxies
h_proxy = gobjects(2 + num_selected, 1);
h_proxy(1) = plot(NaN, NaN, 'LineStyle', line_styles{1}, 'Color', 'k', 'LineWidth', line_width, 'DisplayName', 'Exact');
h_proxy(2) = plot(NaN, NaN, 'LineStyle', line_styles{2}, 'Color', 'k', 'LineWidth', line_width, 'DisplayName', 'Prediction');
for i = 1:num_selected
    current_color = colors(mod(i-1, size(colors, 1)) + 1, :);
    h_proxy(2+i) = plot(NaN, NaN, 'LineStyle', '-', 'Color', current_color, 'LineWidth', line_width, 'DisplayName', sprintf('t = %d', selected_frames(i)));
end
hold(ax_profile, 'off');

% Format 1D plot
title(ax_profile, '1D Centerline Profile Comparison', 'FontSize', 22, 'FontWeight', 'bold');
xlabel(ax_profile, 'Position along x-axis', 'FontSize', 18, 'FontWeight', 'bold');
ylabel(ax_profile, 'Field Value u', 'FontSize', 18, 'FontWeight', 'bold');
%ylim(ax_profile, [-1.1, 1.5]);
% Adjust ylim based on case study
switch case_study
    case 'MBE3D'
        ylim(ax_profile, [-1.2, 1.2]);  % Custom ylim for MBE3D
    otherwise
        ylim(ax_profile, [-1.1, 1.5]);  % Default for other cases
end

xlim(ax_profile, [x_grid(1), x_grid(end)]);
set(ax_profile, 'FontSize', 18, 'FontWeight', 'bold', 'LineWidth', 1.5);
lgd = legend(h_proxy, 'Location', 'northeast', 'NumColumns', 1);
set(lgd, 'FontSize', 20, 'FontWeight', 'bold', 'Box', 'on');

disp('Comparison visualization complete.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART 5: SAVE THE FIGURE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Saving the figure...');

% Create filename based on case study, method, and initial condition
filename = sprintf('%s_%s_%s.png', case_study, method_str, initial_condition_type);
filename = fullfile('OOD', filename);

print(fig, filename, '-dpng', '-r300');
disp(['Figure saved as ', filename]);