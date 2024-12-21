clc;
clear;
close all
fclose('all');

%% Parameter Initialization

% Spatial Parameters
Nx=32;
Ny=32;
Nz=32;
Lx=1.1;
Ly=1.1;
Lz=1.1;
hx=Lx/Nx;
hy=Ly/Ny;
hz=Lz/Nz;
x=linspace(-0.5*Lx+hx,0.5*Lx,Nx);
y=linspace(-0.5*Ly+hy,0.5*Ly,Ny);
z=linspace(-0.5*Lz+hz,0.5*Lz,Nz);
[xx,yy,zz]=ndgrid(x,y,z);

% Interfacial energy constant
epsilon=0.0125; 
Cahn=epsilon^2;

% Discrete Fourier Transform
kx=2*pi/Lx*[0:Nx/2 -Nx/2+1:-1];
ky=2*pi/Ly*[0:Ny/2 -Ny/2+1:-1];
kz=2*pi/Lz*[0:Nz/2 -Nz/2+1:-1];
k2x=kx.^2; 
k2y=ky.^2; 
k2z=kz.^2;
[kxx,kyy,kzz]=ndgrid(k2x,k2y,k2z);

% Time Discretization
dt=0.01; 
Nt=200; 
T=Nt*dt; 
num_saved_steps = 101;
ns=Nt/(num_saved_steps-1);

% Dataset
data_size = 2200;
binary_filename = "CH3D_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
                  "_Nx_" + num2str(Nx) + ".bin";
mat_filename = "CH3D_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
               "_Nx_" + num2str(Nx) + ".mat";

%% Prepare Binary File
fileID = fopen(binary_filename, 'wb');
if fileID == -1
    error("Cannot open binary file for writing.");
end

%% Initial Condition
tau = 400;
alpha = 115;
% figure(1);

for data_num = 1:data_size
    disp("data number = " + num2str(data_num))
    
    norm_a = GRF3D(alpha, tau, Nx);
    norm_a = norm_a - 1.0 * std(norm_a(:));
    %norm_a = norm_a + 0.2 * std(norm_a(:));   

    %u = norm_a;
    u = zeros(Nx,Nx,Nz);
    u(norm_a >= 0) = 1;
    u(norm_a < 0) = -1;



    all_iterations = zeros(num_saved_steps, Nx, Ny, Nz, 'single');
    
    %% Initial Preview
    
    % clf;
    % p1=patch(isosurface(xx,yy,zz,real(u),0.));
    % set(p1,'FaceColor','g','EdgeColor','none'); 
    % daspect([1 1 1])
    % camlight;
    % lighting phong; 
    % box on; 
    % axis image;
    % view(45,45);
    % pause(2)
    
    %% Update
    save_idx = 1;
    for iter=1:Nt
        % disp("Iteration = " + num2str(iter))
        if iter == 1 || mod(iter,ns) == 0 || iter == Nt
            all_iterations(save_idx, :, :, :) = u;
            % disp("Saved = " + num2str(iter) + "at" + num2str(save_idx))
            save_idx = save_idx + 1;
        end

        u=real(u);
        s_hat=fftn(u)-dt*(kxx+kyy+kzz).*fftn(u.^3-3*u);
        v_hat=s_hat./(1.0+dt*(2.0*(kxx+kyy+kzz)+Cahn*(kxx+kyy+kzz).^2));
        u=ifftn(v_hat);
        % if (mod(iter,ns)==0)
        %     figure(1);
        %     clf;
        %     p1=patch(isosurface(xx,yy,zz,real(u),0.));
        %     set(p1,'FaceColor','g','EdgeColor','none'); 
        %     daspect([1 1 1])
        %     camlight;
        %     lighting phong; 
        %     box on; 
        %     axis image;
        %     view(45,45);
        %     pause(0.01)
        % end
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
% phi_mat.phi = zeros([data_size, num_saved_steps, Nx, Ny, Nz], 'single');

split_size = ceil(data_size / 4);
num_dims = [data_size, num_saved_steps, Nx, Ny, Nz]; 

initial_chunk = zeros([split_size, num_saved_steps, Nx, Ny, Nz], 'single');
phi_mat.phi = initial_chunk;
clear initial_chunk;

for i = 2:4
    start_idx = (i - 1) * split_size + 1;
    end_idx = min(i * split_size, data_size);
    chunk_size = end_idx - start_idx + 1;
    chunk = zeros([chunk_size, num_saved_steps, Nx, Ny, Nz], 'single');
    phi_mat.phi(start_idx:end_idx, :, :, :, :) = chunk;
    clear chunk; 
end

for data_num = 1:data_size
    disp("Saving dataset " +  num2str(data_num));
    data_chunk = fread(fileID, num_saved_steps * Nx * Ny * Nz, 'single');
    phi_mat.phi(data_num, :, :, :, :) = reshape(data_chunk, [1, num_saved_steps, Nx, Ny, Nz]);
end

fclose(fileID);
