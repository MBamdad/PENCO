clc 
clear; 
 
%% Parameter Initialization 
 
% Spatial Parameters 
Nx=32;  
Ny=Nx;  
Nz=Nx;  
Lx=3;  
Ly=3;  
Lz=3;  
hx=Lx/Nx;  
hy=Ly/Ny;  
hz=Lz/Nz; 
 
x=linspace(-0.5*Lx+hx,0.5*Lx,Nx); 
y=linspace(-0.5*Ly+hy,0.5*Ly,Ny); 
z=linspace(-0.5*Lz+hz,0.5*Lz,Nz); 
 
[xx,yy,zz]=ndgrid(x,y,z);  
 
% Interfacial energy constant 
epsilon=0.1; 
Cahn=epsilon^2; 
 
% Discrete Fourier Transform 
kx=2*pi/Lx*[0:Nx/2 -Nx/2+1:-1]; 
ky=2*pi/Ly*[0:Ny/2 -Ny/2+1:-1]; 
kz=2*pi/Lz*[0:Nz/2 -Nz/2+1:-1]; 
k2x = kx.^2;  
k2y = ky.^2;  
k2z = kz.^2; 
[kxx,kyy,kzz]=ndgrid(k2x,k2y,k2z); 
 
% Time Discretization 
dt=0.001;  
T=100*0.001;  
Nt=round(T/dt);  
ns=1; 
 
% Dataset 
data_size = 1000; 
num_saved_steps = Nt + 1;  % Save all steps 
binary_filename = "AC3D_" + num2str(Nx) + "_" + num2str(data_size) + ".bin"; 
mat_filename = "AC3D_" + num2str(Nx) + "_" + num2str(data_size) + ".mat"; 
 
%% Prepare Binary File 
fileID = fopen(binary_filename, 'wb'); 
if fileID == -1 
    error("Cannot open binary file for writing."); 
end 
 
%% Initial Condition 
tau = 400; 
alpha = 115; 
 
for data_num = 1:data_size 
    disp("Data number = " + num2str(data_num)) 
     
    norm_a = GRF3D(alpha, tau, Nx); 
    norm_a = norm_a + 0.2 * std(norm_a(:));    
    u = ones(Nx, Nx, Nz); 
    u(norm_a < 0) = -1; 
 
    all_iterations = zeros(num_saved_steps, Nx, Ny, Nz, 'single'); 
     
    %% Update 
    for iter = 1:Nt 
        all_iterations(iter, :, :, :) = u; 
        u = real(u);  
 
        s_hat = fftn(Cahn*u - dt*(u.^3 - 3*u)); 
        v_hat = s_hat ./ (Cahn + dt*(2 + Cahn*(kxx + kyy + kzz))); 
        u = ifftn(v_hat); 
    end 
    all_iterations(end, :, :, :) = u;  % Save final step 
 
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
