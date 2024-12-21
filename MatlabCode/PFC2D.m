clc;
clear;
close all

%% Parameter Initialization

% Spatial Parameters
Nx=64; 
Ny=Nx; 
Lx=64; 
Ly=Lx; 
hx=Lx/Nx; 
hy=Ly/Ny;
x=linspace(-0.5*Lx+hx,0.5*Lx,Nx);
y=linspace(-0.5*Ly+hy,0.5*Ly,Ny);

% constant
epsilon=0.025;

% Discrete Fourier Transform
p=2*pi/Lx*[0:Nx/2 -Nx/2+1:-1];
q=2*pi/Ly*[0:Ny/2 -Ny/2+1:-1];
p2=p.^2; 
q2=q.^2; 
[pp2,qq2]=ndgrid(p2,q2);

% Time Discritization
dt=0.1; 
Nt = 10000;
T=Nt*dt; 
% Nt=round(T/dt); 
Np=100;
ns=Nt/Np;
%% Initial Condition
u_mean = 0.07;
% u=u_mean + u_mean*(2*rand(Nx,Ny)-1);

tau = 100;
alpha = 1.5;
u = u_mean + u_mean*GRF(alpha, tau, Nx);

%% Initial Preview
figure(1); 
clf;
set(gcf, 'Position', [100, 100, 900, 900]);
surf(x,y,real(u'));
colormap jet
shading interp; 
view(0,90); 
axis image;
clim([u_mean-0.2 u_mean+0.2]);
colorbar;
pause(1)
disp(['Maximum = ' num2str(max(u, [],'all'))])
disp(['Minimum = ' num2str(min(u, [],'all'))])

%% Update
for iter=1:Nt
    % disp("Iteration Number = " + num2str(iter))
    u=real(u);
    s_hat=fft2(u/dt)-(pp2+qq2).*fft2(u.^3)+2*(pp2+qq2).^2.*fft2(u);
    v_hat=s_hat./(1.0/dt+(1-epsilon)*(pp2+qq2)+(pp2+qq2).^3);
    u=ifft2(v_hat);
    if (mod(iter,ns)==0)
        disp(['Maximum = ' num2str(max(u, [],'all'))])
        disp(['Minimum = ' num2str(min(u, [],'all'))])

        surf(x,y,real(u')); 
        colormap jet
        shading interp; 
        view(0,90);
        axis image;
        clim([u_mean-0.2 u_mean+0.2]);
        colorbar;
        pause(0.01)
    end
end