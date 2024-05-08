clear; 
set(0, 'defaultaxesfontsize',20,'defaultaxeslinewidth',1.5,...
    'defaultlinelinewidth',2,'defaultpatchlinewidth',2,...
    'defaulttextfontsize',20,'defaulttextInterpreter','latex'); 
maxit = 100; 
tol = 1e-6;  
 %Example from paper "Operator splitting methods for pricing American options under stochastic volatility
K=10; T=0.25; kappa=5; eta=0.16; sigma=0.9; r=0.1; rho=0.1; Vmax=1;v_0=0.25;
Smax=20;  price_ref= 0.795968397469444; %computed by optByHestonFD

fprintf('--------------------------------------------------------------[Time-stepping]\t\t [All-at-once]\n');
fprintf('Nt&Nx\t S(K,V_0,0)\t Error\t CPU\t P-It (avg.) \t CPU\t P-It (avg.)\n')
UH=[];xH=[];tH=[];
for M =[10 20 40 80 160] % number of time step
tau = T/M; % increment of time step
t = (0:M)'*tau; % The axis of time 
Ns = 80; % number of step in Stock price 
Nv = Ns/2;     % the number of grid nodes in v-axis
hs = Smax/Ns;  % grid size in s-axis
hv = Vmax/Nv;  % grid size in v-axis
% grid nodes
s = (0:Ns)'*hs;
v = (0:Nv)'*hv;

%% the initial value condition
u = zeros(Ns + 1,Nv + 1,M + 1);
u(:,:,1) = repmat(max(0,K - s),1,Nv + 1); 
u(1,:,:) = K; % u(0,v,t) = K;
 
%% the matrices
% % some auxiliary matrices
% Diagonal matrices
Ds = spdiags(s,0,Ns + 1,Ns + 1);
Dv = spdiags(v,0,Nv + 1,Nv + 1); 
Is = speye(Ns); 
Iv = speye(Nv + 1);
Isv = speye(Ns*(Nv + 1));
% matrices from u(0,v,t) = K, u_v(s,Vmax,t) = 0 and u_s(Smax,v,t) = 0
u_s0 = sparse(u(:,:,2));
us_smax = sparse(Ns + 1,Nv + 1);
uv_vmax = us_smax;
% us_smax = [sparse(Ns,Nv + 1); ones(1,Nv + 1)];
% matrices from s-direction
e1 = ones(Ns - 1,1);
As = 1/hs/2*spdiags([[-e1;0;0],[0;0;e1]],[-1 1],Ns + 1,Ns + 1);
Ass = 1/hs^2*spdiags([[e1;2;0],[0;-2*e1;-2],[0;0;e1]],[-1 0 1],Ns + 1,Ns + 1);
% matrices from v-direction
e1 = ones(Nv - 1,1);
Av = 1/hv/2*spdiags([[-e1;0;0],[0;0;e1]],[-1 1],Nv + 1,Nv + 1);
Avv = 1/hv^2*spdiags([[e1;2;0],[0;-2*e1;-2],[0;0;e1]],[-1 0 1],Nv + 1,Nv + 1);
% % note: the matrix from the term "-r*u", no boundary vector
% the matrix A0 from the term "rho*sigma*s*v*u_sv", boundary vector: Rsv
A0 = rho*sigma*kron(Dv*Av,Ds(2:Ns + 1,2:Ns + 1)*As(2:Ns + 1,2:Ns + 1));
Rsv = rho*sigma*(Ds*As*u_s0)*(Av.'*Dv); Rsv = Rsv(2:Ns + 1,:); Rsv = Rsv(:);
% the matrix A1 from the term "r*s*u_s + 1/2*s^2*v*u_ss - 1/2*r*u", 
%    boundary vector: Rs, Rss
A1 = r*kron(Iv,Ds(2:Ns + 1,2:Ns + 1)*As(2:Ns + 1,2:Ns + 1)) + ...
       1/2*kron(Dv,Ds(2:Ns + 1,2:Ns + 1).^2*Ass(2:Ns + 1,2:Ns + 1)) - 1/2*r*Isv;
Rs = r*Ds*(As*u_s0 + us_smax); Rs = Rs(2:Ns + 1,:); Rs = Rs(:);
Rss = 1/2*Ds.^2*(Ass*u_s0 + (2/hs)*us_smax)*Dv;
Rss = Rss(2:Ns + 1,:); Rss = Rss(:);
% % 
% the matrix A2 from the term "kappa*(eta - v)*u_v + 1/2*sigma^2*v*u_vv - 1/2*r*u", 
%    boundary vector: Rv,Rvv
% %  the matrix tilde_Av from u_v: forward formula, if v = 0; 
% %   central formula, if 0 < v < 1; upwind formula, if v >= 1; 
% backward formula
tilde_Av = 1/hv/2*spdiags([[e1;0;0],[0;-4*e1;0],[0;0;3*e1]],[-2 -1 0],Nv + 1,Nv + 1); 
% if v < 1, apply the central formula
% find the maximum index of v < 1
central_index = find(v < 1,1,'last');
% replace the corresponding positions by the central formula
tilde_Av(1:central_index,:) = Av(1:central_index,:);
% forward formula, if v = 0
tilde_Av(1,:) = 0;
tilde_Av(1,1:3) = [-3,4,-1]/hv/2;

A2 = kappa*kron((eta*Iv - Dv)*tilde_Av,Is) + ...
     1/2*sigma^2*kron(Dv*Avv,Is) - 1/2*r*Isv;
Rvv = 1/2*sigma^2*((2/hv)*uv_vmax)*Dv; Rvv = Rvv(2:Ns + 1,:); Rvv = Rvv(:);
Rv = kappa*(uv_vmax)*(eta*speye(Nv + 1) - Dv);
Rv = Rv(2:Ns + 1,:); Rv = Rv(:);
% % matrix from space discrezation
A = A0 + A1 + A2;  % space-discretized matrix A,
% % boundary vector 
g = Rsv + Rss + Rs + Rvv + Rv; % the vector g, see the note [xx]

u0 = u(2:Ns + 1,:,1); % unknown points
u0 = u0(:);   % initial value vector \phi, see note [xx]
 
%% All-at-once min(M*x - c, x - g) = 0, find x in R^{N}  %%
e1 = ones(M,1);
%% backward Euler
B = (1/tau)*spdiags([-e1 e1],[-1 0],M,M);  
%% Leap-frog 
%B = (1/(2*tau))*spdiags([-e1 e1],[-1 1],M,M);
%B(M,M-1:M) = [-1 1]/tau;
 
It = speye(M); Is = speye(size(A,1));
Ac = kron(B,Is) - kron(It,A);  % \mathcal{A}
G = kron(e1,g);
% Backward Euler 
G(1:length(u0)) = G(1:length(u0)) + u0/tau;
% Leap-frog 
%G(1:length(u0)) = G(1:length(u0)) + u0/(2*tau);
% initial guess
U0 = kron(e1,u0); 

alf=1e-8; Da=alf.^((0:M-1)'/M);
c1=zeros(M,1); c1(1:2)=[1 -1]/tau; D1=fft(Da.*c1);
Ix=speye(size(A,1)); itvec=[];
tic 
%% time-stepping for solve LCP at each time step, for comparison
Aj=Ix/tau-A; X0=[]; x0 = u0; 
for j=1:M     
    X0=[X0;x0]; %save as initial guess for PinT solver
    [xT,jIt]=LCP_policy(Aj,(x0/tau)+g,u0,x0,tol,maxit);%smaller LCP
    %[xT,jIt]=LCP_newton(Aj,(x0/tau)+g,u0,x0,tol,maxit);
    itvec=[itvec;jIt];
    x0=xT;        
end
timecpu1=toc; 
pIt=sum(itvec);

%x0=X0; %initial guess from time-stepping solution
%x0=U0; %initial guess from pay-off function
x0=max(Ac\G,U0);%European solution as initial guess
itvec2=0;
%%  choose different solvers 
tic
%[xp2,pIt2,err2,phi2,itvec2]=LCP_policy_PinT(Ac,G,U0,x0,tol,maxit,M);%no diag
%[xp2,pIt2,err2,phi2,itvec2]=LCP_policy_PinT(Ac,G,U0,x0,tol,maxit,M,Da,D1,A,Ix);
%[xp2,pIt2,err2,phi2]=LCP_policy(Ac,G,U0,x0,tol,maxit);%no PinT 
% [xp2,pIt2,err2,phi2]=LCP_policy_block(Ac,G,U0,x0,tol,maxit);%no PinT
[xp2,pIt2,err2,phi2,itvec2]=LCP_policy_block_PinT(Ac,G,U0,x0,tol,maxit,M,Da,D1,A,Ix);
 timecpu2=toc; 
 
%use 2D interpolation to evaluate check points (K,v_0) 
uT=reshape(xp2(end-length(xT)+1:end),Ns,Nv+1);%PinT solution
price_lp =interp2(v(1:end)',s(2:end),full(uT),v_0,K,'spline'); 
 
 err_diff= norm(xT-uT(:),inf);%compare two methods 
err_price =norm(price_ref - price_lp,inf);
fprintf('%d&%d&\t %1.6f&\t %1.2e&\t %1.2f&\t %d &\t %1.2f&\t %d (%d)&\t %1.2e \n',...
    M,Ns,price_lp,err_price,timecpu1,pIt,timecpu2, pIt2,sum(itvec2),err_diff) 
end
fprintf('S(K,0) reference= %1.6f\n',price_ref);