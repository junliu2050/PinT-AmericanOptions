clear; 
set(0, 'defaultaxesfontsize',20,'defaultaxeslinewidth',1.5,...
    'defaultlinelinewidth',2,'defaultpatchlinewidth',2,...
    'defaulttextfontsize',20,'defaulttextInterpreter','latex');
maxit = 100; 
tol = 1e-6;  

fprintf('--------------------------------------------------------------[Time-stepping] \t [All-at-once PinT]\n');
fprintf('Nt&Nx \t S(K,0) \t Error \t CPU\t P-It \t CPU\t P-It (G-It)\n')
% Terminal time
T = 1;
% Maximum stock price
Smax = 300; 
Smin = 0;
 
%Example from paper "BENCHOP—The BENCHmarking project in Option Pricing"
exname='Ex1';K=100;r=0.03;sigma=0.15;q=0; price_ref=4.820608184813253; %case I 
%Benchmarking from the book: http://www.mi.uni-koeln.de/~seydel/numerik/compfin/BENCHMARK00
%exname='Ex0';K=100;r=0.05;sigma=0.5;q=0; price_ref=0.1744876443E+02; 
 
 
diff = sigma^2/2;% diffusion coeff
adv = r-q;% advection coeff
reac = r;% reaction coeff 
UH=[];xH=[];tH=[];
for Nx =10*2.^(1:7) % number of time step
        fprintf("\n")        
    for Nt =10*2.^(1:7)  
M=Nt; N=Nx;
tau = T/M; % increment of time step
t = (0:M)'*tau; % The axis of time
%h = sqrt(50*tau); % % increment of stock price: h^2/tau = 50
%N = round((Smax-Smin)/h); % number of step in Stock price
h = Smax/N; % reset h to make sure N*h = Smax
x = Smin + (0:N)'*h; % The axis of stock price

% Option Price
u = zeros(N + 1,1);
u(:,1) = max(K - x,0);  % put option
u0 = u(2:N,1);     % initial value u_0 !!!

% The matrices
Is = speye(N-1);
e1 = ones(N-1,1);
sN = x(2:N);
Dss = spdiags(sN.^2,0,N-1,N-1);
Ds = spdiags(sN,0,N-1,N-1);
%% u_ss
Ass = (1/h^2)*spdiags([e1 -2*e1 e1],[-1 0 1],N-1,N-1);
%% u_s
As = (1/2/h)*spdiags([-e1 e1],[-1 1],N-1,N-1);  % central difference
%As = (1/h)*spdiags([-e1 e1],[0 1],N-1,N-1); % upwind difference
%%% the matrix $A$ in the ODE system
A = diff*Dss*Ass + adv*Ds*As - reac*Is;
%A = -(1/h^2)*spdiags([-e1 2*e1 -e1],[-1 0 1],N-1,N-1);
%% source term vector g(t) = g (constant vector)
%% Put option
g = zeros(N-1,1);
g(1) = diff*sN(1).^2*((K-Smin)/h^2)- adv*sN(1)*((K-Smin)/2/h); % central difference
%g(1) = diff*sN(1).^2*K; % upwind difference

%% All-at-once min(M*x - c, x - g) = 0, find x in R^{N}  %%
e1 = ones(M,1);
%% backward Euler
B = (1/tau)*spdiags([-e1 e1],[-1 0],M,M);  
%% Leap-frog 
%B = (1/(2*tau))*spdiags([-e1 e1],[-1 1],M,M);
%B(M,M-1:M) = [-1 1]/tau;
 
It = speye(M);
Ac = kron(B,Is) - kron(It,A);  % \mathcal{A}
G = kron(e1,g);
% Backward Euler 
G = G + [u0/tau;zeros((M-1)*(N-1),1)];
% Leap-frog 
%G = G + [u0/(2*tau);zeros((M-1)*(N-1),1)];
% initial guess
U0 = kron(e1,u0); 


Ix=speye(size(A,1)); 
alf=1e-8; Da=alf.^((0:M-1)'/M);
c1=zeros(M,1); c1(1:2)=[1 -1]/tau; D1=fft(Da.*c1);

Calf=B; Calf(1,end)=-alf/tau;
Palf = kron(Calf,Is) - kron(It,A);  
 
x0=U0; %initial guess from pay-off function 
%% choose different solvers 
tic
%[xp2,pIt2,err2,phi2,itvec2]=LCP_policy_PinT(Ac,G,U0,x0,tol,maxit,M);%no diag
[xp2,pIt2,err2,phi2,itvec2,phi_x,AkJ]=LCP_policy_PinT(Ac,G,U0,x0,tol,maxit,M,Da,D1,A,Ix);
%[xp2,pIt2,err2,phi2]=LCP_policy(Ac,G,U0,x0,tol,maxit); itvec2=[];%no PinT 
%[xp2,pIt2,err2,phi2]=LCP_policy_block(Ac,G,U0,x0,tol,maxit);itvec2=[];%split 0-1 blocks
%[xp2,pIt2,err2,phi2,itvec2]=LCP_policy_block_PinT(Ac,G,U0,x0,tol,maxit,M,Da,D1,A,Ix);
timecpu2=toc; 
 
itvec=[];
tic 
%% Sequential time-stepping for solve LCP at each time step, as comparison
Aj=Ix/tau-A;   x0 = u0; %PhiAll=[];
for j=1:M      
    [xT,jIt,res,Phi_j]=LCP_policy(Aj,(x0/tau)+g,u0,u0,tol,maxit);%smaller LCP 
    itvec=[itvec;jIt];
    x0=xT;   
end 
timecpu1=toc; 
pIt=sum(itvec);%add all time steps
 
optionvec = [K;xT;0];
price_lp = interp1(x,optionvec,K,'cubic');

optionvec2 = [K;xp2((M-1)*(N-1)+1:end);0];
price_lp2 = interp1(x,optionvec2,K,'cubic');
err_price = abs(price_ref - price_lp2);
err_diff=norm(xT-xp2((M-1)*(N-1)+1:end),inf);%compare two methods 
fprintf('%d&%d&\t %1.6f&\t %1.2e&\t %1.2f&\t %d &\t %1.2f& \t %d (%d)&\t%1.2e\n',...
    M,N,price_lp,err_price,timecpu1,pIt,timecpu2, pIt2,sum(itvec2),err_diff)
xH=x; tH=t;
UH=[u [K*ones(1,M);reshape(xp2,[],M);zeros(1,M)]];%save for initial guess
UHeu=[u [K*ones(1,M);reshape(Ac\G,[],M);zeros(1,M)]];%European option
    end
end 
%end
 fprintf('S(K,0) reference= %1.6f\n',price_ref);
 
  
 