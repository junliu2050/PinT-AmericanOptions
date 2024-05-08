clear;
set(0, 'defaultaxesfontsize',20,'defaultaxeslinewidth',1.5,...
    'defaultlinelinewidth',2,'defaultpatchlinewidth',2,...
    'defaulttextfontsize',20,'defaulttextInterpreter','latex');
global mu1 mu2 fineL LEVEL0
maxit = 50;
tol = 1e-6;
mu1=1; mu2=1;  fineL=6; LEVEL0=3;%multigrid smoothing iterations
%% Model parameters
% http://www.math.ualberta.ca/ijnam/Volume-15-2018/No-3-18/2018-03-03.pdf
%% American Put Spread option: AP_P3 case in Table 1
T=122/365; sigma1=0.35; sigma2=0.38; r=0.035; rho=0.6;
K=25; Spot=[127.68,99.43]; price_ref=6.932875; %MSC_nu in Table 5
S1max=300;S2max=300;
%% paper:  SPREAD OPTION PRICING USING ADI METHODS
cpflag=-1; %call option=1, put option=-1;
payoff=@(s1,s2) max(cpflag*((s1-s2)-K),0); %Spread option
%The discounted pay-off boundary conditions for Spread option
%http://www.math.ualberta.ca/ijnam/Volume-15-2018/No-3-18/2018-03-03.pdf
BC_left=@(s2,t) max(max(cpflag*(-K*exp(-r*t)-s2),0),max(cpflag*(-K-s2),0));%s1=0
BC_bot=@(s1,t) max(max(cpflag*(s1-K*exp(-r*t)),0),max(cpflag*(s1-K),0));%s2=0
BC_right=@(s2,t) max(max(cpflag*(S1max-K*exp(-r*t)-s2),0),max(cpflag*(S1max-K-s2),0));%s1=S1max
BC_top=@(s1,t) max(max(cpflag*(s1-K*exp(-r*t)-S2max),0),max(cpflag*(s1-K-S2max)));%s2=S2max
%set up functions for scheme use
syms x y h
d0=-sym(r); dx=r*x;dxx=0.5*sigma1^2*x^2;dy=r*y;dyy=0.5*sigma2^2*y^2;
dxy=rho*sigma1*sigma2*x*y; %PDE coefficients
%Set up scheme stencil, using symbolic coefficients and step size h
%based on central difference in both x and y direction
% a0=-2*dxx-2*dyy+d0*h^2;        a1=dxx+h*dx/2;        a2=dyy+h*dy/2;
% a3=dxx-h*dx/2;        a4=dyy-h*dy/2;        a5=dxy/4;
% a6=-dxy/4;        a7=dxy/4;        a8=-dxy/4;
%upwind 7 point scheme for mixed term to get M matrix
a0=-2*dxx-2*dyy+d0*h^2+dxy+h*dx+h*dy;        a1=dxx-dxy/2;        a2=dyy-dxy/2;
a3=dxx-h*dx-dxy/2;        a4=dyy-h*dy-dxy/2;        a5=dxy/2;
a6=0;        a7=dxy/2;        a8=0;

gfun=sym(0);    rhs=gfun; %not used in this example
fsch=matlabFunction(a0,a1,a2,a3,a4,a5,a6,a7,a8,rhs);

fprintf('---------------------------------[Time-stepping]----------\t [All-at-once]\n');
fprintf('Nt&Nx\t S(K,V_0,0)\t Error\t CPU\t P-It  \t CPU\t P-It (G-It)\n')

for M =10*2.^(0:6)
    N=2^fineL; tau = T/M; % increment of time step
    t = (0:M)'*tau; % The axis of time
    iluflag=false;
    [A,A_bc,xgrid,ygrid,Xint,Yint]=BuildMatrix(N-1,S1max,sigma1,sigma2,rho,r);
    %initial condition at t=0
    u0=payoff(Xint,Yint); u0=u0(:);
    tic
    %Backward Euler time-stepping for solve LCP at each time step
    Ix=speye(size(A,1)); itvec=[];
    Aj=Ix/tau-A; X0=[]; x0 = u0; G=zeros(size(A,1),M);
    Xstep=zeros(N-1,N-1,M);
    for j=1:M
        %construct boundary condition vector at time tj
        tj=j*tau;
        fleft=BC_left(ygrid,tj);       fright=BC_right(ygrid,tj);
        ftop=BC_top(xgrid(2:end-1)',tj); fbot=BC_bot(xgrid(2:end-1)',tj);
        fcenter=[fbot;ftop];   fUb=[fleft;fcenter(:);fright];
        G(:,j)=A_bc*fUb; %time-dependent boundary vector
        %solve small LCP at each time step
        [xT,jIt]=LCP_policy(Aj,(x0/tau)+G(:,j),u0,x0,tol,maxit);%smaller LCP
        itvec=[itvec;jIt];
        Xstep(:,:,j)=reshape(xT,N-1,N-1);
        x0=xT;
    end
    timecpu1=toc;
    pIt=sum(itvec);

    %% All-at-once min(M*x - c, x - g) = 0, find x in R^{N}  %%
    e1 = ones(M,1);
    %% backward Euler
    B = (1/tau)*spdiags([-e1 e1],[-1 0],M,M);
    %% Leap-frog
    %B = (1/(2*tau))*spdiags([-e1 e1],[-1 1],M,M);
    %B(M,M-1:M) = [-1 1]/tau;
    It = speye(M);
    Ac = kron(B,Ix) - kron(It,A);  %may cost a lot of memory

    % Backward Euler
    G(:,1) = G(:,1) + u0/tau;
    % Leap-frog
    %G(:,1) = G(:,1) + u0/(2*tau);
    G=G(:);

    alf=1e-8; Da=alf.^((0:M-1)'/M);
    c1=zeros(M,1); c1(1:2)=[1 -1]/tau; D1=fft(Da.*c1); %shifts

    tic
    %% setup multigrid operators without shifts for all levels
    mg=[];
    %define the matrix at each level
    for Level=fineL:-1:LEVEL0
        nn=2^Level; h=S1max/(nn); m=nn-1;
        if(Level==fineL)
            mg(Level).Ax=A; %already  constructed above
        else
            mg(Level).Ax=BuildMatrix(m,S1max,sigma1,sigma2,rho,r);
        end
        e=ones(m,1);
        %interpolation operator in matrix form
        Pn=(1/2)*spdiags([e 2*e e],-2:0,m,m);
        mg(Level).P=kron(Pn(:,1:2:end-2),Pn(:,1:2:end-2));
        mg(Level).Ix=speye(m^2);
        %perform ilu for each shift only once and then save them
        for j=1:M
            [mg(Level).AL{j},mg(Level).AU{j}]=ilu(D1(j)*mg(Level).Ix-mg(Level).Ax);
        end
    end


    % pay-off over all time steps
    U0 = kron(e1,u0);  
    x0=max(Ac\G,U0); %European solution as initial guess works better 
    %%  choose different solvers
    %[xp2,pIt2,err2,phi2,itvec2]=LCP_policy_PinT(Ac,G,U0,x0,tol,maxit,M);%no diag
    %[xp2,pIt2,err2,phi2,itvec2]=LCP_policy_PinT(Ac,G,U0,x0,tol,maxit,M,Da,D1,A,Ix);

    %[xp2,pIt2,err2,phi2]=LCP_policy(Ac,G,U0,x0,tol,maxit);itvec2=0;%no PinT
    %[xp2,pIt2,err2,phi2]=LCP_policy_block(Ac,G,U0,x0,tol,maxit);itvec2=0;%no PinT
    [xp2,pIt2,err2,phi2,itvec2]=LCP_policy_block_PinT_GMG(Ac,G,U0,x0,tol,maxit,M,Da,D1,A,Ix,mg);
    timecpu2=toc;
    %use 2D interpolation to evaluate check points
    uT=reshape(xp2(end-length(xT)+1:end),N-1,N-1);%PinT solution
    price_lp =interp2(Xint,Yint,uT,Spot(1),Spot(2),'spline');

    err_diff= norm(xT-uT(:),inf);%compare two methods
    err_price =norm(price_ref - price_lp,inf);
    fprintf('%d&%d&\t %1.6f&\t %1.2e&\t %1.2f&\t %d &\t %1.2f&\t %d (%d) &\t%1.2e\n',...
        M,N,price_lp,err_price,timecpu1,pIt,timecpu2, pIt2,sum(itvec2),err_diff)
end
fprintf('S(K,0) reference= %1.6f\n',price_ref);


function [A,A_bc,xgrid,ygrid,Xint,Yint]=BuildMatrix(N,S1max,sigma1,sigma2,rho,r)
syms x y h
d0=-sym(r); dx=r*x;dxx=0.5*sigma1^2*x^2;dy=r*y;dyy=0.5*sigma2^2*y^2;
dxy=rho*sigma1*sigma2*x*y; %PDE coefficients
%Set up scheme stencil, using symbolic coefficients and step size h
%based on central difference in both x and y direction
% a0=-2*dxx-2*dyy+d0*h^2;        a1=dxx+h*dx/2;        a2=dyy+h*dy/2;
% a3=dxx-h*dx/2;        a4=dyy-h*dy/2;        a5=dxy/4;
% a6=-dxy/4;        a7=dxy/4;        a8=-dxy/4;
%upwind 7 point scheme for mixed term to get M matrix
a0=-2*dxx-2*dyy+d0*h^2+dxy;        a1=dxx+h*dx/2-dxy/2;        a2=dyy+h*dy/2-dxy/2;
a3=dxx-h*dx/2-dxy/2;        a4=dyy-h*dy/2-dxy/2;        a5=dxy/2;
a6=0;        a7=dxy/2;        a8=0;

gfun=sym(0);    rhs=gfun; %not used in this example
fsch=matlabFunction(a0,a1,a2,a3,a4,a5,a6,a7,a8,rhs);
hx=S1max/(N+1); %Assume S2max=S1max
xgrid=(0:N+1)'*hx;  ygrid=(0:N+1)'*hx;%uniform mesh

%interior grids
[Xint,Yint] = meshgrid(xgrid(2:end-1),ygrid(2:end-1));
%construct spatial matrix A
[c0,c1,c2,c3,c4,c5,c6,c7,c8,frhs]=fsch(hx,Xint,Yint);
if(length(c6(:))==1) c6=c6*ones(length(c0(:)),1); end
if(length(c8(:))==1) c8=c8*ones(length(c0(:)),1); end
B=(1/hx^2)*[c7(:) c3(:) c6(:) c4(:) c0(:) c2(:) c8(:) c1(:) c5(:)];%coefficients
n=(N+2)*(N+2);
An=[]; %construct matrix block row by block row, kind of slow
for k=1:N
    An=[An;spdiags(B((k-1)*N+1:k*N,:),...
        [1 2 3 N+3 N+4 N+5 2*N+5 2*N+6 2*N+7]-1+(k-1)*(N+2),N,n)];
end
%extract boundary nodes column locations
AUb=blkdiag(speye(N+2,N+2), ...
    kron(speye(N,N),sparse([1 2],[1 N+2],[1 1],2,N+2)), ...
    speye(N+2,N+2));
[~,jcol] = find(AUb);
jcol1=setdiff((1:n),jcol);
A=An(:,jcol1); A_bc=An(:,jcol); %seperate boundary columns
end
