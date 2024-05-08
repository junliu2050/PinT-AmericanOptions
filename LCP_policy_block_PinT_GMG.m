function [x,k,res,phi,itvec]=LCP_policy_block_PinT_GMG(A,b,c,x,tol,maxit,Nt,Da,D1,Ax,Ix,mg)
%PinT policy iteration after projection to reduced system
%Use GMG for approximately solve shifted linear systems in step-(b)
II=speye(size(A,1));   
err1=(A*x-b); err2=(x-c); 
res0=norm(min(err1,err2),inf);
xnew=x;itvec=[];  
for k=1:maxit     
    phi=(err1<=err2); %0-1 matrix 
    %projected Jacobian system
    idx0=find(~phi);idx1=find(phi);     
    xnew(idx0)=c(idx0); %first solve identity block then leftover block
    %x(idx1)=A(idx1,idx1)\(b(idx1)-A(idx1,idx0)*c(idx0));%direct solver   
    %define projection operators
    AkJ=A(idx1,idx1); bk=(b(idx1)-A(idx1,idx0)*c(idx0));
    Pmat=II(idx1,:); %project matrix    
    Ak_fun=@(r) Pmat*Ak_PinT(Pmat'*r,Nt,Da,D1,Ax,Ix,mg); 
     
    warning off
    iter=0;x0=x(idx1);%initial guess from previous solution      
    %[xnew(idx1),flag,~,iter] =gmres(AkJ,bk,[],1e-12,maxit,Ak_fun,[],x0); 
     Afun=@(z) AkJ*z;
     [xnew(idx1),flag,~,iter] = fgmres(Afun,bk,1e-12,maxit,Ak_fun,x0);  
    errx=norm((xnew-x)./max(1,abs(xnew)),inf); %relative change
    x=xnew;
    itvec=[itvec;iter(end)]; 
    err1=(A*x-b); err2=(x-c); 
    res=norm(min(err1,err2),inf); %minimum as residual
  % fprintf('policy-iter=%d: res/res0=%1.2e,g-iter=%d,g-flag=%d, |errx|=%1.2e\n',k,res/res0,iter(end),flag,errx);
     if(res/res0<tol||errx<tol)  
        break;
    end  
end
end

function z=Ak_PinT(Res,Nt,Da,D1,Ax,Ix,mg) 
global mu1 mu2 fineL 
R=reshape(Res,[],Nt);
S1=fft(Da.*(R.')).'; %step (a)  
%solve only half of systems
Nt2=ceil((Nt+1)/2);  
for j=1:Nt2
    %S1(:,j)=(D1(j)*Ix-Ax)\S1(:,j); %step (b), major cost  
    %use one GMG V-cycle to approximately solve each shifted system,   
    S1(:,j)=mg_iter_2d(mg,zeros(length(S1(:,j)),1),S1(:,j),fineL,mu1,mu2,'ILU',D1(j),j);
end
S1(:,Nt2+1:Nt)=conj(S1(:,floor((Nt+1)/2):-1:2));

X=real((Da.\ifft(S1.')).'); %step (c)  
z=X(:);
end

