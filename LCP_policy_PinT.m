function [x,k,res,phi,itvec,phi_x,AkJ]=LCP_policy_PinT(A,b,c,x,tol,maxit,Nt,Da,D1,Ax,Ix)
%PinT policy iteration
II=speye(size(A,1));  itvec=[];   
err1=(A*x-b); err2=(x-c); 
res0=norm(min(err1,err2),inf); 
for k=1:maxit        
    phi=(err1<=err2); %0-1 matrix 
    AkJ=A; AkJ(~phi,:)=II(~phi,:);  
    bk=c+phi.*(b-c); 
   
    Phi=reshape(phi,[],Nt);   
    phi_x=mean(Phi,2);  %NKPA 
    if(nargin>=8)
        Ak_fun=@(r) Ak_PinT(r,phi_x,Nt,Da,D1,Ax,Ix);
    else
        phi_app=kron(ones(Nt,1),phi_x); 
        Ak=II+phi_app.*(A-II); 
        Ak_fun=@(r) Ak\r; %direct solver 
    end  
    [xnew,~,~,iter] =gmres(AkJ,bk,50,1e-10,maxit,Ak_fun,[],x);   
    errx=norm((xnew-x)./max(1,abs(xnew)),inf); %relative change  
    x=xnew; 
    itvec=[itvec;iter(end)]; 
    err1=(A*x-b); err2=(x-c); 
    res=norm(min(err1,err2),inf); %minimum as residual
   % fprintf('policy-iter=%d: res=%1.2e,errx=%1.2e,g-iter=%d, g-flag=%d\n',k,res,errx,iter(end),flag);  
    if(res/res0<tol||errx<tol)  
        break;
    end 
end
end

function z=Ak_PinT(Res,phi_x,Nt,Da,D1,Ax,Ix)
R=reshape(Res,[],Nt);
S1=fft(Da.*(R.')).'; %step (a)
A_phi=phi_x.*Ax+(phi_x-1).*Ix; 
% for j=1:Nt
%     S1(:,j)=(D1(j)*phi_x.*Ix-A_phi)\S1(:,j); %step (b) 
% end  
%solve only half of systems
Nt2=ceil((Nt+1)/2);  
for j=1:Nt2
    S1(:,j)=(D1(j)*phi_x.*Ix-A_phi)\S1(:,j); %step (b), major cost
    %S1(:,j)=agmg((D1(j)*phi_x.*Ix-A_phi),S1(:,j)); %agmg,much slower
end
S1(:,Nt2+1:Nt)=conj(S1(:,floor((Nt+1)/2):-1:2));
X=real((Da.\ifft(S1.')).'); %step (c)  
z=X(:);
end