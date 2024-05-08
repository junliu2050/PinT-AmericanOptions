function [x,k,res,Phi]=LCP_policy(A,b,c,x,tol,maxit)
%The direct policy iteration
II=speye(size(A,1));   
err1=(A*x-b); err2=(x-c); res0=norm(min(err1,err2),inf); 
Phi=x;
format long
for k=1:maxit
    phi=(err1<=err2);  
    Ak=II; Ak(phi,:)=A(phi,:);       
    bk=c; bk(phi)=b(phi);
    x=Ak\bk;  Phi=[Phi,x];
    err1=(A*x-b); err2=(x-c); 
    res=norm(min(err1,err2),inf);
    %fprintf('policy-iter=%d: res=%1.2e\n',k,res);  
    if(res/res0<tol) 
        break;
    end 
end
end