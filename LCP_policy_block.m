function [x,k,res,phi]=LCP_policy_block(A,b,c,x,tol,maxit)
%policy iteration after projection to reduced system 
err1=(A*x-b); err2=(x-c); res0=norm(min(err1,err2),inf); 
for k=1:maxit
    phi=(err1<=err2);   
    %split into 0-1 blocks
    idx0=find(~phi);idx1=find(phi);
    x(idx0)=c(idx0); %first solve identity block then leftover block 
    x(idx1)=A(idx1,idx1)\(b(idx1)-A(idx1,idx0)*c(idx0)); 
    err1=(A*x-b); err2=(x-c); 
    res=norm(min(err1,err2),inf);
    %fprintf('policy-iter=%d: res/res0=%1.2e\n',k,res/res0);  
    if(res/res0<tol) 
        break;
    end 
end
end