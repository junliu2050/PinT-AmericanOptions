function [x]=mg_iter_2d(mg,x0,b,level,pre,post,smoother,shift,j)
%fprintf('+++++++++++++++++++Level[%d] Begin+++++++++++++++++++++\n',level);
 global LEVEL0 
if(level==LEVEL0) %Coarest level
    x=(shift*mg(level).Ix-mg(level).Ax)\b;
else
    % presmooth
    x = mg_smooth(mg(level),x0,b,pre,smoother,shift,j);
    % Restrict residual
    r = b + mg(level).Ax*x - shift*x;
    %rc = mg_restricth(r,h);
    %rc = rest(r);
    rc=(mg(level).P)'*r/4;
    % coarse grid correction
    cc = mg_iter_2d(mg,zeros(size(rc)),rc,level-1,pre,post,smoother,shift,j);
    %cc = mg_iter_2d(mg,cc,rc,level-1,pre,post,smoother,shift); %add this for W cycle
    %x = x + intp2(cc);
    %x=x + mg_interph(cc,h/2);
    x = x + mg(level).P*cc; %
    % postsmooth
    %fprintf('Level[%d],Before Postsmooth res=%1.2e\n',level,norm(b-A*x));
    x = mg_smooth(mg(level),x,b,post,smoother,shift,j);
    %fprintf('Level[%d],After Postsmooth res=%1.2e\n',level,norm(b-A*x));
end
%fprintf('+++++++++++++++++++Level[%d] End+++++++++++++++++++++\n',level);
end
function [x]=mg_smooth(mglev,x0,b,nv,smoother,shift,j)
x=x0;
switch smoother
    case 'ILU'
        for k = 1:nv
            x=x + mglev.AU{j}\(mglev.AL{j}\(b+mglev.Ax*x-shift*x));
        end
    case 'Jacobi'
        w=4/5;
        dinv = 1./(-diag(mglev.Ax)+shift);
        for k = 1:nv
            x=x + w*dinv.*(b + mglev.Ax*x-shift*x);
        end
    case 'GS' %G-S Smoother
        L = -tril(mglev.Ax)+shift*mglev.Ix;
        for k = 1:nv
            x=x + (L\(b + mglev.Ax*x-shift*x));
        end
end
end
