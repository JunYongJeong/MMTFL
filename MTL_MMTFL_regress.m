function [A,c_old,B_old,fun] = MTL_MMTFL_regress(X_cell,Y_cell,hyp, opts)
%% Initialization 
% hyp =[gamma1, gamma2)

T = length(Y_cell);
d = size(X_cell{1},2);
c_old = ones(d,1);
exponent = 1/(opts.p+ opts.k);
for t=1:T
    B_old(:,t) = regress(Y_cell{t},X_cell{t});
end

fun =[];
% fun = Obj_multiplicative(c_old,B_old);
%% main
tic
iter=0;
while iter<opts.max_iter
%     disp(iter)

    % update b    
    for t=1:T
        Z = X_cell{t}*diag(c_old);
        switch opts.p
            case 1                
%                 [~,D] = eig(Z'*Z/size(Z,1));
%                 opts.L = D(end);
                [B_new(:,t),b_fun] = L1_FISTA(Z,Y_cell{t}, hyp(1), opts);                
            case 2
                left = Z'*Z/size(Z,1) + 2 * hyp(1) * eye(size(Z,2));
                right = Z'*Y_cell{t}/size(Z,1);
                B_new(:,t) =left\right;
            otherwise
                disp('other value')
        end                
    end
    A =diag(c_old)* B_new;
    
    % update c  

    for j=1:d
        switch opts.p
            case 1
                temp = norm(A(j,:),1);
            case 2
                temp = sum(A(j,:).^2);
        end
        c_new(j,1) = (hyp(1) * temp / hyp(2))^exponent;
    end      
   
    fun = cat(1,fun,Obj_multiplicative(c_new,B_new));
    
    
    
    % evaluation
    diff = diag(c_old) * B_old - diag(c_new)*B_new;
    if max(abs(diff(:)))<=10^-4
        break;
    end
    
    if iter>=2 & abs(fun(end)-fun(end-1))<= opts.rel_tol*fun(end-1)
        break;
    end           
    % stopping criteria       
    iter=iter+1;
    c_old=c_new;
    B_old=B_new;
end


    %%
    
    function val = Obj_multiplicative(c,B)
        
        val = hyp(2) * norm(c,opts.k)^opts.k;
        
        for t2= 1:T
            val = val + hyp(1) * norm(B(:,t2),opts.p)^opts.p;
        end        
        
        for t2=1:T
            val = val + norm(Y_cell{t2} - X_cell{t2} * diag(c) * B(:,t2) )^2 /(2*length(Y_cell{t2}));
        end             
    end

%    




end

