function [beta_opt,fun] = L1_FISTA(X,y,lambda,opts)


%% initilziation


if isfield(opts, 'init_beta')
    beta_current=opts.init_beta;
    beta_old = opts.init_beta;    
else    
    beta_current = zeros(size(X,2),1);
    beta_old = beta_current;
end

N  = length(y);

t=1;
t_old=0;

iter=0;
fun=[];

if isfield(opts, 'L')
    L = opts.L;
else
    [~,D] = eig(X'*X/N);
    L = max(D(:));
end

is_contin=1;
if max(abs(X))==0
    beta_opt = zeros(size(X,2),1);
    fun=0;
    is_contin=0;
end

%% main loop
while iter<opts.max_iter & is_contin
    alpha = (t_old-1)/t;
    beta_s = (1+alpha)*beta_current - alpha*beta_old;
    grad = grad_f(beta_s);
        
    beta_old =beta_current;
    beta_current = proximalL1norm(beta_s - grad/L, lambda/L);      

    fun = cat(1,fun, eval_f(beta_current) + lambda*norm(beta_current,1));
    

    if iter>=2 & fun(end-1) - fun(end) <=opts.rel_tol * fun(end-1)
        break;
    end
    
    iter=iter+1;
    t_old=t;
    t=0.5 * (1+(1+4*t^2)^0.5);

end
beta_opt = beta_current;

%% private function

    function fun=eval_f(beta)
        fun = norm(y - X * beta,2)^2 /(2*N);;
        
    end

    function grad = grad_f(beta)
        grad = X'*(X*beta-y)/N;
    end

end

