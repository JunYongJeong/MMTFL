function [ w_current, fun ] = L1_FISTA_logistic(X,y, lambda,opts)
if isfield(opts, 'init_w')
    w_current=opts.init_w;
    w_old = opts.init_w;    
else    
    w_current = zeros(size(X,2),1);
    w_old = w_current;
end
N  = length(y);

t=1;
t_old=0;
fun =[];

if isfield(opts, 'L')
    L = opts.L;
else
%     [~,D] = eig(X'*X/N);
    [~,~,H] = Loss_logistic(X,y,w_current);
    L  =eigs(H,1)*0.96;
%     L = max(D(:))*0.95;
%     L =eigs(X'*X,1)/N;
end

if max(abs(X))==0
    w_opt = zeros(size(X,2),1);
    fun=0;
    is_contin=0;
end

% 
% [N,dim_input] = size(X);
% iter=1;
% w = zeros(dim_input,1);
% fun=[];

%%
iter=1;
for iter=1:opts.max_iter
%     disp(iter)
    alpha = (t_old-1)/t;
    w_s = (1+alpha)*w_current - alpha*w_old;
    [~,g_s,~] = Loss_logistic(X,y,w_s);
    w_old = w_current;
    w_current = proximalL1norm(w_s - g_s/L, lambda/L);
    fun = cat(1,fun,Loss_logistic(X,y,w_current));

    
    if iter>=2 && fun(end-1)-fun(end) < opts.rel_tol*fun(end-1)
        break;
    end;
    t_old=t;
    t=0.5 * (1+(1+4*t^2)^0.5);

end


%% private function
    function [X] = proximalL1norm(D, tau)
        % min_X 0.5*||X - D||_F^2 + tau*||X||_{1,1}
        % where ||X||_{1,1} = sum_ij|X_ij|, where X_ij denotes the (i,j)-th entry of X
        X = sign(D).*max(0,abs(D)-tau);
    end


end

