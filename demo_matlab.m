clc
clear
warning off;
%% Regression
load('schoo_rep1.mat')
opts.p=1;
opts.k=1;
opts.max_iter=1000;
opts.rel_tol=1e-4;

hyp = [1,0.1];
W = MTL_MMTFL_regress(school_train_input, school_train_output, hyp, opts);

for task=1:139
    y_test_hat = school_test_input{task} * W(:,task);
    resi = school_test_output{task} - y_test_hat;
    rmse(task) = sqrt(mean(resi.^2));
end
fprintf(sprintf('School dataset RMSE: %f\n',mean(rmse)));

%% Classification
load('mnistPCA_1k.mat')

hyp2= [0.1,0.1];
W_mnist= MTL_MMTFL_regress(X_train,Y_train, hyp2, opts);

for task=1:10
    Y_test_hat{task} = sign(1./(1+exp(-X_test{task}*W_mnist(:,task)))-0.5);
end
Y_test_vec = cell2mat(Y_test);
Y_test_hat_vec = cell2mat(Y_test_hat);
Err = mean(Y_test_vec~=Y_test_hat_vec(:));
fprintf(sprintf('Mnist dataset Error Rate: %f\n',Err));

