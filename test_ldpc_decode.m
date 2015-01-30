load('test_code.mat')

f1=1./(1+exp(-2*z/sigma^2));        % likelihoods
f0=1-f1;
%prevent f from becoming exactly 0.5
f1(f1==0.5) = 0.5+1e-20;
f0(f0==0.5) = 0.5-1e-20;


%decode conventionally
tic
for i = 1:10
    [z_hat, success, k] = ldpc_decode(f0,f1,full(H),100);
end
toc
x_hat = z_hat(size_G(2)+1-size_G(1):size_G(2));
b = x_hat';

nErrors = sum(x ~= b)
k
success
