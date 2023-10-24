clc
clear
close all
%% Generate data
n = 1000;      % instance number
d = 80;        % feature number
T = 100;       % task number
pos_num = 10;  % the number k of positive samples in each task

data = randn(T, n, d);
label = zeros(T, n, 1);
s = zeros(T, n);

A = rand(T, d);
[U, S, V] = svd(A);
k = 3;
W = U(:,1:k)*S(1:k,1:k)*V(:,1:k)'; % rank-5 approximation
eps = randn(T, n) * 0.0001;

for i = 1: T
   s(i, [1:n]) = W(i, [1:d]) * reshape(data(i,[1:n],[1:d]), n, d)' + eps(i,[1:n]);
end

% scoring the label of top-k instances as positive
for j = 1: pos_num
    [maxVal, maxInd] = max(s');
    for k = 1: T
        label(k, maxInd(k), 1) = 1;
        s(k, maxInd(k)) = -100;
    end
end

clearvars -except data label W


x=-10:0.1:10;
y= -x + 6; %y=xµÄ2´Î·½
plot(x,y)
grid on
axis([-10,6,0,16])
xlabel('x1')
ylabel('x2')