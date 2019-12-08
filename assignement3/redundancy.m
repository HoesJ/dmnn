%% Gaussian X
X = randn(50,500);
[Z,E,mu] = pca(X,0.8);
Xhat = E*Z+mu;

sqrt(mean(mean((X-Xhat).^2)))
%% Choles all
load('choles_all')

X = p;
[Z,E,mu,sort_d] = pca(X,0.98);
Xhat = E*Z+mu;

sqrt(mean(mean((X-Xhat).^2)))

%% Matlabs pca
