%% Gaussian X
X = randn(50,500);
errs = zeros(50,1);
for i = 1:50
[Z,E,mu,d] = pca(X,0.8,i);
Xhat = E*Z+mu;
errs(i) = sqrt(mean(mean((X-Xhat).^2)));
end
figure; 
subplot(2,2,1); bar(d); ylabel('Eigenvalues'); title('A) Gaussian data'); xlim([1 50]);
subplot(2,2,3); plot(errs, 'linewidth', 2); xlabel('dimensions kept'); ylabel('Error estimate');xlim([1 50]);

%% Choles all
load('choles_all')
X = p;

errs = zeros(21,1);
for i = 1:21
[Z,E,mu,d] = pca(X,0.8,i);
Xhat = E*Z+mu;
errs(i) = sqrt(mean(mean((X-Xhat).^2)));
end
subplot(2,2,2); bar(d); ylabel('Eigenvalues'); title('B) Choles data');xlim([1 21]);
subplot(2,2,4);plot(errs, 'linewidth', 2);  xlabel('dimensions kept'); ylabel('Error estimate');xlim([1 21]);

%% Matlabs pca
load('choles_all')
X = p;
[X,PS1] = mapstd(X);
[Y,PS] = processpca(X, 0.04);
