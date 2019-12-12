clear; clc; close all;
load('-ascii','threes');
figure;
colormap('gray');
imagesc(reshape(threes(50,:),16,16),[0,1]);

%% Mean
mu = mean(threes,1);
figure; subplot(1,2,1);
colormap('gray'); imagesc(reshape(mu,16,16),[0,1]); title('A) Average of dataset');

%% Covariance
cov_threes = cov(threes);
[v,d] = eig(cov_threes);
subplot(1,2,2);
bar(flip(diag(d))); xlim([1 256]); title('B) Eigenvalues'); ylabel('eigenvalue');

%% Compress dataset
figure;
piece = 50;
subplot(1,5,1); colormap('gray'); imagesc(reshape(threes(piece,:),16,16),[0,1]); title('256');
for i = 1:4
    [Z,E,mu] = pca(threes',1,i);
    reconstruct = E*Z+mu;
%     subplot(1,5,i+1);colormap('gray'); imagesc(reshape(mean(reconstruct,2),16,16),[0,1]); title(num2str(i));
    subplot(1,5,i+1);colormap('gray'); imagesc(reshape(reconstruct(:,piece),16,16),[0,1]); title(num2str(i));
end

%% Compress multiple times
figure; subplot(1,2,1);
range = 1:50;
errs = zeros(length(range),1);
for i = range
    [Z,E,mu] = pca(threes',1,i);
    reconstruct = E*Z+mu;
    errs(i) = sqrt(mean(mean((reconstruct-threes').^2)));
end
plot(range, errs, 'linewidth', 2); xlabel('dimensions kept'); ylabel('reconstruction error (RMSE)'); title('A) Reconstruction error');
%% Vector
subplot(1,2,2);
v = flip(cumsum(diag(d)));
plot(v, 'linewidth', 2); xlim([1 256]); xlabel('i'); ylabel('cumsum of 256-i smallest eigs'); title('B) Cumulative sum');
