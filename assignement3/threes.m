load('-ascii','threes')
colormap('gray');
imagesc(reshape(threes(50,:),16,16),[0,1])

%% Mean
mu = mean(threes,1);
colormap('gray'); imagesc(reshape(mu,16,16),[0,1]);

%% Covariance
cov_threes = cov(threes);
[v,d] = eig(cov_threes);
bar(diag(d));

%% Compress dataset
figure;
subplot(1,5,1); colormap('gray'); imagesc(reshape(threes(50,:),16,16),[0,1]);
for i = 1:4
    [Z,E,mu] = pca(threes',1,i);
    reconstruct = E*Z+mu;
    subplot(1,5,i+1);colormap('gray'); imagesc(reshape(reconstruct(:,50),16,16),[0,1]);
end
