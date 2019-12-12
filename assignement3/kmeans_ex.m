clear; close all; clc;
load('rings.mat');
xrange = [min(X(:,1)) max(X(:,1))+2];
yrange = [min(X(:,2)) max(X(:,2))];
%%
figure; subplot(1,2,1);
plot(X(Y==0,1),X(Y==0,2),'.','MarkerSize',7);
hold on
plot(X(Y==1,1),X(Y==1,2),'.','MarkerSize',7);
plot(X(Y==2,1),X(Y==2,2),'.','MarkerSize',7);
hold off
legend('cluster 1', 'cluster 2','cluster 3'); xlim(xrange); ylim(yrange);
title 'A) Correct clustering'; 
%% apply k-means with three centers 1
num = 3;
[idx,C] = kmeans(X,num,'Distance','sqeuclidean','Replicates',100);
subplot(1,2,2);
hold on
for i = 1:num
    plot(X(idx==i,1),X(idx==i,2),'.','MarkerSize',7)
end
plot(C(:,1),C(:,2),'kx','MarkerSize',10,'LineWidth',3)
legend('cluster 1', 'cluster 2','cluster 3','Centroids'); xlim(xrange); ylim(yrange);
title 'Kmeans clustering'
hold off
%% apply k-means with three centers 100
% num = 3;
% [idx,C] = kmeans(X,num,'Distance','sqeuclidean','Replicates',100);
% subplot(1,3,3);
% hold on
% for i = 1:num
%     plot(X(idx==i,1),X(idx==i,2),'.','MarkerSize',7)
% end
% plot(C(:,1),C(:,2),'kx','MarkerSize',10,'LineWidth',3)
% legend('cluster 1', 'cluster 2','cluster 3','Centroids'); xlim(xrange); ylim(yrange);
% title 'Kmeans clustering - 100 iteration'
% hold off