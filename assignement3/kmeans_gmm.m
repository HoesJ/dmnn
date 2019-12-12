clear; close all; clc;
load('rings.mat');
xrange = [min(X(:,1)) max(X(:,1))+2];
yrange = [min(X(:,2)) max(X(:,2))];
%%
figure; subplot(1,3,1);
plot(X(Y==0,1),X(Y==0,2),'.','MarkerSize',7);
hold on
plot(X(Y==1,1),X(Y==1,2),'.','MarkerSize',7);
plot(X(Y==2,1),X(Y==2,2),'.','MarkerSize',7);
hold off
legend('1', '2','3'); xlim(xrange); ylim(yrange);
title 'A) Correct clustering'; 
%% apply k-means with three centers 1
num = 3;
[idx,C] = kmeans(X,num,'Distance','sqeuclidean','Replicates',100);
subplot(1,3,2);
hold on

h1 = gscatter(X(:,1),X(:,2),idx);

x1 = xrange(1):0.05:xrange(2);
x2 = yrange(1):0.05:yrange(2);
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot
idx2Region = kmeans(XGrid,num,'MaxIter',1,'Start',C);
h2 = gscatter(XGrid(:,1),XGrid(:,2),idx2Region,[h1(1).Color*0.75 - 0.5*(h1(1).Color - 1);h1(2).Color*0.75 - 0.5*(h1(2).Color - 1);h1(3).Color*0.75 - 0.5*(h1(3).Color - 1)],'..');
uistack(h2,'bottom');

plot(C(:,1),C(:,2),'kx','MarkerSize',10,'LineWidth',3)
legend('1', '2','3'); xlim(xrange); ylim(yrange);
title 'B) Kmeans clustering'
hold off
%% GMM
k = 3; % Number of GMM components
options = statset('MaxIter',1000);

%Create a 2-D grid
d = 500; % Grid length
x1 = linspace(min(X(:,1))-2, max(X(:,1))+2, d);
x2 = linspace(min(X(:,2))-2, max(X(:,2))+2, d);
[x1grid,x2grid] = meshgrid(x1,x2);
X0 = [x1grid(:) x2grid(:)];

threshold = sqrt(chi2inv(0.99,2));
gmfit = fitgmdist(X,k,'CovarianceType','diagonal','SharedCovariance',false,'Options',options); % Fitted GMM
clusterX = cluster(gmfit,X); % Cluster index 
mahalDist = mahal(gmfit,X0); % Distance from each grid point to each GMM component

% Draw ellipsoids over each GMM component and show clustering result.
subplot(1,3,3);
h1 = gscatter(X(:,1),X(:,2),clusterX);
hold on
    for m = 1:k
        idx = mahalDist(:,m)<=threshold;
        Color = h1(m).Color*0.75 - 0.5*(h1(m).Color - 1);
        h2 = plot(X0(idx,1),X0(idx,2),'.','Color',Color,'MarkerSize',1);
        uistack(h2,'bottom');
    end    
plot(gmfit.mu(:,1),gmfit.mu(:,2),'kx','LineWidth',2,'MarkerSize',10)
title('C) Gaussian Mixture Model')
legend(h1,{'1','2','3'});  xlim(xrange); ylim(yrange);
hold off
%% Preprocessing
processData = sqrt(X(:,1).^2+X(:,2).^2);
%%
figure; subplot(1,3,1);
plot(processData(Y==0),'.','MarkerSize',7);
hold on
plot(processData(Y==1),'.','MarkerSize',7);
plot(processData(Y==2),'.','MarkerSize',7);
hold off
legend('1', '2','3'); xlabel('index'); ylabel('radius');
title 'A) Preprocessed data'; 
%% apply k-means with three centers 1
num = 3;
[idx,C] = kmeans(processData,num,'Distance','sqeuclidean','Replicates',100);
subplot(1,3,2);
hold on

h1 = gscatter(X(:,1),X(:,2),idx);

x1 = xrange(1):0.05:xrange(2);
x2 = yrange(1):0.05:yrange(2);
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot
idx2Region = kmeans(sqrt(XGrid(:,1).^2+XGrid(:,2).^2),num,'MaxIter',1,'Start',C);
h2 = gscatter(XGrid(:,1),XGrid(:,2),idx2Region,[h1(1).Color*0.75 - 0.5*(h1(1).Color - 1);h1(2).Color*0.75 - 0.5*(h1(2).Color - 1);h1(3).Color*0.75 - 0.5*(h1(3).Color - 1)],'..');
uistack(h2,'bottom');

plot(C*cos(0:0.01:2*pi),C*sin(0:0.01:2*pi), '.k', 'MarkerSize', 1);
legend('1', '2','3'); xlim(xrange); ylim(yrange);
title 'B) Kmeans clustering'
hold off

%% GMM on preprocessed
k = 3; % Number of GMM components
options = statset('MaxIter',1000);
Sigma = {'diagonal'};
SharedCovariance = {false};
SCtext = {'false'};

%Create a 2-D grid
d = 500; % Grid length
x1 = linspace(min(X(:,1))-2, max(X(:,1))+2, d);
x2 = linspace(min(X(:,2))-2, max(X(:,2))+2, d);
[x1grid,x2grid] = meshgrid(x1,x2);
X0 = [x1grid(:) x2grid(:)];

threshold = sqrt(chi2inv(0.99,2));
gmfit = fitgmdist(processData,k,'CovarianceType','diagonal','SharedCovariance',false,'Options',options); % Fitted GMM
clusterX = cluster(gmfit,processData); % Cluster index 
mahalDist = mahal(gmfit,sqrt(X0(:,1).^2+X0(:,2).^2)); % Distance from each grid point to each GMM component

% Draw ellipsoids over each GMM component and show clustering result.
subplot(1,3,3);
h1 = gscatter(X(:,1),X(:,2),clusterX);
hold on
    for m = 1:k
        idx = mahalDist(:,m)<=threshold;
        Color = h1(m).Color*0.75 - 0.5*(h1(m).Color - 1);
        h2 = plot(X0(idx,1),X0(idx,2),'.','Color',Color,'MarkerSize',1);
        uistack(h2,'bottom');
    end
plot(gmfit.mu*cos(0:0.01:2*pi),gmfit.mu*sin(0:0.01:2*pi), '.k', 'MarkerSize', 1);
title('C) Gaussian Mixture Model')
legend(h1,{'1','2','3'});  xlim(xrange); ylim(yrange);
hold off