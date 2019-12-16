%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML110
% Project Title: Implementation of DBSCAN Clustering in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

clc;
clear;
close all;

%% Load Data
data=load('rings');
X=data.X;
processData = sqrt(X(:,1).^2+X(:,2).^2);

xrange = [min(X(:,1)) max(X(:,1))+2];
yrange = [min(X(:,2)) max(X(:,2))];

%% Run DBSCAN Clustering Algorithm
figure; subplot(1,3,1);
epsilon=0.6;
MinPts=6;
IDX=DBSCAN(X,epsilon,MinPts);
PlotClusterinResult(X, IDX);
title({['A) Raw data'],['(\epsilon = ' num2str(epsilon) ', MinPts = ' num2str(MinPts) ')']}); xlim(xrange); ylim(yrange);

subplot(1,3,2);
epsilon=1.4;
MinPts=10;
IDX=DBSCAN(X,epsilon,MinPts);
PlotClusterinResult(X, IDX);
title({['B) Raw data'],['(\epsilon = ' num2str(epsilon) ', MinPts = ' num2str(MinPts) ')']}); xlim(xrange); ylim(yrange);

subplot(1,3,3);
epsilon=0.2;
MinPts=10;
IDX=DBSCAN(processData,epsilon,MinPts);
PlotClusterinResult(X, IDX);
title({['C) Processed data'],['(\epsilon = ' num2str(epsilon) ', MinPts = ' num2str(MinPts) ')']}); xlim(xrange); ylim(yrange);
