%% perform unsupervised learning with SOM  

% Marco Signoretto, March 2011

close all;
clear all;
clc;

% first we generate data uniformely distributed within two
% concentric cylinders

X=2*(rand(5000,3)-.5);
indx=(X(:,1).^2+X(:,2).^2<.6)&(X(:,1).^2+X(:,2).^2>.1);
X=X(indx,:)';

% we then initialize the SOM with hextop as topology function
% ,linkdist as distance function and gridsize 5x5x5
net = newsom(X,[5 5 5],'hextop','linkdist'); 

% plot the data distribution with the prototypes of the untrained network
figure;plot3(X(1,:),X(2,:),X(3,:),'.g','markersize',10);
axis([-2 2 -2 2]);
hold on
plotsom(net.iw{1},net.layers{1}.distances)
hold off

% finally we train the network and see how their position changes
net.trainParam.epochs = 100;
net = train(net,X);
figure;plot3(X(1,:),X(2,:),X(3,:),'.g','markersize',10);
axis([-1 1 -1 1]);
hold on
plotsom(net.iw{1},net.layers{1}.distances)
hold off
%% Banana data set
load('banana.mat');

s = ceil(sqrt(5*sqrt(size(X,1))));
s = 11;
net = newsom(X',[s s],'hextop','linkdist'); % 'tritop', 'randrop', 'gridtop' / 

% plot the data distribution with the prototypes of the untrained network
figure;
plot(X(:,1),X(:,2),'.','markersize',5);
hold on
plotsom(net.iw{1},net.layers{1}.distances)
hold off
title('Initial SOM prototypes');

% finally we train the network and see how their position changes
net.trainParam.epochs = 1;
net = train(net,X');
figure; subplot(1,3,1);
plot(X(:,1),X(:,2),'.','markersize',2);
hold on
plotsom(net.iw{1},net.layers{1}.distances)
hold off
title('1 epoch');

% pause

net.trainParam.epochs = 9;
net = train(net,X');
subplot(1,3,2);
plot(X(:,1),X(:,2),'.','markersize',2);
hold on
plotsom(net.iw{1},net.layers{1}.distances)
hold off
title('10 epochs');

% pause

net.trainParam.epochs = 90;
net = train(net,X');
subplot(1,3,3);
plot(X(:,1),X(:,2),'.','markersize',2);
hold on
plotsom(net.iw{1},net.layers{1}.distances)
hold off
title('100 epoch');
%  VErklaring uiterste punten liggennite in de neighbourhoud van de andere

%%
subplot(1,3,1); axis equal;
subplot(1,3,2); axis equal;
subplot(1,3,3); axis equal;