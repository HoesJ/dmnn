clear
clc
close all

%Load data
load('covtype.mat')


%% Training the SOM
x_length = 10;
y_length = 10;
gridsize=[y_length x_length];
net = newsom(X',gridsize,'hextop','linkdist');

net.trainParam.epochs = 1000;
net = train(net,X');
%% Eval
% load('SOM_net_12x12.mat');
% load('SOM_net_3x3.mat');
% load('SOM_net_10x10.mat');

% Assigning examples to clusters
outputs = sim(net,X');
[~,assignment]  =  max(outputs);

%Compare clusters with true labels
ARI=RandIndex(assignment,Y');
