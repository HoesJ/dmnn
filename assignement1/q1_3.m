load('data_personal_regression_problem.mat')

d1 = 8; d2 = 7; d3 = 6; d4 = 6; d5 = 3;

X = [X1, X2];
Tnew = (d1*T1+d2*T2+d3*T3+d4*T4+d5*T5)/(d1+d2+d3+d4+d5);
%% create training, validation, test sets
sel = randperm(13600);

% training set
Xtr = X(sel(1:1000), :);
Ttr = Tnew(sel(1:1000), :);

% validation set
Xv = X(sel(1001:2000), :);
Tv = Tnew(sel(1001:2000), :);

% test set
Xt = X(sel(2001:3000), :);
Tt = Tnew(sel(2001:3000), :);

%% 基于散点通过插值去画曲面的图像
x1 = Xtr(:, 1);
y1 = Xtr(:, 2);
z1 = Ttr;

count=1;     % 新变量计数器
interval=1; % 抽稀间隔
for i = 1 : interval : length(x1),
    x(count, 1) = x1(i, 1);
    y(count, 1) = y1(i, 1);
    z(count, 1) = z1(i, 1);
    count = count + 1;
end

% 确定网格坐标（x和y方向的步长均取0.1）
[Xplot,Yplot]=meshgrid(min(x):0.1:max(x),min(y):0.1:max(y)); 
% 在网格点位置插值求Z，注意：不同的插值方法得到的曲线光滑度不同
Zplot=griddata(x,y,z,Xplot,Yplot,'v4');
% 绘制曲面
figure;
surf(Xplot,Yplot,Zplot);
%shading interp;
%colormap(autumn(5));
% view(0, 90);
colorbar;


%% Build and train my feedforward Neural Networks: 
% the first sturcture which is [50, 1]; 
iteration = 50;
Err = 0;

for i = 1:iteration,
    trainFcn = 'trainlm';
    hiddenLayerSize = 50;
    net = feedforwardnet(hiddenLayerSize, trainFcn);

    net.trainParam.epochs = 50;
    net.performFcn = 'mse';

    p = con2seq(Xtr');
    t = con2seq(Ttr');
    [net,tr] = train(net,p,t); 

    % this is the mse on validation set
    pv = con2seq(Xv');
    tv = con2seq(Tv');
    Thv = net(pv);
    err = perform(net, tv, Thv);
    Err = Err + err;
end

Err = Err/iteration;
fprintf('The MSE of structure [50, 1] is %d \n', Err); 


%% the second sturcture which is [20, 1]; 
iteration = 1;
Err = 0;

for i = 1:iteration,
    trainFcn = 'trainlm';
    hiddenLayerSize = 20;
    net = feedforwardnet(hiddenLayerSize, trainFcn);

    net.trainParam.epochs = 50;
    net.performFcn = 'mse';

    p = con2seq(Xtr');
    t = con2seq(Ttr');
    [net,tr] = train(net,p,t); 

    % this is the mse on validation set
    pv = con2seq(Xv');
    tv = con2seq(Tv');
    Thv = net(pv);
    err = perform(net, tv, Thv);
    Err = Err + err;
end

Err = Err/iteration;
fprintf('The MSE of structure [20, 1] is %d \n', Err); 

%% the third sturcture which is [50, 8, 1]; 
iteration = 50;
Err = 0;

for i = 1:iteration,
    trainFcn = 'trainlm';
    hiddenLayerSize = 50;
    net = feedforwardnet(hiddenLayerSize, trainFcn);

    net.trainParam.epochs = 50;
    net.performFcn = 'mse';

    net.numLayers = 3;
    net.layerConnect(3,2) = 1;
    %net.layerConnect(4,3) = 1;
    net.outputConnect = [0 0 1];
    net.layers{2}.size = 8;
    net.layers{2}.transferFcn = 'tansig';
    %net.layers{3}.size = 6;
    %net.layers{3}.transferFcn = 'tansig';
    net.biasConnect(3) = 1;
    %net.biasConnect(4) = 1;

    p = con2seq(Xtr');
    t = con2seq(Ttr');
    [net,tr] = train(net,p,t); 

    % this is the mse on validation set
    pv = con2seq(Xv');
    tv = con2seq(Tv');
    Thv = net(pv);
    err = perform(net, tv, Thv);
    Err = Err+err;
end

Err = Err/iteration;
fprintf('The MSE of structure [50, 8, 1] is %d \n', Err); 

%% the fourth structure which is [20, 8, 1]; 
iteration = 50;
Err = 0;

for i = 1:iteration,
    trainFcn = 'trainlm';
    hiddenLayerSize = 20;
    net = feedforwardnet(hiddenLayerSize, trainFcn);

    net.trainParam.epochs = 50;
    net.performFcn = 'mse';

    net.numLayers = 3;
    net.layerConnect(3,2) = 1;
    %net.layerConnect(4,3) = 1;
    net.outputConnect = [0 0 1];
    net.layers{2}.size = 8;
    net.layers{2}.transferFcn = 'tansig';
    %net.layers{3}.size = 6;
    %net.layers{3}.transferFcn = 'tansig';
    net.biasConnect(3) = 1;
    %net.biasConnect(4) = 1;

    p = con2seq(Xtr');
    t = con2seq(Ttr');
    [net,tr] = train(net,p,t); 

    % this is the mse on validation set
    pv = con2seq(Xv');
    tv = con2seq(Tv');
    Thv = net(pv);
    err = perform(net, tv, Thv);
    Err = Err + err;
end

Err = Err/iteration;
fprintf('The MSE of structure [20, 8, 1] is %d \n', Err); 


%% the fifth structure which is [10, 8, 3, 1]; 
iteration = 1;
Err = 0;

for i = 1:iteration,
    trainFcn = 'trainlm';
    hiddenLayerSize = 10;
    net = feedforwardnet(hiddenLayerSize, trainFcn);

    net.trainParam.epochs = 50;
    net.performFcn = 'mse';

    net.numLayers = 4;
    net.layerConnect(3,2) = 1;
    net.layerConnect(4,3) = 1;
    net.outputConnect = [0 0 0 1];
    net.layers{2}.size = 8;
    net.layers{2}.transferFcn = 'tansig';
    net.layers{3}.size = 3;
    net.layers{3}.transferFcn = 'tansig';
    net.biasConnect(3) = 1;
    net.biasConnect(4) = 1;

    p = con2seq(Xtr');
    t = con2seq(Ttr');
    [net,tr] = train(net,p,t); 

    % this is the mse on validation set
    pv = con2seq(Xv');
    tv = con2seq(Tv');
    Thv = net(pv);
    err = perform(net, tv, Thv);
    Err = Err + err;
end

Err = Err/iteration;
fprintf('The MSE of structure [10, 8, 3, 1] is %d \n', Err); 

%% network [20, 8, 1] on the test set.

    trainFcn = 'trainlm';
    hiddenLayerSize = 20;
    net = feedforwardnet(hiddenLayerSize, trainFcn);

    net.trainParam.epochs = 50;
    net.performFcn = 'mse';

    net.numLayers = 3;
    net.layerConnect(3,2) = 1;
    %net.layerConnect(4,3) = 1;
    net.outputConnect = [0 0 1];
    net.layers{2}.size = 8;
    net.layers{2}.transferFcn = 'tansig';
    %net.layers{3}.size = 6;
    %net.layers{3}.transferFcn = 'tansig';
    net.biasConnect(3) = 1;
    %net.biasConnect(4) = 1;

    p = con2seq(Xtr');
    t = con2seq(Ttr');
    [net,tr] = train(net,p,t); 

    % this is the mse on validation set
    pt = con2seq(Xt');
    tt = con2seq(Tt');
    Tht = net(pt);
    err = perform(net, tt, Tht);

fprintf('The MSE of structure [20, 8, 1] is %d \n', err); 

%% plot the test set
x1 = Xt(:, 1);
y1 = Xt(:, 2);
z1 = cell2mat(Tht)';

count=1;     % 新变量计数器
interval=1; % 抽稀间隔
for i = 1 : interval : length(x1),
    x(count, 1) = x1(i, 1);
    y(count, 1) = y1(i, 1);
    z(count, 1) = z1(i, 1);
    count = count + 1;
end

% 确定网格坐标（x和y方向的步长均取0.1）
[Xplot,Yplot]=meshgrid(min(x):0.1:max(x),min(y):0.1:max(y)); 
% 在网格点位置插值求Z，注意：不同的插值方法得到的曲线光滑度不同
Zplot=griddata(x,y,z,Xplot,Yplot,'v4');
% 绘制曲面
figure;
surf(Xplot,Yplot,Zplot);
%shading interp;
%colormap(autumn(5));
% view(0, 90);
colorbar;

hold on;
scatter3(x1,y1,Tt,'r.')
legend('approximation surface' ,'test data');

%% Error level curve.
x1 = Xt(:, 1);
y1 = Xt(:, 2);
z1 = (abs(cell2mat(Tht)' - Tt));

count=1;     % 新变量计数器
interval=1; % 抽稀间隔
for i = 1 : interval : length(x1),
    x(count, 1) = x1(i, 1);
    y(count, 1) = y1(i, 1);
    z(count, 1) = z1(i, 1);
    count = count + 1;
end

% 确定网格坐标（x和y方向的步长均取0.1）
[Xplot,Yplot]=meshgrid(min(x):0.1:max(x),min(y):0.1:max(y)); 
% 在网格点位置插值求Z，注意：不同的插值方法得到的曲线光滑度不同
Zplot=griddata(x,y,z,Xplot,Yplot,'v4');
% 绘制曲面
figure;
surf(Xplot,Yplot,Zplot);
%shading interp;
%colormap(autumn(5));
% view(0, 90);
colorbar;
