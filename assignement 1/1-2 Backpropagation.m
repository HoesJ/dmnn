%% 
clear

%% Compare learning algorithms
clear all ; close all ; clc ;

x = 0:0.05:3*pi ;   % input
y = sin(x.^2) ;     % output
p = con2seq(x) ;    % converts the input to a best format for the toolbox
t = con2seq(y) ;    % converts the output to a best format for the toolbox

algs = ["traingd", "traingda", "traincgf", "traincgp", "trainbfg", "trainlm"];
default = feedforwardnet(50,'traingd') ;   % creates a forst net with the 'trainlm' algorithm
nets = cell(length(algs),1);
a = cell(length(algs),1);

pics = figure;
posts = figure;

for i = 1:length(algs)
    disp(algs(i));
    nets{i} = feedforwardnet(50,algs(i));
    nets{i}.iw{1,1} = default.iw{1,1} ;           % sets the same weights for both networks
    nets{i}.lw{2,1} = default.lw{2,1} ;           % sets the same weights for both networks
    nets{i}.b{1} = default.b{1} ;                 % sets the same biases for both networks
    nets{i}.b{2} = default.b{2} ;
    
    nets{i}.trainParam.epochs = 20 ;            % set the number of epochs for the training (network 1)
    nets{i} = train(nets{i},p,t) ;                % train network 1
    a{i}=sim(nets{i},p) ;                       % simulate network 2 with the input vector p
    
    figure(posts)
    subplot(2,3,i)
    postregm(cell2mat(a{i}),y); 
    title(algs(i));
    
    figure(pics)
    subplot(2,3,i);
    plot(x,y,'bx',x,cell2mat(a{i}),'r');
    title(algs(i));
end

%% Compare noises
clear all ; close all ; clc ;
noises = 0.1:0.1:0.9;
graphs = figure;
posts = figure;
default = feedforwardnet(50,'traincgp') ;

for i = 1:length(noises)

    x = 0:0.05:3*pi ;   % input
    y = sin(x.^2) ;     % output
    noisy_y = y + noises(i)*randn(1,length(y));
    p = con2seq(x) ;    % converts the input to a best format for the toolbox
    t = con2seq(noisy_y) ;    % converts the output to a best format for the toolbox

    net = feedforwardnet(50,'traincgp');
    net.iw{1,1} = default.iw{1,1} ;           % sets the same weights for both networks
    net.lw{2,1} = default.lw{2,1} ;           % sets the same weights for both networks
    net.b{1} = default.b{1} ;                 % sets the same biases for both networks
    net.b{2} = default.b{2} ;
    
    net.trainParam.epochs = 40 ;            % set the number of epochs for the training (network 1)
    net = train(net,p,t) ;                % train network 1
    a = sim(net,p) ;                       % simulate network 2 with the input vector p
    
    figure(posts)
    subplot(3,3,i);
    postregm(cell2mat(a),y); 
    title(noises(i));
    
    figure(graphs)
    subplot(3,3,i);
    plot(x,y,'bx',x,cell2mat(a),'r');
    title(noises(i));
end

%% Number of data points
clear all ; close all ; clc ;
numdatas = linspace(20,260,9);
graphs = figure;
posts = figure;
default = feedforwardnet(50,'traincgp') ;

for i = 1:length(numdatas)

    x = linspace(0,3*pi,numdatas(i)) ;   % input
    y = sin(x.^2) ;     % output
    p = con2seq(x) ;    % converts the input to a best format for the toolbox
    t = con2seq(y) ;    % converts the output to a best format for the toolbox

    net = feedforwardnet(50,'traincgp');
    net.iw{1,1} = default.iw{1,1} ;           % sets the same weights for both networks
    net.lw{2,1} = default.lw{2,1} ;           % sets the same weights for both networks
    net.b{1} = default.b{1} ;                 % sets the same biases for both networks
    net.b{2} = default.b{2} ;
    
    net.trainParam.epochs = 40 ;            % set the number of epochs for the training (network 1)
    net = train(net,p,t) ;                % train network 1
    a = sim(net,p) ;                       % simulate network 2 with the input vector p
    
    figure(posts)
    subplot(3,3,i);
    postregm(cell2mat(a),y); 
    title(numdatas(i));
    
    figure(graphs)
    subplot(3,3,i);
    plot(x,y,'bx',x,cell2mat(a),'r');
    title(numdatas(i));
end

%% personal regression example
load('data_personal_regression_problem.mat');
% r0666420
Tnew = (6*T1 + 6*T2 + 6*T3 + 4 * T4 + 2 * T5) / (6+6+6+4+2);
scatter3(X1,X2,Tnew,'.');
X1 = reshape(X1, [17,800]);
X2 = reshape(X2, [17, 800]);
Tnew = reshape(Tnew, [17, 800]);

tmp1 = reshape(X1(2,:), [4 200]);
tmp2 = reshape(X2(2,:), [4 200]);
tmp3 = reshape(Tnew(2,:), [4 200]);
trainSet = [X1(1,:),tmp1(1,:) ;X2(1,:),tmp2(1,:) ;Tnew(1,:),tmp3(1,:) ];
tmp1 = reshape(X1(4,:), [4 200]);
tmp2 = reshape(X2(4,:), [4 200]);
tmp3 = reshape(Tnew(4,:), [4 200]);
valSet = [X1(3,:),tmp1(1,:) ;X2(3,:),tmp2(1,:) ;Tnew(3,:),tmp3(1,:) ];
tmp1 = reshape(X1(6,:), [4 200]);
tmp2 = reshape(X2(6,:), [4 200]);
tmp3 = reshape(Tnew(6,:), [4 200]);
testSet = [X1(5,:),tmp1(1,:) ;X2(5,:),tmp2(1,:) ;Tnew(5,:),tmp3(1,:) ];

figure
scatter3(trainSet(1,:), trainSet(2,:), trainSet(3,:),'.');
hold on
scatter3(valSet(1,:), valSet(2,:), valSet(3,:),'.');
scatter3(testSet(1,:), testSet(2,:), testSet(3,:),'.');
hold off

%% Define network
p = trainSet(1:2,:);
t = trainSet(3,:);
net = fitnet(50, 'trainlm');
net.inputs{1}.processFcns = {};
net.outputs{2}.processFcns = {};
net = configure(net,p,t);
view(net)

prev_error = 1e20;
curr_error = 1e19;
while (curr_error < prev_error)
   [net,Y,E] = adapt(net,p,t);
   prev_error = curr_error;
   curr_error = sse(sim(net,valSet(1:2,:)) - valSet(3,:))
   curr_error - prev_error
end