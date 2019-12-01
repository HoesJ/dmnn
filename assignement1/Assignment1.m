%% 1
clear



%%
clear all ; close all ; clc ;

x = 0:0.05:3*pi ;   % input
y = sin(x.^2) ;     % output
% noisy_y = y + 0.6*randn(1,length(y));

p = con2seq(x) ;    % converts the input to a best format for the toolbox
t = con2seq(y) ;    % converts the output to a best format for the toolbox

algs = ["traingd", "traingda", "traincgf", "traincgp", "trainbfg", "trainlm"];
default = feedforwardnet(50,'traingd') ;   % creates a forst net with the 'trainlm' algorithm
nets = cell(length(algs),1);
a = cell(length(algs),1);

pics = figure;

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
    
    figure
    postreg(cell2mat(a{i}),y); 
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

%% Compare noises
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