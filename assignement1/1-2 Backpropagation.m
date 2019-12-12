
%% Compare learning algorithms with epochs
clear all; close all; clc;

x = 0:0.05:3 * pi; % input
y = sin(x.^2); % output
p = con2seq(x); % converts the input to a best format for the toolbox
t = con2seq(y); % converts the output to a best format for the toolbox

algs = ['traingd', 'traingda', 'traincgf', 'traincgp', 'trainbfg', 'trainlm'];
epochs = 5:10:100;
default = feedforwardnet(50, 'traingd'); % creates a first net with the 'trainlm' algorithm
nets = cell(length(algs), 1);
a = cell(length(algs), 1);

pics = figure;
posts = figure;

for i = 1:length(algs)
    disp(algs(i));
    nets{i} = feedforwardnet(50, algs(i));
    nets{i}.iw{1, 1} = default.iw{1, 1}; % sets the same weights for both networks
    nets{i}.lw{2, 1} = default.lw{2, 1}; % sets the same weights for both networks
    nets{i}.b{1} = default.b{1}; % sets the same biases for both networks
    nets{i}.b{2} = default.b{2};

    nets{i}.trainParam.epochs = 20; % set the number of epochs for the training (network 1)
    nets{i} = train(nets{i}, p, t); % train network 1
    a{i} = sim(nets{i}, p); % simulate network 2 with the input vector p

    figure(posts)
    subplot(2, 3, i)
    postregm(cell2mat(a{i}), y);
    title(algs(i));

    figure(pics)
    subplot(2, 3, i);
    plot(x, y, 'bx', x, cell2mat(a{i}), 'r');
    title(algs(i));
end

%% Compare noises
clear all; close all; clc;
noises = 0.1:0.1:0.9;
graphs = figure;
posts = figure;
default = feedforwardnet(50, 'traincgp');

for i = 1:length(noises)

    x = 0:0.05:3 * pi; % input
    y = sin(x.^2); % output
    noisy_y = y + noises(i) * randn(1, length(y));
    p = con2seq(x); % converts the input to a best format for the toolbox
    t = con2seq(noisy_y); % converts the output to a best format for the toolbox

    net = feedforwardnet(50, 'traincgp');
    net.iw{1, 1} = default.iw{1, 1}; % sets the same weights for both networks
    net.lw{2, 1} = default.lw{2, 1}; % sets the same weights for both networks
    net.b{1} = default.b{1}; % sets the same biases for both networks
    net.b{2} = default.b{2};

    net.trainParam.epochs = 40; % set the number of epochs for the training (network 1)
    net = train(net, p, t); % train network 1
    a = sim(net, p); % simulate network 2 with the input vector p

    figure(posts)
    subplot(3, 3, i);
    postregm(cell2mat(a), y);
    title(noises(i));

    figure(graphs)
    subplot(3, 3, i);
    plot(x, y, 'bx', x, cell2mat(a), 'r');
    title(noises(i));
end

%% Number of data points
clear all; close all; clc;
numdatas = linspace(20, 260, 9);
graphs = figure;
posts = figure;
default = feedforwardnet(50, 'traincgp');

for i = 1:length(numdatas)

    x = linspace(0, 3 * pi, numdatas(i)); % input
    y = sin(x.^2); % output
    p = con2seq(x); % converts the input to a best format for the toolbox
    t = con2seq(y + 0.3*randn(1, length(y))); % converts the output to a best format for the toolbox

    net = feedforwardnet(50, 'traincgp');
    net.iw{1, 1} = default.iw{1, 1}; % sets the same weights for both networks
    net.lw{2, 1} = default.lw{2, 1}; % sets the same weights for both networks
    net.b{1} = default.b{1}; % sets the same biases for both networks
    net.b{2} = default.b{2};

    net.trainParam.epochs = 40; % set the number of epochs for the training (network 1)
    net = train(net, p, t); % train network 1
    a = sim(net, p); % simulate network 2 with the input vector p

    figure(posts)
    subplot(3, 3, i);
    postregm(cell2mat(a), y);
    title(numdatas(i));

    figure(graphs)
    subplot(3, 3, i);
    plot(x, y, 'bx', x, cell2mat(a), 'r');
    title(numdatas(i));
end
%% Overparamaterised Bayes
x = linspace(0.05,3*pi,190); % input
y = sin(x.^2); % output

vars = 0:0.02:0.9;
% vars = 2:3:200;
default = feedforwardnet(500, 'traingd'); % creates a first net with the 'trainlm' algorithm

Rs = zeros(length(vars));
times = zeros(length(vars));
for it = 1:10
    for j = 1:length(vars)
        net = feedforwardnet(500, 'trainbr');
        net.iw{1, 1} = default.iw{1, 1}; % sets the same weights for both networks
        net.lw{2, 1} = default.lw{2, 1}; % sets the same weights for both networks
        net.b{1} = default.b{1}; % sets the same biases for both networks
        net.b{2} = default.b{2};
        
        noisy_y = y + vars(j) * randn(1, length(y));
        p = con2seq(x);
        t = con2seq(noisy_y);
        net.trainParam.showWindow = 1;
        net.trainParam.epochs = 200; % set the number of epochs for the training (network 1)
        tic
        net = train(net, p, t); % train network 1
        time = toc;
        
        out = sim(net, p); % simulate network 2 with the input vector p
        out = cell2mat(out);

        [~,~,tmp] = postregm((out), y);
        Rs(j,i) = ((it-1)*Rs(j,i) + tmp) / it;
        times(j,i) = ((it-1)*times(j,i) + time)/it;
        disp(strcat(num2str(it),' -- ',num2str(vars(j))));
    end
end

%% Compare training to vars
x = linspace(0.05,3*pi,190); % input
y = sin(x.^2); % output

algs = {'traingd', 'traingda', 'traincgf', 'traincgp', 'trainbfg', 'trainlm', 'trainbr'};
vars = 0:0.02:0.9;
% vars = 2:3:200;
default = feedforwardnet(15, 'traingd'); % creates a first net with the 'trainlm' algorithm

Rs = zeros(length(vars), length(algs));
times = zeros(length(vars), length(algs));
for it = 1:10
for i = 1:length(algs)
    for j = 1:length(vars)
        net = feedforwardnet(15, char(algs(i)));
        net.iw{1, 1} = default.iw{1, 1}; % sets the same weights for both networks
        net.lw{2, 1} = default.lw{2, 1}; % sets the same weights for both networks
        net.b{1} = default.b{1}; % sets the same biases for both networks
        net.b{2} = default.b{2};
        
        noisy_y = y + vars(j) * randn(1, length(y));
        p = (x);
        t = (noisy_y);
        if (strcmp(char(algs(i)),'trainbr'))
            p = con2seq(x);
            t = con2seq(noisy_y);
        end
%         t = con2seq(y);
        net.trainParam.showWindow = 0;
        net.trainParam.epochs = 200; % set the number of epochs for the training (network 1)
        tic
        net = train(net, p, t); % train network 1
        time = toc;
        
        out = sim(net, p); % simulate network 2 with the input vector p
        if (strcmp(char(algs(i)),'trainbr'))
            out = cell2mat(out);
        end
        [~,~,tmp] = postregm((out), y);
        Rs(j,i) = ((it-1)*Rs(j,i) + tmp) / it;
        times(j,i) = ((it-1)*times(j,i) + time)/it;
        disp(strcat(algs(i),' -- ',num2str(vars(j))));
    end
end
end
%%
plot(vars, fliplr(Rs), 'linewidth', 2);
xlabel('noise rate');
ylabel('Correlation coefficient');
legend(fliplr(algs));
%%
algs = {'traingd', 'traingda', 'traincgf', 'traincgp', 'trainbfg', 'trainlm'};
load('Rs_epochs.mat');
figure;
subplot(2,1,1); plot(2:3:100, fliplr(Rs_epochs), 'linewidth',2); xlabel('epochs'); ylabel('Correlation coefficient'); legend(fliplr(algs)); title('Performance'); ylim([0,1]);
subplot(2,1,2); plot(2:3:100, fliplr(times_epochs), 'linewidth',2); xlabel('epochs'); ylabel('training time [s]'); legend(fliplr(algs)); title('Time'); ylim([0,1]);

load('Rs_noise.mat');
figure;
subplot(2,2,1); plot(0.1:0.02:0.9,fliplr(Rs_noise_10ep),'linewidth',2); xlabel('noise rate'); ylabel('Correlation coefficient'); legend(fliplr(algs)); title('10 epochs and 190 points'); ylim([0,1]);
subplot(2,2,2); plot(0.1:0.02:0.9,fliplr(Rs_noise_40ep),'linewidth',2); xlabel('noise rate'); ylabel('Correlation coefficient'); legend(fliplr(algs)); title('40 epochs and 190 points'); ylim([0,1]);
subplot(2,2,3); plot(0.1:0.02:0.9,fliplr(Rs_noise_10ep75p),'linewidth',2); xlabel('noise rate'); ylabel('Correlation coefficient'); legend(fliplr(algs)); title('10 epochs and 75 points'); ylim([0,1]);
subplot(2,2,4); plot(0.1:0.02:0.9,fliplr(Rs_noise_40ep75p),'linewidth',2); xlabel('noise rate'); ylabel('Correlation coefficient'); legend(fliplr(algs)); title('40 epochs and 75 points'); ylim([0,1]);

load('Rs_bayes.mat');
algs = {'traingd', 'traingda', 'traincgf', 'traincgp', 'trainbfg', 'trainlm', 'trainbr'};
figure;
% subplot(2,1,1); plot(2:3:100, Rs_bayes_epochs, 2:3:100, Rs_bayes_epochs_500, 'linewidth', 2);
subplot(2,1,2); plot(0:0.02:0.9,fliplr(Rs_bayes_noise),'linewidth',2); xlabel('noise rate'); ylabel('Correlation coefficient'); legend(fliplr(algs)); title('B) Bayesian learning with noise'); ylim([0,1]);

%% personal regression example
load('data_personal_regression_problem.mat');
% r0666420
Tnew = (6 * T1 + 6 * T2 + 6 * T3 + 4 * T4 + 2 * T5) / (6 + 6 + 6 + 4 + 2);
scatter3(X1, X2, Tnew, '.');
X1 = reshape(X1, [17, 800]);
X2 = reshape(X2, [17, 800]);
Tnew = reshape(Tnew, [17, 800]);

tmp1 = reshape(X1(2, :), [4 200]);
tmp2 = reshape(X2(2, :), [4 200]);
tmp3 = reshape(Tnew(2, :), [4 200]);
trainSet = [X1(1, :), tmp1(1, :); X2(1, :), tmp2(1, :); Tnew(1, :), tmp3(1, :)];
tmp1 = reshape(X1(4, :), [4 200]);
tmp2 = reshape(X2(4, :), [4 200]);
tmp3 = reshape(Tnew(4, :), [4 200]);
valSet = [X1(3, :), tmp1(1, :); X2(3, :), tmp2(1, :); Tnew(3, :), tmp3(1, :)];
tmp1 = reshape(X1(6, :), [4 200]);
tmp2 = reshape(X2(6, :), [4 200]);
tmp3 = reshape(Tnew(6, :), [4 200]);
testSet = [X1(5, :), tmp1(1, :); X2(5, :), tmp2(1, :); Tnew(5, :), tmp3(1, :)];

figure
scatter3(trainSet(1, :), trainSet(2, :), trainSet(3, :), '.');
hold on
scatter3(valSet(1, :), valSet(2, :), valSet(3, :), '.');
scatter3(testSet(1, :), testSet(2, :), testSet(3, :), '.');
hold off

%% Define network
% p = [trainSet(1:2,:),valSet(1:2,:),testSet(1:2,:)];
% t = [trainSet(3,:),valSet(3,:),testSet(3,:)];
% net.divideFcn = 'divideind';
% net.divideParam.trainInd = 1:1000
% net.divideParam.valInd = 1001:2000;
% net.divideParam.testInd = 2001:3000;
load('data_personal_regression_problem.mat');
Tnew = (6 * T1 + 6 * T2 + 6 * T3 + 4 * T4 + 2 * T5) / (6 + 6 + 6 + 4 + 2);
p = randperm(length(Tnew)); Tnew = Tnew(p); X1 = X1(p); X2 = X2(p);
[trInd,valInd,testInd] = dividerand(length(Tnew),1000 / length(Tnew), 1000 / length(Tnew), 1000 / length(Tnew));
trInd = trInd(1:1000); valInd = valInd(1:1000); testInd = testInd(1:1000);
figure
scatter3(X1(trInd), X2(trInd), Tnew(trInd), '.');
hold on
scatter3(X1(valInd), X2(valInd), Tnew(valInd), '.');
scatter3(X1(testInd), X2(testInd), Tnew(testInd), '.');
hold off
p = [[X1(trInd)';X2(trInd)'],[X1(valInd)';X2(valInd)'],[X1(testInd)';X2(testInd)']];
t = [Tnew(trInd)',Tnew(valInd)',Tnew(testInd)'];
save('bk_pers_reg_ind', 'trInd','valInd', 'testInd');

neurons = 10:10:100;
algs = {'traingd', 'traingda', 'traincgf', 'traincgp', 'trainbfg', 'trainlm'};
res_epochs = zeros(length(neurons), length(algs));
res_valmses = zeros(length(neurons), length(algs));
res_trainmses = zeros(length(neurons), length(algs));
% structures = {[30 30],[50 50],[80 80],[80, 50], [50,20], [30 30 30], [50 50 50], [80 50 20], [50 20 5]};
% res_epochs = zeros(length(structures),1);
% res_valmses = zeros(length(structures),1);
% res_trainmses = zeros(length(structures),1);

for it = 1:10
for i = 1:length(neurons)
    for j= 1:length(algs)
    net = feedforwardnet(neurons(i), char(algs(j)));
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = 1:length(trInd);
    net.divideParam.valInd = (length(trInd)+1):(length(trInd)+length(valInd));
    net.divideParam.testInd = (length(trInd)+length(valInd)+1):(length(trInd)+length(valInd)+length(testInd));
    net.trainParam.epochs = 200;
    [net, tmp] = train(net,p,t);


    res_epochs(i,j) = ((it-1)*res_epochs(i,j) + tmp.num_epochs) / it;
    res_valmses(i,j) = ((it-1)*res_valmses(i,j) + tmp.best_vperf) / it;
    res_trainmses(i,j) = ((it-1)*res_trainmses(i,j) + tmp. best_perf) / it;
    fprintf('%d - %s - %s\n', it,mat2str(neurons(i)), char(algs(j)));
    save('backup_pers_reg', 'res_epochs','res_valmses', 'res_trainmses');
end
end
end

%% Best model surface
load('personal_regression_indices');
net = feedforwardnet('best', 'trainlm');
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:length(trInd);
net.divideParam.valInd = (length(trInd)+1):(length(trInd)+length(valInd));
net.divideParam.testInd = (length(trInd)+length(valInd)+1):(length(trInd)+length(valInd)+length(testInd));
net.trainParam.epochs = 200;
[net, tmp] = train(net,p,t, 'useparallel', 'yes');
out = sim(net,[X1(testInd)';X2(testInd)']);
%%
TFs = {'tansig', 'logsig', 'radbas'};
algs = {'traingd', 'traingda', 'traincgf', 'traincgp', 'trainbfg', 'trainlm'};
load('personal_regression_run_reduced.mat');
% res_testmses = permute(res_testmses, [1,3,2]);
% res_epochs = permute(res_epochs, [1,3,2]);
figure;
semilogy(10:10:100, res_testmses(:,:,1), 'linewidth', 2,'marker', '+'); xlabel('neurons');ylabel('validation set MSE'); legend(algs); title('Networks trained with tansig');
%  text(10:10:100, res_testmses(:,1,6), num2str(res_epochs(:,1,6)))

load('personal_regression_run_final_models.mat');
res_valmses(7) = 0;
structures = {'[30 30]','[50 50]','[80 80]','[80, 50]', '[50,20]', '[30 30 30]', '[50 50 50]', '[80 50 20]', '[50 20 5]'};
bar(res_valmses); xticklabels(structures);

figure; subplot(1,3,1); % Original surface
T = delaunay(X1,X2);
TO = triangulation(T,X1,X2,Tnew);
trisurf(TO, 'EdgeColor', 'none'); hold on;
scatter3(X1(testInd), X2(testInd), Tnew(testInd), '.r');

subplot(1,3,2); % Best model surface
T = delaunay(X1(testInd),X2(testInd));
TO = triangulation(T,X1(testInd),X2(testInd),out);
trisurf(TO, 'EdgeColor', 'none'); hold on;
scatter3(X1(testInd), X2(testInd), Tnew(testInd), '.r');

subplot(1,3,3); % Error surface
T = delaunay(X1(testInd),X2(testInd));
TO = triangulation(T,X1(testInd),X2(testInd),out-Tnew(testInd));
trisurf(TO, 'EdgeColor', 'none');
% figure(1); 
% hold on;
% 
% %First x value
% xval = 1; 
% h = bar3(10:10:100, res_testmses(:,3:6,xval),'grouped');
% Xdat = get(h,'Xdata');
% for ii=1:length(Xdat)
%     Xdat{ii}=Xdat{ii}+(xval-1)*ones(size(Xdat{ii}));
%     set(h(ii),'XData',Xdat{ii});
% end
% %Second x value
% xval = 2;
% h = bar3(10:10:100, res_testmses(:,3:6,xval),'grouped');
% Xdat = get(h,'Xdata');
% for ii=1:length(Xdat)
%     Xdat{ii}=Xdat{ii}+(xval-1)*ones(size(Xdat{ii}));
%     set(h(ii),'XData',Xdat{ii});
% end
% %third x value
% xval = 3; 
% h = bar3(10:10:100, res_testmses(:,3:6,xval),'grouped');
% Xdat = get(h,'Xdata');
% for ii=1:length(Xdat)
%     Xdat{ii}=Xdat{ii}+(xval-1)*ones(size(Xdat{ii}));
%     set(h(ii),'XData',Xdat{ii});
% end
% 
% % xlim([0 3]);
% view(3);
% title('Grouped Style')
% xlabel('x');
% ylabel('y');
% zlabel('z');
