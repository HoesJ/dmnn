load('breast.mat');
% The plan:
%   1. Run multiple layer test with 20 neurons trainbr
%   2. Run same test with trainlm and traincgp
%   3. Optimize best layer + best number of neurons
%   4. Select model en test on test set
%% Visualize
dataset = [trainset;testset];
labels = [labels_train;labels_test];
positives = reshape(dataset(repmat(labels,1,30) == 1), [], 30);
negatives = reshape(dataset(repmat(labels,1,30) == -1), [], 30);

bins = 15;
figure
for i = 1:size(positives,2)
    min = 0;
    m = max(max(positives(:,i)),max(negatives(:,i)));
    edges = linspace(min,m,bins);
    
    subplot(5,6,i);
    histogram(positives(:,i),edges);
    hold on;
    histogram(negatives(:,i),edges);
    hold off;
end
%% Visualize
data = testset; labels = labels_test;
co = zeros(size(data,1),size(data,2),3);
co(labels == 1,:,1) = 0.9; co(labels == -1,:,1) = 0.3;
co(labels == 1,:,2) = 0.3; co(labels == -1,:,2) = 0.9;
co(labels == 1,:,3) = 0.3; co(labels == -1,:,3) = 0.3;
% 
% co = zeros(size(data,1),size(data,2));
% co(labels == 1,:) = [0.9,0.3,0.3]; co(labels == -1,:) = [0.3,0.9,0.3];

fs = 25; tfso = 5;
figure; %subplot(1,2,1);
surf(data, co); xlabel('Feature', 'Fontsize', fs); ylabel('Instance','Fontsize', fs); zlabel('linear','Fontsize', fs); title('Breast Cancer test set','Fontsize', fs + tfso);
% subplot(1,2,2);
figure;
surf(log(data),co);  xlabel('Feature','Fontsize', fs); ylabel('Instance','Fontsize', fs); zlabel('log','Fontsize', fs);title('Breast Cancer test set','Fontsize', fs + tfso);

%% Network
epstep = 10;
epochs = 10:epstep:50;

layers = 3;
% neurons = 10:10:50;
neurons = 10:10:50;
performance = zeros(length(epochs),length(neurons), length(layers));
S = 10;
iteration = 5;
for it = 1:iteration
    for i = 1:length(layers)
        for j = 1:length(neurons)
            ber = zeros(length(epochs),1);
            for k = 1:S          
                testsize = size(trainset,1) / S;
                ptest = trainset(1:testsize,:);
                ttest = labels_train(1:testsize,:);
                
                ptr = trainset(testsize+1:end,:);
                ttr = labels_train(testsize+1:end,:);    
                
                trainset = [ptr;ptest];
                labels_train = [ttr;ttest];
                
                ptr = con2seq(transpose(ptr));
                ttr = con2seq(transpose(ttr));
                ptest = con2seq(transpose(ptest));
                ttest = (transpose(ttest));
                                
                topology = ones(1,layers(i))*neurons(j);
                alg = 'trainlm';
                net = patternnet(topology, alg);
                net.performFcn = 'mse';
                net.layers{end}.transferFcn = 'tansig';
                net.trainParam.epochs = epochs(1);
                net.trainParam.showWindow = 0;
                for n = 1:length(epochs)
                    net = train(net, ptr, ttr);
                    out = cell2mat(sim(net, ptest));
                    out(out>0) = 1;
                    out(out<=0) = -1;
                    ber(n) = ber(n) + sum(abs(ttest-out))/(2*length(out));
                    net.trainParam.epochs = epstep;
                end
                fprintf('%d --> The MSE of neurons %d, layers %d at %d \n', it, neurons(j), layers(i),k);
%                 disp(ber);
            end
            performance(:,j,i) = ber ./ S;            
        end
    end
end
%% Initial tests
figure;
load('breast_performance_trainbr__1_1_5__ 20__10_15_70.mat');
pl = performance(1,:,:); semilogy(1:5,pl(:)); hold on;
load('breast_performance_traincgp__1_1_5__ 20__10_20_50.mat')
% pl = mean(performance,1); semilogy(1:5,pl(:));
load('breast_performance_traincgp__1_1_7__ 20__10_20_50.mat')
pl = mean(performance,1); semilogy(1:7,pl(:));
load('breast_performance_trainlm__1_1_5__ 20__10_10_50.mat')
pl = performance(1,:,:); semilogy(1:5,pl(:));
legend('trainbr', 'traincgp', 'trainlm');
title('Breast Cancer: 20 neuron models'); xlabel('layers'); ylabel('FICS');
%% Advanced tests
load('breast_performance_trainlm__3__ 10_10_50__10_10_50.mat');
perf_neur = performance;
load('breast_performance_trainlm__3__ models__10_10_50.mat');
perf_adv = performance;
performance = [perf_neur, perf_adv];

mins = zeros(size(performance,2),1);
for i = 1:size(performance,2)
    mins(i) = min(performance(:,i));
end
bar(mins); xticklabels({'[10,10,10]','[20,20,20]','[30,30,30]','[40,40,40]','[50,50,50]','[50 30 20]','[30 20 10]','[40 20 5]'});text(0.9:7.9,mins+0.005,['10';'10';'10';'30';'10';'10';'10';'10']);
ylabel('FICS'); title('Breast Cancer: classification error'); xtickangle(45);
%% Final model on test set
ber = 0;
for it = 1:20
ptr = con2seq(trainset'); ttr = con2seq(labels_train'); ptest = con2seq(testset'); ttest = labels_test';
net = patternnet([50 50 50], 'trainlm');
net.performFcn = 'mse';
net.layers{end}.transferFcn = 'tansig';
net.trainParam.epochs = 10;
net.trainParam.showWindow = 1;
net = train(net, ptr, ttr);
out = cell2mat(sim(net, ptest));
out(out>0) = 1;
out(out<=0) = -1;
ber = ber + sum(abs(ttest-out))/(2*length(out));
end
ber = ber / 20