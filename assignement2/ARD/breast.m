load('breast.mat');

%% ARD
load('ard_alpha');
rel = [relevance_2,relevance_3,relevance_4,relevance_5,relevance_6,relevance_7]; 
semilogy(rel);
keep = 17;
[s2,ind2] = sort(relevance_2); ind2 = sort(ind2(1:keep));
[s3,ind3] = sort(relevance_3);ind3 = sort(ind3(1:keep));
[s4,ind4] = sort(relevance_4);ind4 = sort(ind4(1:keep));
[s5,ind5] = sort(relevance_5);ind5 = sort(ind5(1:keep));
[s6,ind6] = sort(relevance_6);ind6 = sort(ind6(1:keep));

combined = [];
for i = 1:30
   if (~isempty(find(ind2==i,1)) && ~isempty(find(ind3==i,1)) && ~isempty(find(ind4==i,1)) && ~isempty(find(ind5==i,1)) && ~isempty(find(ind6==i,1)))
       combined = [combined;i];
   end
end

trainset = trainset(:,combined);
labels_train = labels_train(:,combined);
testset = trainset(:,combined);
labels_test = labels_train(:,combined);
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
pl = mean(performance,1); semilogy(1:5,pl(:));
load('breast_performance_traincgp__1_1_7__ 20__10_20_50.mat')
pl = mean(performance,1); semilogy(1:7,pl(:));
load('breast_performance_trainlm__1_1_5__ 20__10_10_50.mat')
pl = performance(1,:,:); semilogy(1:5,pl(:));
legend('trainbr', 'traincgp', 'traincgp', 'trainlm');