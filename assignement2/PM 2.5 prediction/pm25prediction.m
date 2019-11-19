load('shanghai2017.mat');
preprocessing

iteration = 10;
lags = 25:5:100;
neurons = 10:10:50;
layers = 1:1:5;

errors = zeros(length(lags), length(neurons), length(layers));
times = zeros(length(lags), length(neurons), length(layers));

test = con2seq(Xpred);

for it = 1:iteration
    for i = 1:length(layers)
        for j = 1:length(neurons)
            for k = 1:length(lags)
                [Xtr, Ytr] = getTimeSeriesTrainData(Xtrain, lags(k));
                
                topology = ones(1,layers(i))*neurons(j);
                alg = 'traincgf';
                [net, time] = trainModel(topology, alg, Xtr,Ytr);
                [err, ~] = evalModel(net, Xtr(:,end), Xpred);
                
                times(k,j,i) = (times(k,j,i) * (it-1) + time) / it;
                errors(k,j,i) = (errors(k,j,i) * (it-1) + err) / it;
                fprintf('The MSE of lag %d and neurons %d, layers %d is %f \n', lags(k), neurons(j), layers(i), err);
            end
        end
    end
end

%%
% 
% [Xtr, Ytr] = getTimeSeriesTrainData(Xtrain, 50);
% % p = con2seq(Xtr);
% % t = con2seq(Ytr);
% test = con2seq(Xpred);
% 
% topology = ones(1,2)*30;
% alg = 'traincgf';
% pc = figure;    
% net = feedforwardnet(topology, alg);
%     net.trainParam.epochs = 1000;
%     net = train(net, Xtr, Ytr, 'useParallel', 'yes');
% [err, ~] = evalModel(net, Xtr(:,end), Xpred, pc);
