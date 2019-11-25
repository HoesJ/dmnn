load('breast.mat');

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

%% Network
layers = 1:1:5;
neurons = 10:10:50;
performance = zeros(length(neurons), length(layers));
S = 10;
iteration = 1;
for it = 1:iteration
    for i = 1:length(layers)
        for j = 1:length(neurons)
            err = 0;
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
                ttest = con2seq(transpose(ttest));
                                
                topology = ones(1,layers(i))*neurons(j);
                alg = 'trainbr';
                net = feedforwardnet(topology, alg);
%                 net = patternnet(topology, alg);
                net.trainParam.epochs = 50;
                net = train(net, ptr, ttr);
                
                out = sim(net, ptest);
                err = err + mse(net, ttest, out);
                
                fprintf('The MSE of neurons %d, layers %d at %d is %f \n', neurons(j), layers(i),k, err);
            end
            performance(j,i) = err / S;            
        end
    end
end