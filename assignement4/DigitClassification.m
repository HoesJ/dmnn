% clear all
% close all
% nntraintool('close');
% nnet.guis.closeAllViews();

% Neural networks have weights randomly initialized before training.
% Therefore the results from training are different each time. To avoid
% this behavior, explicitly set the random number generator seed.
% rng('default')

structures = {[50],[75],[100],[150],[100,50],[100,75],[75,50],[100,75,50]};
sparse=[ {0.4,0.3,0.2,0.1,0.05};
         {[0.3,0.3],[0.15,0.15],[0.3,0.1],[0.4,0.1],[0.2,0.05]};
         {[0.4,0.2,0.1], [0.2,0.1,0.05],[0.1,0.1,0.05],[0.2,0.2,0.2],[0.1,0.1,0.1]}];
% preClassErrs = zeros(8,5);
% postClassErrs = zeros(8,5);
%%
times = zeros(5,1);
for it = 1:10
for i = 8:8
for j = 4:4
load('digittrain_dataset');
layers = cell2mat(structures(i));

feat = xTrainImages;

for n = 1:length(layers)
    hiddenSize = layers(n);
    sp = cell2mat(sparse(length(layers),j));
    sp = sp(n);

    autoenc = trainAutoencoder(feat,hiddenSize, ...
        'MaxEpochs',400, ...
        'L2WeightRegularization',0.004, ...
        'SparsityRegularization',4, ...
        'SparsityProportion',sp, ...
        'ScaleData', false);
    feat = encode(autoenc,feat);
    
    figure;
    plotWeights(autoenc);
    
    if n == 1
        deepnet = autoenc;
    else
        deepnet = stack(deepnet,autoenc);
    end
end

softnet = trainSoftmaxLayer(feat,tTrain,'MaxEpochs',400);
deepnet = stack(deepnet, softnet);

% Test deep net
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;
load('digittest_dataset');
xTest = zeros(inputSize,numel(xTestImages));
for p = 1:numel(xTestImages)
    xTest(:,p) = xTestImages{p}(:);
end
y = deepnet(xTest);
preClassErrs(i,j) = preClassErrs(i,j) + 100*(1-confusion(tTest,y));

% Test fine-tuned deep net
xTrain = zeros(inputSize,numel(xTrainImages));
for p = 1:numel(xTrainImages)
    xTrain(:,p) = xTrainImages{p}(:);
end
tic;
deepnet = train(deepnet,xTrain,tTrain);
times(j) = times(j) + toc
y = deepnet(xTest);
postClassErrs(i,j) = postClassErrs(i,j) + 100*(1-confusion(tTest,y));

save('bk-classerr', 'postClassErrs', 'preClassErrs');
end
end
end

%%
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;
xTest = zeros(inputSize,numel(xTestImages));
for p = 1:numel(xTestImages)
    xTest(:,p) = xTestImages{p}(:);
end
xTrain = zeros(inputSize,numel(xTrainImages));
for p = 1:numel(xTrainImages)
    xTrain(:,p) = xTrainImages{p}(:);
end

classAcc= zeros(3,1);
for it = 1:10
%Compare with normal neural network (1 hidden layers)
net = patternnet(100);
net=train(net,xTrain,tTrain);
y=net(xTest);
classAcc(1)=classAcc(1) + 100*(1-confusion(tTest,y))


% %Compare with normal neural network (2 hidden layers)
net = patternnet([100,100]);
net=train(net,xTrain,tTrain);
y=net(xTest);
classAcc(2)=classAcc(2) + 100*(1-confusion(tTest,y))


% %Compare with normal neural network (3 hidden layers)
net = patternnet([100,100, 100]);
net=train(net,xTrain,tTrain);
y=net(xTest);
classAcc(3)=classAcc(3) + 100*(1-confusion(tTest,y))

end

%% PLOT
load('classErr.mat');

figure;
subplot(1,2,1);plot([0.4,0.3,0.2,0.1,0.05],preClassErrs(1:4,:),'linewidth', 2, 'Marker','+'); xlabel('Sparsity proportion'); ylabel('Correct classificatio [%]'); title('MNIST - 1 layer - Pre fine-tuning'); legend('[50]','[75]','[100]','[150]');
subplot(1,2,2);plot([0.4,0.3,0.2,0.1,0.05],postClassErrs(1:4,:),'linewidth', 2, 'Marker','+'); xlabel('Sparsity proportion'); ylabel('Correct classificatio [%]'); title('MNIST - 1 layer - Post fine-tuning'); legend('[50]','[75]','[100]','[150]');

figure;
subplot(1,2,1);plot(preClassErrs(5:7,:)', 'linewidth',2,'Marker','+');  xlabel('Sparsity proportion'); ylabel('Correct classificatio [%]'); title('MNIST - 2 layer - Pre fine-tuning'); legend('[100,50]','[100,75]','[75,50]');
xticks([1,2,3,4,5]); xticklabels({'[0.3,0.3]','[0.15,0.15]','[0.3,0.1]','[0.4,0.1]','[0.2,0.05]'}); xtickangle(45);
subplot(1,2,2);plot(postClassErrs(5:7,:)', 'linewidth',2,'Marker','+');  xlabel('Sparsity proportion'); ylabel('Correct classificatio [%]'); title('MNIST - 2 layer - Post fine-tuning'); legend('[100,50]','[100,75]','[75,50]');
xticks([1,2,3,4,5]); xticklabels({'[0.3,0.3]','[0.15,0.15]','[0.3,0.1]','[0.4,0.1]','[0.2,0.05]'}); xtickangle(45); ylim([97,100]);

figure;
subplot(1,2,1);plot(preClassErrs(8,:)', 'linewidth',2,'Marker','+');  xlabel('Sparsity proportion'); ylabel('Correct classificatio [%]'); title('MNIST - 3 layer - Pre fine-tuning'); legend('[100,75,50]');
xticks([1,2,3,4,5]); xticklabels({'[0.4,0.2,0.1]','[0.2,0.1,0.05]','[0.1,0.1,0.05]','[0.2,0.2,0.2]','[0.1,0.1,0.1]'}); xtickangle(45);
subplot(1,2,2);plot(postClassErrs(8,:)', 'linewidth',2,'Marker','+');  xlabel('Sparsity proportion'); ylabel('Correct classificatio [%]'); title('MNIST - 3 layer - Post fine-tuning'); legend('[100,75,50]');
xticks([1,2,3,4,5]); xticklabels({'[0.4,0.2,0.1]','[0.2,0.1,0.05]','[0.1,0.1,0.05]','[0.2,0.2,0.2]','[0.1,0.1,0.1]'}); xtickangle(45); ylim([97,100]);

