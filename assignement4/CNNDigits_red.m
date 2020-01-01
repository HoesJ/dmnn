digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
        'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    
eps = 5:5:20;
lns = [0.00001,0.0001, 0.001, 0.005];
%%
% figure;
% perm = randperm(10000,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(digitData.Files{perm(i)});
% end
% 
% CountLabel = digitData.countEachLabel;

%%
% img = readimage(digitData,1);
% size(img)

%%
trainingNumFiles = 750;
% rng(1) % For reproducibility
[trainDigitData,testDigitData] = splitEachLabel(digitData, ...
				trainingNumFiles,'randomize'); 

%%
layers = {
    [   % Default
        imageInputLayer([28 28 1]); % 28x28x1
        convolution2dLayer(5,12);   % 24x24x12
        reluLayer;
        maxPooling2dLayer(2,'Stride',2);  % 12x12x12
        convolution2dLayer(5,24);  % 8x8x24
        reluLayer;
        fullyConnectedLayer(10);
        softmaxLayer;
        classificationLayer();
    ],[
        imageInputLayer([28 28 1]); % 28x28x1
        convolution2dLayer(5,12);   % 24x24x12
        reluLayer;
        maxPooling2dLayer(2,'Stride',2);  % 12x12x12
        fullyConnectedLayer(10);
        softmaxLayer;
        classificationLayer();
    ],[
        imageInputLayer([28 28 1]); % 28x28x1
        convolution2dLayer(11,22);   % 18x18x22
        reluLayer;
        maxPooling2dLayer(2,'Stride',2);  % 9x9x22
        fullyConnectedLayer(10);
        softmaxLayer;
        classificationLayer();
    ],[
        imageInputLayer([28 28 1]); % 28x28x1
        convolution2dLayer(15,36);   % 14x14x36
        reluLayer;
        maxPooling2dLayer(2,'Stride',2);  % 7x7x36
        fullyConnectedLayer(10);
        softmaxLayer;
        classificationLayer();
    ],[   % Effect of MaxPool
        imageInputLayer([28 28 1]); % 28x28x1
        convolution2dLayer(5,12);   % 24x24x12
        reluLayer;
        maxPooling2dLayer(4,'Stride',2);  % 10x10x12
        convolution2dLayer(5,24);  % 6x6x24
        reluLayer;
        fullyConnectedLayer(10);
        softmaxLayer;
        classificationLayer();
    ],[   % Bigger conv masks
        imageInputLayer([28 28 1]); % 28x28x1
        convolution2dLayer(9,18);   % 20x20x18
        reluLayer;
        maxPooling2dLayer(2,'Stride',2);  % 10x10x18
        convolution2dLayer(9,32);  % 6x6x32
        reluLayer;
        fullyConnectedLayer(10);
        softmaxLayer;
        classificationLayer();
    ],[   % Less overlap in masks - no downsample
        imageInputLayer([28 28 1]); % 28x28x1
        convolution2dLayer(7,18, 'Stride', 3);  % 8x8x18
        reluLayer;
        convolution2dLayer(4,32, 'Stride', 3);  % 4x4x32
        reluLayer;
        fullyConnectedLayer(10);
        softmaxLayer;
        classificationLayer();
    ],[   % 3 layers
        imageInputLayer([28 28 1]); % 28x28x1
        convolution2dLayer(5,12);  % 24x24x12
        reluLayer;
        convolution2dLayer(6,24, 'Stride', 2);  % 10x10x24
        reluLayer;
        convolution2dLayer(4,36);  % 7x7x36
        reluLayer;
        fullyConnectedLayer(10);
        softmaxLayer;
        classificationLayer();
    ],[   % 3 layers - with downsample instead of stride
        imageInputLayer([28 28 1]); % 28x28x1
        convolution2dLayer(5,12);  % 24x24x12
        reluLayer;
        convolution2dLayer(5,24);  % 20x20x24
        reluLayer;
        maxPooling2dLayer(2,'Stride',2);  % 10x10x24
        convolution2dLayer(4,36);  % 7x7x36
        reluLayer;
        fullyConnectedLayer(10);
        softmaxLayer;
        classificationLayer();
    ]
  };

%%
res = cell(1,length(layers));
for i = 1:length(layers)
    tmp_nfos = cell(length(eps), length(lns));
    tmp_acc = zeros(length(eps),length(lns));
    tmp_times = zeros(length(eps), length(lns));
    for k = 1:length(eps)
        for l = 1:length(lns)
            fprintf('exp %i: eps %i, ln %i\n',i,eps(k), lns(l));
            if (lns(l) == 0.01)
                options = trainingOptions('sgdm','MaxEpochs',eps(k),'InitialLearnRate',lns(l), 'LearnRateDropFactor',0.5,'LearnRateDropPeriod',2);
            else
                options = trainingOptions('sgdm','MaxEpochs',eps(k),'InitialLearnRate',lns(l));
            end

            tic
            [convnet1, nfo] = trainNetwork(trainDigitData,layers{i},options);
            t = toc
            tmp_times(k,l) = t;
            tmp_nfos{k,l} = nfo;

            % Classify the Images in the Test Data and Compute Accuracy
            YTest = classify(convnet1,testDigitData);
            TTest = testDigitData.Labels;
            accuracy = sum(YTest == TTest)/numel(TTest)   
            tmp_acc(k,l) = accuracy;
        end
    end
    res{i} = {tmp_nfos, tmp_acc, tmp_times};
    save('bk-convexp','res');
end

% res{<experiment>}{<nfo,acc,time>}
% res{ <exp> }{ <nfo> }{ <ep>,<ln> }{ <loss,acc,learnrate> }
