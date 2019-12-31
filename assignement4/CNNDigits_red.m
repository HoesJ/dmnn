digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
        'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
%%
figure;
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(digitData.Files{perm(i)});
end

CountLabel = digitData.countEachLabel;

%%
img = readimage(digitData,1);
size(img)

%%
trainingNumFiles = 750;
% rng(1) % For reproducibility
[trainDigitData,testDigitData] = splitEachLabel(digitData, ...
				trainingNumFiles,'randomize'); 

%%
layers = [imageInputLayer([28 28 1])
  convolution2dLayer(5,12)
  reluLayer
  
  maxPooling2dLayer(2,'Stride',2)
  
  convolution2dLayer(5,24)
  reluLayer  
  
  fullyConnectedLayer(10)
  softmaxLayer
  classificationLayer()]; %+-10min
      
%% Specify the Training Options
options = trainingOptions('sgdm','MaxEpochs',15, ...
	'InitialLearnRate',0.0001);%,'OutputFcn',@plotTrainingAccuracy);  

%% Train the Network Using Training Data
tic
convnet = trainNetwork(trainDigitData,layers,options);
toc

%% Classify the Images in the Test Data and Compute Accuracy
YTest = classify(convnet,testDigitData);
TTest = testDigitData.Labels;

%% 
% Calculate the accuracy. 
accuracy = sum(YTest == TTest)/numel(TTest)   
