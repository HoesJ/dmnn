%%
result = zeros(30,1);
%%
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
testset = testset(:,combined);
%%
s = length(combined);
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
ber = ber / 20;
result(s) = ber;
save('res_backup', 'result');