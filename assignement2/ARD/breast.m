%%
% result = zeros(30,1);
%%
for iter = 27:27
%% ARD
s = 0; keep = 30;
while s ~= iter
load('breast.mat');
load('ard_alpha_normalized');
% rel = [relevance_2,relevance_3,relevance_4,relevance_5,relevance_6,relevance_7]; 
% semilogy(rel);
semilogy(relevance);

[s2,ind2] = sort(relevance(:,1)); ind2 = sort(ind2(1:keep));
[s3,ind3] = sort(relevance(:,2));ind3 = sort(ind3(1:keep));
[s4,ind4] = sort(relevance(:,3));ind4 = sort(ind4(1:keep));
[s5,ind5] = sort(relevance(:,4));ind5 = sort(ind5(1:keep));
[s6,ind6] = sort(relevance(:,5));ind6 = sort(ind6(1:keep));

combined = [];
for i = 1:30
   if (~isempty(find(ind2==i,1)) && ~isempty(find(ind3==i,1)) && ~isempty(find(ind4==i,1)) && ~isempty(find(ind5==i,1)) && ~isempty(find(ind6==i,1)))
       combined = [combined;i];
   end
end

trainset = trainset(:,combined);
testset = testset(:,combined);
keep = keep - 1
s = length(combined)
end
%%

ber = 0;
for it = 1:10
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


end

%% Plot
load('ard_res.mat');
ind = 1:30;
plot(ind(result~=0), result(result~=0),'linewidth', 2, 'Marker', '+'); xlabel('Input dimension'); ylabel('FICS'); ylim([0 0.2]); title('ARD with [50 50 50]');
hold on
load('ard_res_normalized.mat');
ind = 1:30;
plot(ind(result~=0), result(result~=0),'linewidth', 2, 'Marker', '+');
legend('unnormalized', 'normalized');